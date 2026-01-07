#!/usr/bin/env python

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import random
import time

import numpy as np

import dipy.reconst.dti as dti
from dipy.io import read_bvals_bvecs
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking import utils
from dipy.core.gradients import gradient_table, unique_bvals_magnitude
from dipy.data import default_sphere
from dipy.direction import (
  BootDirectionGetter as cpu_BootDirectionGetter,
  ProbabilisticDirectionGetter as cpu_ProbDirectionGetter,
  PTTDirectionGetter as cpu_PTTDirectionGetter)
from dipy.reconst.shm import OpdtModel, CsaOdfModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.data import get_fnames
from dipy.data import read_stanford_pve_maps

import nibabel as nib
from nibabel.orientations import aff2axcodes

from trx.io import save as save_trx

from cuslines import (
    BootDirectionGetter,
    GPUTracker,
    ProbDirectionGetter,
    PttDirectionGetter,
)

t0 = time.time()

# set seed to get deterministic streamlines
np.random.seed(0)
random.seed(0)

#Get Gradient values
def get_gtab(fbval, fbvec):
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    return gtab

def get_img(ep2_seq):
    img = nib.load(ep2_seq)
    return img

print("parsing arguments")
parser = argparse.ArgumentParser()
parser.add_argument("nifti_file", nargs='?', default='hardi', help="path to the DWI nifti file")
parser.add_argument("bvals", nargs='?', default='hardi', help="path to the bvals")
parser.add_argument("bvecs", nargs='?', default='hardi', help="path to the bvecs")
parser.add_argument("mask_nifti", nargs='?', default='hardi', help="path to the mask file")
parser.add_argument("roi_nifti", nargs='?', default='hardi', help="path to the ROI file")
parser.add_argument("--device", type=str, default ='gpu', choices=['cpu', 'gpu'], help="Whether to use cpu or gpu")
parser.add_argument("--output-prefix", type=str, default ='', help="path to the output file")
parser.add_argument("--chunk-size", type=int, default=100000, help="how many seeds to process per sweep, per GPU")
parser.add_argument("--nseeds", type=int, default=100000, help="how many seeds to process in total")
parser.add_argument("--ngpus", type=int, default=1, help="number of GPUs to use if using gpu")
parser.add_argument("--write-method", type=str, default="trk", help="Can be trx or trk")
parser.add_argument("--max-angle", type=float, default=60, help="max angle (in degrees)")
parser.add_argument("--min-signal", type=float, default=1.0, help="default: 1.0")
parser.add_argument("--step-size", type=float, default=0.5, help="default: 0.5")
parser.add_argument("--sh-order",type=int,default=4,help="sh_order")
parser.add_argument("--fa-threshold",type=float,default=0.1,help="FA threshold")
parser.add_argument("--relative-peak-threshold",type=float,default=0.25,help="relative peak threshold")
parser.add_argument("--min-separation-angle",type=float,default=45,help="min separation angle (in degrees)")
parser.add_argument("--sm-lambda",type=float,default=0.006,help="smoothing lambda")
parser.add_argument("--model", type=str, default="opdt", choices=['opdt', 'csa', 'csd'], help="model to use")
parser.add_argument("--dg", type=str, default="boot", choices=['boot', 'prob', 'ptt'], help="direction getting scheme to use")

args = parser.parse_args()

if args.device == "cpu" and args.write_method != "trk":
  print("WARNING: only trk write method is implemented for cpu testing.")
  write_method = "trk"
else:
  write_method = args.write_method

if 'hardi' in [args.nifti_file, args.bvals, args.bvecs, args.mask_nifti, args.roi_nifti]:
  if not all(arg == 'hardi' for arg in [args.nifti_file, args.bvals, args.bvecs, args.mask_nifti, args.roi_nifti]):
    raise ValueError("If any of the arguments is 'hardi', all must be 'hardi'")
  # Get Stanford HARDI data
  hardi_nifti_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(
     name='stanford_hardi')
  csf, gm, wm = read_stanford_pve_maps()
  wm_data = wm.get_fdata()

  img = get_img(hardi_nifti_fname)
  voxel_order = "".join(aff2axcodes(img.affine))

  gtab = get_gtab(hardi_bval_fname, hardi_bvec_fname)

  data = img.get_fdata()
  roi_data = (wm_data > 0.5)
  mask = roi_data
else:
  img = get_img(args.nifti_file)
  voxel_order = "".join(aff2axcodes(img.affine))
  gtab = get_gtab(args.bvals, args.bvecs)
  roi = get_img(args.roi_nifti)
  mask = get_img(args.mask_nifti)
  data = img.get_fdata()
  roi_data = roi.get_fdata()
  mask = mask.get_fdata()

tenmodel = dti.TensorModel(gtab, fit_method='WLS')
print('Fitting Tensor')
tenfit = tenmodel.fit(data, mask=mask)
print('Computing anisotropy measures (FA,MD,RGB)')
FA = tenfit.fa

# Setup tissue_classifier args
tissue_classifier = ThresholdStoppingCriterion(FA, args.fa_threshold)

# Create seeds for ROI
seed_mask = np.asarray(utils.random_seeds_from_mask(
  roi_data, seeds_count=args.nseeds,
  seed_count_per_voxel=False,
  affine=np.eye(4)))

# Setup model
sphere = default_sphere
if args.model == "opdt":
  if args.device == "cpu":
    model = OpdtModel(gtab, sh_order=args.sh_order, smooth=args.sm_lambda, min_signal=args.min_signal)
    dg = cpu_BootDirectionGetter
  else:
    dg = BootDirectionGetter.from_dipy_opdt(
      gtab,
      sphere,
      sh_order_max=args.sh_order,
      sh_lambda=args.sm_lambda,
      min_signal=args.min_signal)
elif args.model == "csa":
  if args.device == "cpu":
    model = CsaOdfModel(gtab, sh_order=args.sh_order, smooth=args.sm_lambda, min_signal=args.min_signal)
    dg = cpu_BootDirectionGetter
  else:
    dg = BootDirectionGetter.from_dipy_csa(
      gtab,
      sphere,
      sh_order_max=args.sh_order,
      sh_lambda=args.sm_lambda,
      min_signal=args.min_signal)
else:
  print("Running CSD model...")
  unique_bvals = unique_bvals_magnitude(gtab.bvals)
  if len(unique_bvals[unique_bvals > 0]) > 1:
    low_shell_idx = gtab.bvals <= unique_bvals[unique_bvals > 0][0]
    response_gtab = gradient_table( # reinit as single shell for this CSD
      gtab.bvals[low_shell_idx],
      gtab.bvecs[low_shell_idx])
    data = data[..., low_shell_idx]
  else:
    response_gtab = gtab
  response, _ = auto_response_ssst(
    response_gtab,
    data,
    roi_radii=10,
    fa_thr=0.7)
  model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=args.sh_order)
  fit = model.fit(data, mask=(FA >= args.fa_threshold))
  data = fit.odf(sphere).clip(min=0)
  if args.model == "ptt":
      if args.device == "cpu":
          dg = cpu_PTTDirectionGetter()
      else:
        # Set FOD to 0 outside mask for probing
        data[FA < args.fa_threshold, :] = 0
        dg = PttDirectionGetter()
  elif args.model == "prob":
      if args.device == "cpu":
        dg = cpu_ProbDirectionGetter()
      else:
        dg = ProbDirectionGetter()
  else:
      raise ValueError("Unknown model type: {}".format(args.model))

# Setup direction getter args
if args.device == "cpu":
  if args.dg != "boot":
    dg = dg.from_pmf(data, max_angle=args.max_angle, sphere=sphere, relative_peak_threshold=args.relative_peak_threshold, min_separation_angle=args.min_separation_angle)
  else:
    dg = dg.from_data(data, model, max_angle=args.max_angle, sphere=sphere, sh_order=args.sh_order, relative_peak_threshold=args.relative_peak_threshold, min_separation_angle=args.min_separation_angle)

    ts = time.time()
    streamline_generator = LocalTracking(dg, tissue_classifier, seed_mask, affine=np.eye(4), step_size=args.step_size)
    sft = StatefulTractogram(streamline_generator, img, Space.VOX)
    n_sls = len(sft.streamlines)
    te = time.time()
else:
    with GPUTracker(
        dg,
        data,
        FA,
        args.fa_threshold,
        sphere.vertices,
        sphere.edges,
        max_angle=args.max_angle * np.pi/180,
        step_size=args.step_size,
        relative_peak_thresh=args.relative_peak_threshold,
        min_separation_angle=args.min_separation_angle * np.pi/180,
        ngpus=args.ngpus,
        rng_seed=0,
        chunk_size=args.chunk_size
    ) as gpu_tracker:
        ts = time.time()
        if args.output_prefix and write_method == "trx":
            trx_file = gpu_tracker.generate_trx(seed_mask, img)
            n_sls = len(trx_file.streamlines)
        else:
            sft = gpu_tracker.generate_sft(seed_mask, img)
            n_sls = len(sft.streamlines)
        te = time.time()
print("Generated {} streamlines from {} seeds, time: {} s".format(n_sls,
                                                                  seed_mask.shape[0],
                                                                  te-ts))

if args.output_prefix:
  if write_method == "trx":
    fname = "{}.trx".format(args.output_prefix)
    save_trx(trx_file, fname)
  else:
    fname = "{}.trk".format(args.output_prefix)
    save_tractogram(sft, fname)
