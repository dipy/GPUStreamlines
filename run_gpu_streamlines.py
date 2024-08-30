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
import numpy.linalg as npl

import dipy.reconst.dti as dti
from dipy.io import read_bvals_bvecs
from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking import utils
from dipy.core.gradients import gradient_table, unique_bvals_magnitude
from dipy.data import default_sphere
from dipy.direction import (BootDirectionGetter, ProbabilisticDirectionGetter, PTTDirectionGetter)
from dipy.reconst.shm import OpdtModel, CsaOdfModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.reconst import shm
from dipy.data import get_fnames
from dipy.data import read_stanford_pve_maps

import nibabel as nib
from nibabel.orientations import aff2axcodes

# Import custom module
import cuslines

t0 = time.time()

# set seed to get deterministic streamlines
np.random.seed(0)
random.seed(0)

#Get Gradient values
def get_gtab(fbval, fbvec):
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
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
parser.add_argument("--use-fast-write", action='store_true', help="use fast file write (only if using gpu)")
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

if 'hardi' in [args.nifti_file, args.bvals, args.bvecs, args.mask_nifti, args.roi_nifti]:
  if not all(arg == 'hardi' for arg in [args.nifti_file, args.bvals, args.bvecs, args.mask_nifti, args.roi_nifti]):
    raise ValueError("If any of the arguments is 'hardi', all must be 'hardi'")
  # Get Stanford HARDI data
  hardi_nifti_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
  csf, gm, wm = read_stanford_pve_maps()
  wm_data = wm.get_fdata()

  img = get_img(hardi_nifti_fname)
  voxel_order = "".join(aff2axcodes(img.affine))
  img_affine = img.affine

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
  data = img.get_data()
  roi_data = roi.get_data()
  mask = mask.get_data()

tenmodel = dti.TensorModel(gtab, fit_method='WLS')
print('Fitting Tensor')
tenfit = tenmodel.fit(data, mask)
print('Computing anisotropy measures (FA,MD,RGB)')
FA = tenfit.fa
FA[np.isnan(FA)] = 0

# Setup tissue_classifier args
tissue_classifier = ThresholdStoppingCriterion(FA, args.fa_threshold)
metric_map = np.asarray(FA, 'float64')

# Create seeds for ROI
# seed_mask = utils.seeds_from_mask(roi_data, density=args.sampling_density, affine=np.eye(4))
# seed_mask = seed_mask[0:args.nseeds]
seed_mask = np.asarray(utils.random_seeds_from_mask(
  roi_data, seeds_count=args.nseeds,
  seed_count_per_voxel=False,
  affine=np.eye(4)))

# Setup model
sphere = default_sphere
if args.model == "opdt":
  model_type = cuslines.ModelType.OPDT
  print("Running OPDT model...")
  model = OpdtModel(gtab, sh_order=args.sh_order, smooth=args.sm_lambda, min_signal=args.min_signal)
  fit_matrix = model._fit_matrix
  delta_b, delta_q = fit_matrix
elif args.model == "csa":
  model_type = cuslines.ModelType.CSA
  print("Running CSA model...")
  model = CsaOdfModel(gtab, sh_order=args.sh_order, smooth=args.sm_lambda, min_signal=args.min_signal)
  fit_matrix = model._fit_matrix
  # Unlike OPDT, CSA has a single matrix used for fit_matrix. Populating delta_b and delta_q with necessary values for
  # now.
  delta_b = fit_matrix
  delta_q = fit_matrix
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
  # TODO: we shouldnt have to do this, also for CSA, but we populate delta_b, delta_q.
  # we need to name change delta_b/delta_q and make it possible for them to be None, or something like this
  delta_b = model._X
  delta_q = model.B_reg

if args.dg != "boot":
  if args.dg == "prob":
    model_type = cuslines.ModelType.PROB
    dg = ProbabilisticDirectionGetter
  else:
    model_type = cuslines.ModelType.PTT
    dg = PTTDirectionGetter
  fit = model.fit(data, mask=(metric_map >= args.fa_threshold))
  data = fit.odf(sphere).clip(min=0)
else:
  dg = BootDirectionGetter

global_chunk_size = args.chunk_size

# Setup direction getter args
if args.device == "cpu":
  if args.dg != "boot":
    dg = dg.from_pmf(data, max_angle=args.max_angle, sphere=sphere, relative_peak_threshold=args.relative_peak_threshold, min_separation_angle=args.min_separation_angle)
  else:
    dg = BootDirectionGetter.from_data(data, model, max_angle=args.max_angle, sphere=sphere, sh_order=args.sh_order, relative_peak_threshold=args.relative_peak_threshold, min_separation_angle=args.min_separation_angle)
else:
  # Setup direction getter args
  b0s_mask = gtab.b0s_mask
  dwi_mask = ~b0s_mask

  # setup sampling matrix
  theta = sphere.theta
  phi = sphere.phi
  sampling_matrix, _, _ = shm.real_sym_sh_basis(args.sh_order, theta, phi)

  ## from BootPmfGen __init__
  # setup H and R matrices
  # TODO: figure out how to get H, R matrices from direction getter object
  x, y, z = model.gtab.gradients[dwi_mask].T
  r, theta, phi = shm.cart2sphere(x, y, z)
  B, _, _ = shm.real_sym_sh_basis(args.sh_order, theta, phi)
  H = shm.hat(B)
  R = shm.lcr_matrix(H)

  # create floating point copy of data
  dataf = np.asarray(data, dtype=np.float64)

  gpu_tracker = cuslines.GPUTracker(model_type,
                                    args.max_angle * np.pi/180,
                                    args.min_signal,
                                    args.fa_threshold,
                                    args.step_size,
                                    args.relative_peak_threshold,
                                    args.min_separation_angle * np.pi/180,
                                    dataf.astype(np.float64), H.astype(np.float64), R.astype(np.float64), delta_b.astype(np.float64), delta_q.astype(np.float64),
                                    b0s_mask.astype(np.int32), metric_map.astype(np.float64), sampling_matrix.astype(np.float64),
                                    sphere.vertices.astype(np.float64), sphere.edges.astype(np.int32),
                                    ngpus=args.ngpus, rng_seed=0)

print('streamline gen')
nchunks = (seed_mask.shape[0] + global_chunk_size - 1) // global_chunk_size

t1 = time.time()
streamline_time = 0
io_time = 0
for idx in range(int(nchunks)):
  # Main streamline computation
  ts = time.time()
  if args.device == "cpu":
    streamline_generator = LocalTracking(dg, tissue_classifier, seed_mask[idx*global_chunk_size:(idx+1)*global_chunk_size], affine=np.eye(4), step_size=args.step_size)
    streamlines = [s for s in streamline_generator]
  else:
    streamlines = gpu_tracker.generate_streamlines(seed_mask[idx*global_chunk_size:(idx+1)*global_chunk_size])
  te = time.time()
  streamline_time += (te-ts)
  print("Generated {} streamlines from {} seeds, time: {} s".format(len(streamlines),
                                                                    seed_mask[idx*global_chunk_size:(idx+1)*global_chunk_size].shape[0],
                                                                    te-ts))

  # Save tracklines file
  if args.output_prefix:
    ts = time.time()
    if (args.device == "gpu") and args.use_fast_write:
      fname = "{}.{}_{}".format(args.output_prefix, idx+1, nchunks)
      gpu_tracker.dump_streamlines(fname, voxel_order, wm.shape, wm.header.get_zooms(), img.affine)
    else:
      fname = "{}.{}_{}.trk".format(args.output_prefix, idx+1, nchunks)
      sft = StatefulTractogram(streamlines, args.nifti_file, Space.VOX)
      save_tractogram(sft, fname)
    te = time.time()
    print("Saved streamlines to {}, time {} s".format(fname, te-ts))
    io_time += (te-ts)

t2 = time.time()

print("Completed processing {} seeds.".format(seed_mask.shape[0]))
print("Initialization time: {} sec".format(t1-t0))
print("Streamline generation total time: {} sec".format(t2-t1))
print("\tStreamline processing: {} sec".format(streamline_time))
if args.output_prefix:
  print("\tFile writing: {} sec".format(io_time))
