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
from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking import utils
from dipy.core.gradients import gradient_table
from dipy.data import small_sphere
#from dipy.direction import BootDirectionGetter
from dipy.reconst.shm import OpdtModel
#from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.reconst import shm
from dipy.data import get_fnames
from dipy.data import read_stanford_pve_maps

import nibabel as nib
from nibabel.orientations import aff2axcodes

# Import custom module
import cuslines.cuslines as cuslines

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
parser.add_argument("--output-prefix", type=str, default ='', help="path to the output file")
parser.add_argument("--chunk-size", type=int, required=True, help="how many seeds to process per sweep, per GPU")
parser.add_argument("--nseeds", type=int, default=None, help="how many seeds to process in total")
parser.add_argument("--ngpus", type=int, required=True, help="number of GPUs to use")
parser.add_argument("--use-fast-write", action='store_true', help="use fast file write")
parser.add_argument("--max-angle", type=float, default=1.0471975511965976, help="default: 60 deg (in rad)")
parser.add_argument("--min-signal", type=float, default=1.0, help="default: 1.0")
parser.add_argument("--tc-threshold", type=float, default=0.1, help="default: 0.1")
parser.add_argument("--step-size", type=float, default=0.5, help="default: 0.5")
args = parser.parse_args()

# Get Stanford HARDI data
hardi_nifti_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

print(hardi_nifti_fname, hardi_bval_fname, hardi_bvec_fname)

csf, gm, wm = read_stanford_pve_maps()
wm_data = wm.get_fdata()

img = get_img(hardi_nifti_fname)
voxel_order = "".join(aff2axcodes(img.affine))

gtab = get_gtab(hardi_bval_fname, hardi_bvec_fname)
#roi = get_img(hardi_nifti_fname)

data = img.get_fdata()
roi_data = (wm_data > 0.5)
mask = roi_data

tenmodel = dti.TensorModel(gtab, fit_method='WLS')
print('Fitting Tensor')
tenfit = tenmodel.fit(data, mask)
print('Computing anisotropy measures (FA,MD,RGB)')
FA = tenfit.fa
FA[np.isnan(FA)] = 0

# Setup tissue_classifier args
#tissue_classifier = ThresholdTissueClassifier(FA, 0.1)
metric_map = np.asarray(FA, 'float64')

# Create seeds for ROI
seed_mask = utils.seeds_from_mask(roi_data, density=3, affine=np.eye(4))
seed_mask = seed_mask[0:args.nseeds]

# Setup model
print('slowadcodf')
sh_order = 6
model = OpdtModel(gtab, sh_order=sh_order, min_signal=1)

# Setup direction getter args
print('Bootstrap direction getter')
#boot_dg = BootDirectionGetter.from_data(data, model, max_angle=60., sphere=small_sphere)
b0s_mask = gtab.b0s_mask
dwi_mask = ~b0s_mask

# get fit_matrix from model
fit_matrix = model._fit_matrix
delta_b, delta_q = fit_matrix

# setup sampling matrix
sphere = small_sphere
theta = sphere.theta
phi = sphere.phi
sampling_matrix, _, _ = shm.real_sym_sh_basis(sh_order, theta, phi)

## from BootPmfGen __init__
# setup H and R matrices
# TODO: figure out how to get H, R matrices from direction getter object
x, y, z = model.gtab.gradients[dwi_mask].T
r, theta, phi = shm.cart2sphere(x, y, z)
B, _, _ = shm.real_sym_sh_basis(sh_order, theta, phi)
H = shm.hat(B)
R = shm.lcr_matrix(H)

# create floating point copy of data
dataf = np.asarray(data, dtype=float)

print('streamline gen')
global_chunk_size = args.chunk_size * args.ngpus
nchunks = (seed_mask.shape[0] + global_chunk_size - 1) // global_chunk_size

#streamline_generator = LocalTracking(boot_dg, tissue_classifier, seed_mask, affine=np.eye(4), step_size=.5)

gpu_tracker = cuslines.GPUTracker(args.max_angle,
                                  args.min_signal,
                                  args.tc_threshold,
                                  args.step_size,
                                  dataf, H, R, delta_b, delta_q,
                                  b0s_mask.astype(np.int32), metric_map, sampling_matrix,
                                  sphere.vertices, sphere.edges.astype(np.int32),
                                  ngpus=args.ngpus, rng_seed=0)
t1 = time.time()
streamline_time = 0
io_time = 0
for idx in range(int(nchunks)):
  # Main streamline computation
  ts = time.time()
  streamlines = gpu_tracker.generate_streamlines(seed_mask[idx*global_chunk_size:(idx+1)*global_chunk_size])
  te = time.time()
  streamline_time += (te-ts)
  print("Generated {} streamlines from {} seeds, time: {} s".format(len(streamlines),
                                                                    seed_mask[idx*global_chunk_size:(idx+1)*global_chunk_size].shape[0],
                                                                    te-ts))

  # Save tracklines file
  if args.output_prefix:
    if args.use_fast_write:
      prefix = "{}.{}_{}".format(args.output_prefix, idx+1, nchunks)
      ts = time.time()
      #gpu_tracker.dump_streamlines(prefix, voxel_order, roi.shape, roi.header.get_zooms(), img.affine)
      gpu_tracker.dump_streamlines(prefix, voxel_order, wm.shape, wm.header.get_zooms(), img.affine)
      te = time.time()
      print("Saved streamlines to {}_*.trk, time {} s".format(prefix, te-ts))
    else:
      fname = "{}.{}_{}.trk".format(args.output_prefix, idx+1, nchunks)
      ts = time.time()
      #save_tractogram(fname, streamlines, img.affine, vox_size=roi.header.get_zooms(), shape=roi_data.shape)
      #save_tractogram(fname, streamlines)
      sft = StatefulTractogram(streamlines, hardi_nifti_fname, Space.VOX)
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
