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
from dipy.direction import BootDirectionGetter
from dipy.reconst.shm import OpdtModel
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.reconst import shm

import nibabel as nib
from nibabel.orientations import aff2axcodes

import numpy.linalg as npl
from dipy.tracking.streamline import transform_streamlines

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
parser.add_argument("nifti_file", help="path to the ep2multiband sequence nifti file")
parser.add_argument("bvals", help="path to the bvals")
parser.add_argument("bvecs", help="path to the bvecs")
parser.add_argument("mask_nifti", help="path to the mask file")
parser.add_argument("roi_nifti", help="path to the ROI file")
parser.add_argument("--output-prefix", type=str, default ='', help="path to the output file")
parser.add_argument("--chunk-size", type=int, required=True, help="how many seeds to process per sweep, per GPU")
args = parser.parse_args()

img = get_img(args.nifti_file)
voxel_order = "".join(aff2axcodes(img.affine))
gtab = get_gtab(args.bvals, args.bvecs)
roi = get_img(args.roi_nifti)
mask = get_img(args.mask_nifti)
data = img.get_data()
roi_data = roi.get_data()
mask = mask.get_data()

img_affine = img.affine

fa_threshold = 0.1
min_relative_peak = 0.25
min_peak_spacing = 0.7
min_peak_deg = 45
sm_lambda = 0
seed_density = 5

tenmodel = dti.TensorModel(gtab, fit_method='WLS')
print('Fitting Tensor')
tenfit = tenmodel.fit(data, mask)
print('Computing anisotropy measures (FA,MD,RGB)')
FA = tenfit.fa
FA[np.isnan(FA)] = 0

# Setup tissue_classifier args
tissue_classifier = ThresholdStoppingCriterion(FA, fa_threshold)
metric_map = np.asarray(FA, 'float64')

# Create seeds for ROI
seed_mask = utils.seeds_from_mask(roi_data, density=seed_density, affine=img_affine)

# Setup model
print('slowadcodf')
sh_order = 4
model = OpdtModel(gtab, sh_order=sh_order, smooth=sm_lambda, min_signal=1)

# Setup direction getter args
print('Bootstrap direction getter')
boot_dg = BootDirectionGetter.from_data(data, model, max_angle=60., sphere=small_sphere, sh_order=sh_order, relative_peak_threshold=min_relative_peak, min_separation_angle=min_peak_deg)

print('streamline gen')
global_chunk_size = args.chunk_size
nchunks = (seed_mask.shape[0] + global_chunk_size - 1) // global_chunk_size

streamline_generator = LocalTracking(boot_dg, tissue_classifier, seed_mask, affine=img_affine, step_size=.5)

t1 = time.time()
streamline_time = 0
io_time = 0
for idx in range(int(nchunks)):
  # Main streamline computation
  ts = time.time()
  streamlines = [s for s in streamline_generator]
  te = time.time()
  streamline_time += (te-ts)
  print("Generated {} streamlines from {} seeds, time: {} s".format(len(streamlines),
                                                                    seed_mask[idx*global_chunk_size:(idx+1)*global_chunk_size].shape[0],
                                                                    te-ts))

  # Invert streamline affine
  inv_affine = npl.inv(img_affine)
  streamlines = transform_streamlines(streamlines,inv_affine)
  
  
  # Save tracklines file
  if args.output_prefix:
    fname = "{}.{}_{}.trk".format(args.output_prefix, idx+1, nchunks)
    ts = time.time()
    #save_tractogram(fname, streamlines, img.affine, vox_size=roi.header.get_zooms(), shape=roi_data.shape)
    sft = StatefulTractogram(streamlines, args.nifti_file, Space.VOX)
    sft.to_rasmm()
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
