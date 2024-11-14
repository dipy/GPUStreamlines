/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __GENERATE_STREAMLINES_CUDA_H__
#define __GENERATE_STREAMLINES_CUDA_H__

#include <vector>

#include "globals.h"

void generate_streamlines_cuda_mgpu(const ModelType model_type, const REAL max_angle, const REAL min_signal, const REAL tc_threshold, const REAL step_size,
                                    const REAL relative_peak_thresh, const REAL min_separation_angle,
                                    const int nseeds, const std::vector<REAL*> &seeds_d,
                                    const int dimx, const int dimy, const int dimz, const int dimt,
                                    const std::vector<REAL*> &dataf_d, const std::vector<REAL*> &H_d, const std::vector<REAL*> &R_d,
                                    const int delta_nr,
                                    const std::vector<REAL*> &delta_b_d, const std::vector<REAL*> &delta_q_d,
                                    const std::vector<int*> &b0s_mask_d, const std::vector<REAL*> &metric_map_d,
                                    const int samplm_nr,
                                    const std::vector<REAL*> &sampling_matrix_d,
                                    const std::vector<REAL*> &sphere_vertices_d, const std::vector<int*> &sphere_edges_d, const int nedges,
                                    std::vector<REAL*> &slines_h, std::vector<int*> &slinesLen_h, std::vector<int> &nSlines_h,
                                    const std::vector<int> nSlines_old_h, const int rng_seed, const int rng_offset,
                                    const int ngpus, const std::vector<cudaStream_t> &streams);
#if 1
void write_trk(const char *fname,
               const /*short*/ int *dims,
	       const REAL *voxel_size,
	       const char *voxel_order,
	       const REAL *vox_to_ras,
	       const int nsline,
	       const int *slineLen,
	       const REAL3 *sline);
#else
void write_trk(const int num_threads,
	       const char *fname,
               const /*short*/ int *dims,
	       const REAL *voxel_size,
	       const char *voxel_order,
	       const REAL *vox_to_ras,
	       const int nsline,
	       const int *slineLen,
	       const REAL3 *sline);
#endif
#endif
