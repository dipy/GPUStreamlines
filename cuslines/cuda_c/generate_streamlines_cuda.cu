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

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "globals.h"
#include "cuwsort.cuh"
#include "ptt.cuh"

#include "utils.cu"
#include "tracking_helpers.cu"
#include "boot.cu"
#include "ptt.cu"

#define MAX_NUM_DIR (128)

#define NTHR_GEN (128)

#define MAX_DIMS        (8)
#define MAX_STR_LEN     (256)

template<int BDIM_X,
         int BDIM_Y,
         bool IS_START,
         typename REAL_T,
         typename REAL3_T>
__device__ int get_direction_prob_d(curandStatePhilox4_32_10_t *st,
                                    const REAL_T *__restrict__ pmf,
                                    const REAL_T max_angle,
                                    const REAL_T relative_peak_thres,
                                    const REAL_T min_separation_angle,
                                    REAL3_T dir,
                                    const int dimx,
                                    const int dimy,
                                    const int dimz,
                                    const int dimt,
                                    const REAL3_T point,
                                    const REAL3_T *__restrict__ sphere_vertices,
                                    const int2 *__restrict__ sphere_edges,
                                    const int num_edges,
                                    REAL3_T *__restrict__ dirs) {
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
	
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

	const int n32dimt = ((dimt+31)/32)*32;

	extern __shared__ REAL_T __sh[];
        REAL_T *__pmf_data_sh = __sh + tidy*n32dimt;

        // pmf = self.pmf_gen.get_pmf_c(&point[0], pmf)
        __syncwarp(WMASK);
        const int rv = trilinear_interp_d<BDIM_X>(dimx, dimy, dimz, dimt, -1, pmf, point, __pmf_data_sh);
        __syncwarp(WMASK);
        if (rv != 0) {
                return 0;
        }

        // for i in range(_len):
        //     if pmf[i] > max_pmf:
        //         max_pmf = pmf[i]
        // absolute_pmf_threshold = pmf_threshold * max_pmf
        const REAL_T absolpmf_thresh = PMF_THRESHOLD_P * max_d<BDIM_X>(dimt, __pmf_data_sh, REAL_MIN);
        __syncwarp(WMASK);

        // for i in range(_len):
        //     if pmf[i] < absolute_pmf_threshold:
        //         pmf[i] = 0.0
        #pragma unroll
        for(int i = tidx; i < dimt; i += BDIM_X) {
                if (__pmf_data_sh[i] < absolpmf_thresh) {
                        __pmf_data_sh[i] = 0.0;
                }
        }
        __syncwarp(WMASK);

        if (IS_START) {
                int *__shInd = reinterpret_cast<int *>(__sh + BDIM_Y*n32dimt) + tidy*n32dimt;
                return peak_directions_d<BDIM_X,
                                         BDIM_Y>(__pmf_data_sh,
                                                 dirs,
                                                 sphere_vertices,
                                                 sphere_edges,
                                                 num_edges,
                                                 dimt,
                                                 __shInd,
                                                 relative_peak_thres,
                                                 min_separation_angle);
        } else {
                REAL_T __tmp;
                #ifdef DEBUG
                        __syncwarp(WMASK);
                        if (tidx == 0) {
                                printArray("__pmf_data_sh initial", 8, dimt, __pmf_data_sh);
                                printf("absolpmf_thresh %10.8f\n", absolpmf_thresh);
                                printf("--->            dir %10.8f, %10.8f, %10.8f\n", dir.x, dir.y, dir.z);
                                printf("--->            point %10.8f, %10.8f, %10.8f\n", point.x, point.y, point.z);
                        }
                        __syncwarp(WMASK);
                        if (tidx == 15) {
                                printf("absolpmf_thresh %10.8f l15\n", absolpmf_thresh);
                                printf("--->            dir %10.8f, %10.8f, %10.8f l15\n", dir.x, dir.y, dir.z);
                                printf("--->            point %10.8f, %10.8f, %10.8f l15\n", point.x, point.y, point.z);
                        }
                        __syncwarp(WMASK);
                        if (tidx == 31) {
                                printArray("__pmf_data_sh initial l31", 8, dimt, __pmf_data_sh);
                                printf("absolpmf_thresh %10.8f l31\n", absolpmf_thresh);
                                printf("--->            dir %10.8f, %10.8f, %10.8f l31\n", dir.x, dir.y, dir.z);
                                printf("--->            point %10.8f, %10.8f, %10.8f l31\n", point.x, point.y, point.z);
                        }
                        __syncwarp(WMASK);
                #endif

                // // These should not be relevant
                // if norm(&direction[0]) == 0:
                //     return 1
                // normalize(&direction[0])

                // for i in range(_len):
                //         cos_sim = self.vertices[i][0] * direction[0] \
                //                 + self.vertices[i][1] * direction[1] \
                //                 + self.vertices[i][2] * direction[2]
                //         if cos_sim < 0:
                //                 cos_sim = cos_sim * -1
                //         if cos_sim < self.cos_similarity:
                //                 pmf[i] = 0
                const REAL_T cos_similarity = COS(max_angle);

                #pragma unroll
                for(int i = tidx; i < dimt; i += BDIM_X) {
                        const REAL_T dot = dir.x*sphere_vertices[i].x+
                                           dir.y*sphere_vertices[i].y+
                                           dir.z*sphere_vertices[i].z;

                        if (FABS(dot) < cos_similarity) {
                                __pmf_data_sh[i] = 0.0;
                        }
                }
                __syncwarp(WMASK);

                #ifdef DEBUG
                        __syncwarp(WMASK);
                        if (tidx == 0) {
                                printArray("__pmf_data_sh after filtering", 8, dimt, __pmf_data_sh);
                        }
                        __syncwarp(WMASK);
                #endif

                // cumsum(pmf, pmf, _len)
                prefix_sum_sh_d<BDIM_X>(__pmf_data_sh, dimt);

                #ifdef DEBUG
                        __syncwarp(WMASK);
                        if (tidx == 0) {
                                printArray("__pmf_data_sh after cumsum", 8, dimt, __pmf_data_sh);
                        }
                        __syncwarp(WMASK);
                #endif

                // last_cdf = pmf[_len - 1]
                // if last_cdf == 0:
                //         return 1
                REAL_T last_cdf = __pmf_data_sh[dimt - 1];
                if (last_cdf == 0) {
                        return 0;
                }

                // idx = where_to_insert(pmf, random() * last_cdf, _len)
                if (tidx == 0) {
                        __tmp = curand_uniform(st) * last_cdf;
                }
                REAL_T selected_cdf = __shfl_sync(WMASK, __tmp, 0, BDIM_X);
// Both these implementations work
#if 1
                int low = 0;
                int high = dimt - 1;
                while ((high - low) >= BDIM_X) {
                        const int mid = (low + high) / 2;
                        if (__pmf_data_sh[mid] < selected_cdf) {
                                low = mid;
                        } else {
                                high = mid;
                        }
                }
                const bool __ballot = (low+tidx <= high) ? (selected_cdf < __pmf_data_sh[low+tidx]) : 0;
                const int __msk = __ballot_sync(WMASK, __ballot);
                const int indProb = low + __ffs(__msk) - 1;
#else
                int indProb = dimt - 1;
                for (int ii = 0; ii < dimt; ii+=BDIM_X) {
                        int __is_greater = 0;
                        if (ii+tidx < dimt) {
                                __is_greater = selected_cdf < __pmf_data_sh[ii+tidx];
                        }
                        const int __msk = __ballot_sync(WMASK, __is_greater);
                        if (__msk != 0) {
                                indProb = ii + __ffs(__msk) - 1;
                                break;
                        }
                }
#endif

                #ifdef DEBUG
                        __syncwarp(WMASK);
                        if (tidx == 0) {
                                printf("last_cdf %10.8f\n", last_cdf);
                                printf("selected_cdf %10.8f\n", selected_cdf);
                                printf("indProb %i out of %i\n", indProb, dimt);
                        }
                        __syncwarp(WMASK);
                #endif

                // newdir = self.vertices[idx]
                // if (direction[0] * newdir[0]
                //     + direction[1] * newdir[1]
                //     + direction[2] * newdir[2] > 0):
                //     copy_point(&newdir[0], &direction[0])
                // else:
                //     newdir[0] = newdir[0] * -1
                //     newdir[1] = newdir[1] * -1
                //     newdir[2] = newdir[2] * -1
                //     copy_point(&newdir[0], &direction[0])
                if (tidx == 0) {
                        if ((dir.x * sphere_vertices[indProb].x +
                             dir.y * sphere_vertices[indProb].y +
                             dir.z * sphere_vertices[indProb].z) > 0) {
                                *dirs = MAKE_REAL3(sphere_vertices[indProb].x,
                                                   sphere_vertices[indProb].y,
                                                   sphere_vertices[indProb].z);
                        } else {
                                *dirs = MAKE_REAL3(-sphere_vertices[indProb].x,
                                                   -sphere_vertices[indProb].y,
                                                   -sphere_vertices[indProb].z);
                        }
                        // printf("direction addr write %p, slid %i\n", dirs, blockIdx.x*blockDim.y+threadIdx.y);
                }

                #ifdef DEBUG
                        __syncwarp(WMASK);
                        if (tidx == 0) {
                                printf("last_cdf %10.8f\n", last_cdf);
                                printf("selected_cdf %10.8f\n", selected_cdf);
                                printf("indProb %i out of %i\n", indProb, dimt);
                        }
                        __syncwarp(WMASK);
                        if (tidx == 15) {
                                printf("last_cdf %10.8f l15\n", last_cdf);
                                printf("selected_cdf %10.8f l15\n", selected_cdf);
                                printf("indProb %i out of %i l15\n", indProb, dimt);
                        }
                        __syncwarp(WMASK);
                        if (tidx == 31) {
                                printf("last_cdf %10.8f l31\n", last_cdf);
                                printf("selected_cdf %10.8f l31\n", selected_cdf);
                                printf("indProb %i out of %i l31\n", indProb, dimt);
                        }
                        __syncwarp(WMASK);
                #endif
                return 1;
        }
}

template<int BDIM_X,
         int BDIM_Y,
         ModelType MODEL_T,
         typename REAL_T,
         typename REAL3_T>
__device__ int tracker_d(curandStatePhilox4_32_10_t *st,
			 const REAL_T max_angle,
			 const REAL_T tc_threshold,
			 const REAL_T step_size,
			 const REAL_T relative_peak_thres,
			 const REAL_T min_separation_angle,
                         REAL3_T seed,
                         REAL3_T first_step,
                         REAL_T* ptt_frame,
                         REAL3_T voxel_size,
                         const int dimx,
                         const int dimy,
                         const int dimz,
                         const int dimt,
                         const REAL_T *__restrict__ dataf,
                         const REAL_T *__restrict__ metric_map,
		         const int samplm_nr,
                         const REAL3_T *__restrict__ sphere_vertices,
                         const int2 *__restrict__ sphere_edges,
                         const int num_edges,
                         int *__restrict__ nsteps,
                         REAL3_T *__restrict__ streamline) {

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        int tissue_class = TRACKPOINT;

        REAL3_T point = seed;
        REAL3_T direction = first_step;
        __shared__ REAL3_T __sh_new_dir[BDIM_Y];

        if (tidx == 0) {
                streamline[0] = point;
        }
        __syncwarp(WMASK);

        int step_frac;
        if (MODEL_T == PTT) {
                step_frac = STEP_FRAC; 
        } else {
                step_frac = 1; // STEP_FRAC could be useful in other models
        }

        int i;
        for(i = 1; i < MAX_SLINE_LEN*step_frac; i++) {
                int ndir;
                if constexpr (MODEL_T == PROB) {
                        ndir = get_direction_prob_d<BDIM_X,
                                                    BDIM_Y,
                                                    0>(
                                                        st,
                                                        dataf,
                                                        max_angle,
                                                        relative_peak_thres,
                                                        min_separation_angle,
                                                        direction,
                                                        dimx, dimy, dimz, dimt,
                                                        point,
                                                        sphere_vertices,
                                                        sphere_edges,
                                                        num_edges,
                                                        __sh_new_dir + tidy);
                } else if constexpr (MODEL_T == PTT) {
                        ndir = get_direction_ptt_d<BDIM_X,
                                                   BDIM_Y,
                                                   0>(
                                                        st,
                                                        dataf,
                                                        max_angle,
                                                        step_size,
                                                        direction,
                                                        ptt_frame,
                                                        dimx, dimy, dimz, dimt,
                                                        point,
                                                        sphere_vertices,
                                                        __sh_new_dir + tidy);
                }
                __syncwarp(WMASK);
                direction = __sh_new_dir[tidy];
                __syncwarp(WMASK);

                if (ndir == 0) {
                        break;
                }
#if 0
                if (threadIdx.y == 1 && threadIdx.x == 0) {
                        printf("tracker: i: %d, direction: (%f, %f, %f)\n", i, direction.x, direction.y, direction.z);
                }
                //return;
#endif

                point.x += (direction.x / voxel_size.x) * (step_size / step_frac);
                point.y += (direction.y / voxel_size.y) * (step_size / step_frac);
                point.z += (direction.z / voxel_size.z) * (step_size / step_frac);

                if ((tidx == 0) && ((i % step_frac) == 0)){
                        streamline[i/step_frac] = point;
#if 0
                        if (threadIdx.y == 1) {
                                printf("streamline[%d]: %f, %f, %f\n", i, point.x, point.y, point.z);
                        }
#endif
                }
                __syncwarp(WMASK);

                tissue_class = check_point_d<BDIM_X, BDIM_Y>(tc_threshold, point, dimx, dimy, dimz, metric_map);

#if 0
                __syncwarp(WMASK);
                if (tidx == 0) {
                        printf("step_size %f\n", step_size);
                        printf("direction %f, %f, %f\n", direction.x, direction.y, direction.z);
                        printf("direction addr read %p, slid %i\n", __shDir, blockIdx.x*blockDim.y+threadIdx.y);
                        printf("voxel_size %f, %f, %f\n", voxel_size.x, voxel_size.y, voxel_size.z);
                        printf("point %f, %f, %f\n", point.x, point.y, point.z);
                        printf("tc %i\n", tissue_class);
                }
                __syncwarp(WMASK);
                if (tidx == 15) {
                        printf("step_size %f l15\n", step_size);
                        printf("direction %f, %f, %f l15\n", direction.x, direction.y, direction.z);
                        printf("direction addr read %p, slid %i l15\n", __shDir, blockIdx.x*blockDim.y+threadIdx.y);
                        printf("voxel_size %f, %f, %f l15\n", voxel_size.x, voxel_size.y, voxel_size.z);
                        printf("point %f, %f, %f l15\n", point.x, point.y, point.z);
                        printf("tc %i l15\n", tissue_class);
                }
                __syncwarp(WMASK);
                if (tidx == 31) {
                        printf("step_size %f l31\n", step_size);
                        printf("direction %f, %f, %f l31\n", direction.x, direction.y, direction.z);
                        printf("direction addr read %p, slid %i l31\n", __shDir, blockIdx.x*blockDim.y+threadIdx.y);
                        printf("voxel_size %f, %f, %f l31\n", voxel_size.x, voxel_size.y, voxel_size.z);
                        printf("point %f, %f, %f l31\n", point.x, point.y, point.z);
                        printf("tc %i l31\n", tissue_class);
                }
                __syncwarp(WMASK);
#endif

                if (tissue_class == ENDPOINT ||
                    tissue_class == INVALIDPOINT ||
                    tissue_class == OUTSIDEIMAGE) {
                        break;
                }
        }
        nsteps[0] = i/step_frac;
        if (((i % step_frac) != 0) && i < step_frac*(MAX_SLINE_LEN - 1)){
                nsteps[0]++;
                if (tidx == 0) {
                        streamline[nsteps[0]] = point;
                }
        }

        return tissue_class;
}

template<int BDIM_X,
         int BDIM_Y,
         typename REAL_T,
         typename REAL3_T>
__global__ void getNumStreamlinesProb_k(const REAL_T max_angle,
				        const REAL_T relative_peak_thres,
				        const REAL_T min_separation_angle,
				        const long long rndSeed,
                                        const int nseed,
                                        const REAL3_T *__restrict__ seeds,
                                        const int dimx,
                                        const int dimy,
                                        const int dimz,
                                        const int dimt,
                                        const REAL_T *__restrict__ dataf,
                                        const REAL3_T *__restrict__ sphere_vertices,
                                        const int2 *__restrict__ sphere_edges,
                                        const int num_edges,
                                        REAL3_T *__restrict__ shDir0,
                                        int *slineOutOff) {

        const int tidx = threadIdx.x;
        const int slid = blockIdx.x*blockDim.y + threadIdx.y;
        const size_t gid = blockIdx.x * blockDim.y * blockDim.x + blockDim.x * threadIdx.y + threadIdx.x;

        if (slid >= nseed) {
                return;
        }

        REAL3_T *__restrict__ __shDir = shDir0+slid*dimt;
        curandStatePhilox4_32_10_t st;
        curand_init(rndSeed, gid, 0, &st);

        int ndir = get_direction_prob_d<BDIM_X,
                                        BDIM_Y,
                                        1>(
                                                &st,
                                                dataf,
                                                max_angle,
                                                relative_peak_thres,
                                                min_separation_angle,
                                                MAKE_REAL3(0,0,0),
                                                dimx, dimy, dimz, dimt,
                                                seeds[slid],
                                                sphere_vertices,
                                                sphere_edges,
                                                num_edges,
                                                __shDir);
        if (tidx == 0) {
                slineOutOff[slid] = ndir;
        }

        return;
}

template<int BDIM_X,
         int BDIM_Y,
         ModelType MODEL_T,
         typename REAL_T,
         typename REAL3_T>
__global__ void genStreamlinesMergeProb_k(
				      const REAL_T max_angle,
				      const REAL_T tc_threshold,
				      const REAL_T step_size,
				      const REAL_T relative_peak_thres,
				      const REAL_T min_separation_angle,
				      const long long rndSeed,
                                      const int rndOffset,
                                      const int nseed,
                                      const REAL3_T *__restrict__ seeds,
                                      const int dimx,
                                      const int dimy,
                                      const int dimz,
                                      const int dimt,
                                      const REAL_T *__restrict__ dataf,
                                      const REAL_T *__restrict__ metric_map,
				      const int samplm_nr,
                                      const REAL3_T *__restrict__ sphere_vertices,
                                      const int2 *__restrict__ sphere_edges,
                                      const int num_edges,
                                      const int    *__restrict__ slineOutOff,
                                            REAL3_T *__restrict__ shDir0,
                                            int     *__restrict__ slineSeed,
                                            int     *__restrict__ slineLen,
                                            REAL3_T *__restrict__ sline) {

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int slid = blockIdx.x*blockDim.y + threadIdx.y;

        const int lid = (tidy*BDIM_X + tidx) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        __shared__ REAL_T frame_sh[((MODEL_T == PTT) ? BDIM_Y*18 : 1)]; // Only used by PTT, TODO: way to remove this in other cases
        REAL_T* __ptt_frame = frame_sh + tidy*18;
	// const int hr_side = dimt-1;

        curandStatePhilox4_32_10_t st;
        // const int gbid = blockIdx.y*gridDim.x + blockIdx.x;
        const size_t gid = blockIdx.x * blockDim.y * blockDim.x + blockDim.x * threadIdx.y + threadIdx.x;
        //curand_init(rndSeed, slid+rndOffset, DIV_UP(hr_side, BDIM_X)*tidx, &st); // each thread uses DIV_UP(HR_SIDE/BDIM_X)
        curand_init(rndSeed, gid+1, 0, &st); // each thread uses DIV_UP(hr_side/BDIM_X)
                                                                                 // elements of the same sequence
        if (slid >= nseed) {
                return;
        }

        REAL3_T seed = seeds[slid]; 

        int ndir = slineOutOff[slid+1]-slineOutOff[slid];
#if 0
        if (threadIdx.y == 0 && threadIdx.x == 0) {
                printf("%s: ndir: %d\n", __func__, ndir);
                for(int i = 0; i < ndir; i++) {
                        printf("__shDir[%d][%d]: (%f, %f, %f)\n",
                                tidy, i, __shDir[tidy][i].x, __shDir[tidy][i].y, __shDir[tidy][i].z);
                }
        }
#endif
        __syncwarp(WMASK);

        int slineOff = slineOutOff[slid];

        for(int i = 0; i < ndir; i++) {
                REAL3_T first_step = shDir0[slid*samplm_nr + i];

		REAL3_T *__restrict__ currSline = sline + slineOff*MAX_SLINE_LEN*2;

                if (tidx == 0) {
                        slineSeed[slineOff] = slid;
                }
#if 0
                if (threadIdx.y == 0 && threadIdx.x == 0) {
                        printf("calling trackerF from: (%f, %f, %f)\n", first_step.x, first_step.y, first_step.z);
                }
#endif

                if (MODEL_T == PTT) {
                        if (!init_frame_ptt_d<BDIM_X, BDIM_Y>(
                                &st,
                                dataf,
                                max_angle,
                                step_size,
                                first_step,
                                dimx, dimy, dimz, dimt,
                                seed,
                                sphere_vertices,
                                __ptt_frame
                        )) { // this fails rarely
                                if (tidx == 0) {
                                        slineLen[slineOff] = 1;
                                        currSline[0] = seed;
                                }
                                __syncwarp(WMASK);
                                slineOff += 1;
                                continue;
                        }
                }


                int stepsB;
                const int tissue_classB = tracker_d<BDIM_X,
                                                    BDIM_Y,
                                                    MODEL_T>(&st,
		                		             max_angle,
			        			     tc_threshold,
	                        			     step_size,
	                        			     relative_peak_thres,
	                        			     min_separation_angle,
                                                             seed,
                                                             MAKE_REAL3(-first_step.x, -first_step.y, -first_step.z),
                                                             __ptt_frame,
                                                             MAKE_REAL3(1, 1, 1),
                                                             dimx, dimy, dimz, dimt, dataf,
                                                             metric_map,
		                			     samplm_nr,
                                                             sphere_vertices,
                                                             sphere_edges,
                                                             num_edges,
                                                             &stepsB,
                                                             currSline);
                //if (tidx == 0) {
                //        slineLenB[slineOff] = stepsB;
                //}

                // reverse backward sline
                for(int j = 0; j < stepsB/2; j += BDIM_X) {
                        if (j+tidx < stepsB/2) {
                                const REAL3_T __p = currSline[j+tidx];
                                currSline[j+tidx] = currSline[stepsB-1 - (j+tidx)];
                                currSline[stepsB-1 - (j+tidx)] = __p;
                        }
                }

                int stepsF;
                const int tissue_classF = tracker_d<BDIM_X,
                                                    BDIM_Y,
                                                    MODEL_T>(&st,
     	                    			             max_angle,
		        				     tc_threshold,
	                				     step_size,
			    				     relative_peak_thres,
			            			     min_separation_angle,
                                                             seed,
                                                             first_step,
                                                             __ptt_frame + 9,
                                                             MAKE_REAL3(1, 1, 1),
                                                             dimx, dimy, dimz, dimt, dataf,
                                                             metric_map,
			        			     samplm_nr,
                                                             sphere_vertices,
                                                             sphere_edges,
                                                             num_edges,
                                                             &stepsF,
                                                             currSline + stepsB-1);
                if (tidx == 0) {
                        slineLen[slineOff] = stepsB-1+stepsF;
                }
                
                slineOff += 1;
#if 0
                if (threadIdx.y == 0 && threadIdx.x == 0) {
                        printf("%s: stepsF: %d, tissue_classF: %d\n", __func__, stepsF, tissue_classF);
                }
                __syncwarp(WMASK);
#endif
                //if (/* !return_all || */0 &&
                //    tissue_classF != ENDPOINT &&
                //    tissue_classF != OUTSIDEIMAGE) {
                //        continue;
                //}
                //if (/* !return_all || */ 0 &&
                //    tissue_classB != ENDPOINT &&
                //    tissue_classB != OUTSIDEIMAGE) {
                //        continue;
                //}
        }
        return;
}
