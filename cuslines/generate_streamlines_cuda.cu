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

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <getopt.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cfloat>
#include <omp.h>
#include <vector>
#include "cudamacro.h" /* for time() */
#include "globals.h"
//#include "utils.h"

#include <iostream>

#include "cuwsort.cuh"

#define MAX_NUM_DIR (128)

#define NTHR_GEN (128)

#define THR_X_BL (64)
#define THR_X_SL (32)

#define MAX_DIMS        (8)
#define MAX_STR_LEN     (256)

using namespace cuwsort;

//#define USE_FIXED_PERMUTATION
#ifdef USE_FIXED_PERMUTATION
//__device__ const int fixedPerm[] = {44, 47, 53,  0,  3,  3, 39,  9, 19, 21, 50, 36, 23,
//                                     6, 24, 24, 12,  1, 38, 39, 23, 46, 24, 17, 37, 25, 
//                                    13,  8,  9, 20, 51, 16, 51,  5, 15, 47,  0, 18, 35, 
//                                    24, 49, 51, 29, 19, 19, 14, 39, 32,  1,  9, 32, 31,
//                                    10, 52, 23};
__device__ const int fixedPerm[] = {
  47, 117,  67, 103,   9,  21,  36,  87,  70,  88, 140,  58,  39,  87,  88,  81,  25,  77,
  72,   9, 148, 115,  79,  82,  99,  29, 147, 147, 142,  32,   9, 127,  32,  31, 114,  28,
  34, 128, 128,  53, 133,  38,  17,  79, 132, 105,  42,  31, 120,   1,  65,  57,  35, 102,
 119,  11,  82,  91, 128, 142,  99,  53, 140, 121,  84,  68,   6,  47, 127, 131, 100,  78,
 143, 148,  23, 141, 117,  85,  48,  49,  69,  95,  94,   0, 113,  36,  48,  93, 131,  98,
  42, 112, 149, 127,   0, 138, 114,  43, 127,  23, 130, 121,  98,  62, 123,  82, 148,  50,
  14,  41,  58,  36,  10,  86,  43, 104,  11,   2,  51,  80,  32, 128,  38,  19,  42, 115,
  77,  30,  24, 125,   2,   3,  94, 107,  13, 112,  40,  72,  19,  95,  72,  67,  61,  14,
  96,   4, 139,  86, 121, 109};
#endif


template<int BDIM_X,
         typename REAL_T,
         typename REAL3_T>
__device__ int trilinear_interp_d(const int dimx,
                                  const int dimy,
                                  const int dimz,
                                  const int dimt,
                                  const REAL_T *__restrict__ dataf,
                                  const REAL3_T point,
                                        REAL_T *__restrict__ __vox_data) {

        const int tidx = threadIdx.x;
        //const int tidy = threadIdx.y;
#if 0        
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));
#endif
        const REAL_T HALF = static_cast<REAL_T>(0.5);

        // all thr compute the same here
        if (point.x < -HALF || point.x+HALF >= dimx ||
            point.y < -HALF || point.y+HALF >= dimy ||
               point.z < -HALF || point.z+HALF >= dimz) {
                return -1;
        }

        long long  coo[3][2];
        REAL wgh[3][2]; // could use just one...

        const REAL_T ONE  = static_cast<REAL_T>(1.0);

        const REAL3_T fl = MAKE_REAL3(FLOOR(point.x),
                                      FLOOR(point.y),
                                      FLOOR(point.z));

        wgh[0][1] = point.x - fl.x; 
        wgh[0][0] = ONE-wgh[0][1]; 
        coo[0][0] = MAX(0, fl.x);
        coo[0][1] = MIN(dimx-1, coo[0][0]+1);

        wgh[1][1] = point.y - fl.y; 
        wgh[1][0] = ONE-wgh[1][1]; 
        coo[1][0] = MAX(0, fl.y);
        coo[1][1] = MIN(dimy-1, coo[1][0]+1);

        wgh[2][1] = point.z - fl.z; 
        wgh[2][0] = ONE-wgh[2][1]; 
        coo[2][0] = MAX(0, fl.z);
        coo[2][1] = MIN(dimz-1, coo[2][0]+1);

        //#pragma unroll
        for(int t = tidx; t < dimt; t += BDIM_X) {

                REAL_T __tmp = 0;

                #pragma unroll
                for(int i = 0; i < 2; i++) {
                        #pragma unroll
                        for(int j = 0; j < 2; j++) {
                                #pragma unroll
                                for(int k = 0; k < 2; k++) {
                                        __tmp += wgh[0][i]*wgh[1][j]*wgh[2][k]*
                                                 dataf[coo[0][i]*dimy*dimz*dimt +
                                                       coo[1][j]*dimz*dimt +
                                                       coo[2][k]*dimt +
                                                       t];
                                        /*
                                        if (tidx == 0 && threadIdx.y == 0 && t==0) {
                                                printf("wgh[0][%d]: %f, wgh[1][%d]: %f, wgh[2][%d]: %f\n",
                                                        i, wgh[0][i], j, wgh[1][j], k, wgh[2][k]);
                                                printf("dataf[%d][%d][%d][%d]: %f\n", coo[0][i], coo[1][j], coo[2][k], t+tidx,
                                                                dataf[coo[0][i]*dimy*dimz*dimt +
                                                                coo[1][j]*dimz*dimt +
                                                                coo[2][k]*dimt +
                                                                t+tidx]);
                                        }
                                        */
                                }
                        }
                }
                __vox_data[t] = __tmp;
        }
#if 0
        __syncwarp(WMASK);
        if (tidx == 0 && threadIdx.y == 0) {
                printf("point: %f, %f, %f\n", point.x, point.y, point.z);
                for(int i = 0; i < dimt; i++) {
                        printf("__vox_data[%d]: %f\n", i, __vox_data[i]);
                }
        }
#endif
        return 0;
}

template<int BDIM_X,
         typename VAL_T>
__device__ void ndotp_d(const int N,
			const int M,
			const VAL_T *__restrict__ srcV,
                        const VAL_T *__restrict__ srcM,
                              VAL_T *__restrict__ dstV) {

        const int tidx = threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        //#pragma unroll
        for(int i = 0; i < N; i++) {

                VAL_T __tmp = 0;

                //#pragma unroll
                for(int j = 0; j < M; j += BDIM_X) {
                        if (j+tidx < M) {
                                __tmp += srcV[j+tidx]*srcM[i*M + j+tidx];
                        }
                }
                #pragma unroll
                for(int j = BDIM_X/2; j; j /= 2) {
#if 0
                        __tmp += __shfl_xor_sync(WMASK, __tmp, j, BDIM_X);
#else
                        __tmp += __shfl_down_sync(WMASK, __tmp, j, BDIM_X);
#endif
                }
                // values could be held by BDIM_X threads and written
                // together every BDIM_X iterations...

                if (tidx == 0) {
                        dstV[i] = __tmp;
                }
        }
        return;
}


template<int BDIM_X,
         typename VAL_T>
__device__ void ndotp_log_opdt_d(const int N,
			    const int M,
			    const VAL_T *__restrict__ srcV,
                            const VAL_T *__restrict__ srcM,
                                  VAL_T *__restrict__ dstV) {

        const int tidx = threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
         const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        const VAL_T ONEP5 = static_cast<VAL_T>(1.5);

        //#pragma unroll
        for(int i = 0; i < N; i++) {

                VAL_T __tmp = 0;

                //#pragma unroll
                for(int j = 0; j < M; j += BDIM_X) {
                        if (j+tidx < M) {
                                const VAL_T v = srcV[j+tidx];
                                __tmp += -LOG(v)*(ONEP5+LOG(v))*v * srcM[i*M + j+tidx];
                        }
                }
                #pragma unroll
                for(int j = BDIM_X/2; j; j /= 2) {
#if 0
                        __tmp += __shfl_xor_sync(WMASK, __tmp, j, BDIM_X);
#else
                        __tmp += __shfl_down_sync(WMASK, __tmp, j, BDIM_X);
#endif
                }
                // values could be held by BDIM_X threads and written
                // together every BDIM_X iterations...

                if (tidx == 0) {
                        dstV[i] = __tmp;
                }
        }
        return;
}

template<int BDIM_X,
	 typename VAL_T>
__device__ void ndotp_log_csa_d(const int N,
				const int M,
				const VAL_T *__restrict__ srcV,
				const VAL_T *__restrict__ srcM,
				VAL_T *__restrict__ dstV) {

	const int tidx = threadIdx.x;

	const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
	const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));
	// Clamp values
	constexpr VAL_T min = .001;
	constexpr VAL_T max = .999;

	//#pragma unroll
	for(int i = 0; i < N; i++) {

		VAL_T __tmp = 0;

		//#pragma unroll
		for(int j = 0; j < M; j += BDIM_X) {
			if (j+tidx < M) {
				const VAL_T v = MIN(MAX(srcV[j+tidx], min), max);
				__tmp += LOG(-LOG(v)) * srcM[i*M + j+tidx];
			}
		}
		#pragma unroll
		for(int j = BDIM_X/2; j; j /= 2) {
#if 0
			__tmp += __shfl_xor_sync(WMASK, __tmp, j, BDIM_X);
#else
			__tmp += __shfl_down_sync(WMASK, __tmp, j, BDIM_X);
#endif
		}
		// values could be held by BDIM_X threads and written
		// together every BDIM_X iterations...

		if (tidx == 0) {
			dstV[i] = __tmp;
		}
	}
	return;
}


template<int BDIM_X,
         typename REAL_T>
__device__ void fit_opdt(const int delta_nr,
                         const int hr_side,
                         const REAL_T *__restrict__ delta_q,
                         const REAL_T *__restrict__ delta_b,
                         const REAL_T *__restrict__ __msk_data_sh,
                         REAL_T *__restrict__ __h_sh,
                         REAL_T *__restrict__ __r_sh) {
        const int tidx = threadIdx.x;
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        ndotp_log_opdt_d<BDIM_X>(delta_nr, hr_side, __msk_data_sh, delta_q, __r_sh);
        ndotp_d         <BDIM_X>(delta_nr, hr_side, __msk_data_sh, delta_b, __h_sh);
        __syncwarp(WMASK);
        #pragma unroll
        for(int j = tidx; j < delta_nr; j += BDIM_X) {
                __r_sh[j] -= __h_sh[j];
        }
        __syncwarp(WMASK);
}

template<int BDIM_X, typename REAL_T>
__device__ void fit_csa(const int delta_nr,
                        const int hr_side,
                        const REAL_T *__restrict__ fit_matrix,
                        const REAL_T *__restrict__ __msk_data_sh,
                        REAL_T *__restrict__ __r_sh) {
        const int tidx = threadIdx.x;
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        constexpr REAL _n0_const = 0.28209479177387814; // .5 / sqrt(pi)
        ndotp_log_csa_d<BDIM_X>(delta_nr, hr_side, __msk_data_sh, fit_matrix, __r_sh);
        __syncwarp(WMASK);
        if (tidx == 0) {
                __r_sh[0] = _n0_const;
        }
        __syncwarp(WMASK);
}

// Based on dipy implementation:
// https://github.com/dipy/dipy/blob/104345de6dcbfd6627552b00d180e53db6c17381/dipy/reconst/csdeconv.py#L573C5-L573C13
// template<int BDIM_X, typename REAL_T>
// __device__ void fit_csd(const int delta_nr,
//                         const int hr_side,
//                         const REAL_T *__restrict__ __X,
//                         const REAL_T *__restrict__ B_reg,
//                         const REAL_T *__restrict__ __msk_data_sh,
//                         REAL_T *__restrict__ __r_sh) {
//         constexpr REAL_T mu = 1e-5;
//         constexpr REAL_T tau = 0.1;
//         REAL* __P, __z; // TODO: find somewhere to alloc these, or using existing space
//         nTmuln_d<BDIM_X>(delta_nr, hr_side, __X, __P); // TODO: P can be calculated beforehand
//         ndotp_d <BDIM_X>(delta_nr, hr_side, __X, __msk_data_sh, __z);

//         // TODO: finish this

// }

template<int BDIM_X, typename REAL_T>
__device__ void fit_model_coef(const ModelType model_type,
                               const int delta_nr, // delta_nr is number of ODF directions
                               const int hr_side, // hr_side is number of data directions
                               const REAL_T *__restrict__ delta_q,
                               const REAL_T *__restrict__ delta_b, // these are fit matrices the model can use, different for each model
                               const REAL_T *__restrict__ __msk_data_sh, // __msk_data_sh is the part of the data currently being operated on by this block
                               REAL_T *__restrict__ __h_sh, // these last two are modifications to the coefficients that will be returned
                               REAL_T *__restrict__ __r_sh) {
        switch(model_type) {
                case OPDT:
                        fit_opdt<BDIM_X>(delta_nr, hr_side, delta_q, delta_b, __msk_data_sh, __h_sh, __r_sh);
                        break;
                case CSA:
                        fit_csa<BDIM_X>(delta_nr, hr_side, delta_q, __msk_data_sh, __r_sh);
                        break;
                default:
                        printf("FATAL: Invalid Model Type.\n");
                        break;
        }
}

template<int BDIM_X,
         typename LEN_T,
         typename VAL_T>
__device__ VAL_T max_mask_transl_d(const int n,
				   const LEN_T *__restrict__ srcMsk,
                                   const VAL_T *__restrict__ srcVal,
                                   const VAL_T offset,
                                   const VAL_T minVal) {

        const int tidx = threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        VAL_T __m = minVal;

        for(int i = tidx; i < n; i += BDIM_X) {
		const LEN_T sel = srcMsk[i];
		if (sel > 0) {
			__m = MAX(__m, srcVal[i]+offset);
		}
        }

        #pragma unroll
        for(int i = BDIM_X/2; i; i /= 2) {
                const VAL_T __tmp = __shfl_xor_sync(WMASK, __m, i, BDIM_X);
                __m = MAX(__m, __tmp);
        }

        return __m;
}

template<int BDIM_X,
         typename VAL_T>
__device__ VAL_T max_d(const int n, const VAL_T *__restrict__ src, const VAL_T minVal) {

        const int tidx = threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        VAL_T __m = minVal;

        for(int i = tidx; i < n; i += BDIM_X) {
		__m = MAX(__m, src[i]);
        }

        #pragma unroll
        for(int i = BDIM_X/2; i; i /= 2) {
                const VAL_T __tmp = __shfl_xor_sync(WMASK, __m, i, BDIM_X);
                __m = MAX(__m, __tmp);
        }

        return __m;
}

template<int BDIM_X,
         typename VAL_T>
__device__ VAL_T min_d(const int n, const VAL_T *__restrict__ src, const VAL_T maxVal) {

        const int tidx = threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        VAL_T __m = maxVal;

        for(int i = tidx; i < n; i += BDIM_X) {
		__m = MIN(__m, src[i]);
        }

        #pragma unroll
        for(int i = BDIM_X/2; i; i /= 2) {
                const VAL_T __tmp = __shfl_xor_sync(WMASK, __m, i, BDIM_X);
                __m = MIN(__m, __tmp);
        }

        return __m;
}
			
template<int BDIM_X,
         typename VAL_T>
__device__ VAL_T avgMask(const int mskLen,
			 const int *__restrict__ mask,
			 const VAL_T *__restrict__ data) {
        
	const int tidx = threadIdx.x;
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;

        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        int   __myCnt = 0;
        VAL_T __mySum = 0;

        for(int i = tidx; i < mskLen; i += BDIM_X) {
		if(mask[i]) {
			__myCnt++;
			__mySum += data[i];
		}
        }

        #pragma unroll
        for(int i = BDIM_X/2; i; i /= 2) {
                __mySum += __shfl_xor_sync(WMASK, __mySum, i, BDIM_X);
                __myCnt += __shfl_xor_sync(WMASK, __myCnt, i, BDIM_X);
        }

        return __mySum/__myCnt;

}

template<int BDIM_X,
         int BDIM_Y,
         typename REAL_T,
         typename REAL3_T>
__device__ int peak_directions_d(const REAL_T  *__restrict__ odf,
                                       REAL3_T *__restrict__ dirs,
                                 const REAL3_T *__restrict__ sphere_vertices,
                                 const int2 *__restrict__ sphere_edges,
                                 const int num_edges,
				 int samplm_nr,
				 int *__restrict__ __shInd,
				 const REAL_T relative_peak_thres,
				 const REAL_T min_separation_angle) {

        const int tidx = threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        const unsigned int lmask = (1 << lid)-1;

//        __shared__ int __shInd[BDIM_Y][SAMPLM_NR];

        #pragma unroll
        for(int j = tidx; j < samplm_nr; j += BDIM_X) {
		__shInd[j] = 0;
        }

        REAL_T odf_min = min_d<BDIM_X>(samplm_nr, odf, REAL_MAX);
        odf_min = MAX(0, odf_min);

        __syncwarp(WMASK);

        // local_maxima() + _compare_neighbors()
        // selecting only the indices corrisponding to maxima Ms
        // such that M-odf_min >= relative_peak_thres
        //#pragma unroll
        for(int j = 0; j < num_edges; j += BDIM_X) {
                if (j+tidx < num_edges) {
                        const int u_ind = sphere_edges[j+tidx].x;
                        const int v_ind = sphere_edges[j+tidx].y;

                        //if (u_ind >= NUM_EDGES || v_ind >= NUM_EDGES) { ERROR; }

                        const REAL_T u_val = odf[u_ind];
                        const REAL_T v_val = odf[v_ind];

                        //if (u_val != u_val || v_val != v_val) { ERROR_NANs; }

                        // only check that they are not equal
                        //if (u_val != v_val) {
                        //        __shInd[tidy][u_val < v_val ? u_ind : v_ind] = -1; // benign race conditions...
                        //}
                        if (u_val < v_val) {
                                atomicExch(__shInd+u_ind, -1);
                                atomicOr(  __shInd+v_ind,  1);
                        } else if (v_val < u_val) {
                                atomicExch(__shInd+v_ind, -1);
                                atomicOr(  __shInd+u_ind,  1);
                        }
                }
        }
        __syncwarp(WMASK);

        const REAL_T compThres = relative_peak_thres*max_mask_transl_d<BDIM_X>(samplm_nr, __shInd, odf, -odf_min, REAL_MIN);
#if 1
/*
        if (!tidy && !tidx) {
                for(int j = 0; j < SAMPLM_NR; j++) {
                        printf("local_max[%d]: %d (%f)\n", j, __shInd[tidy][j], odf[j]);
                }
                printf("maxMax with offset %f: %f\n", -odf_min, compThres);
        }
        __syncwarp(WMASK);
*/
        // compact indices of positive values to the right
        int n = 0;

        for(int j = 0; j < samplm_nr; j += BDIM_X) {

                const int __v = (j+tidx < samplm_nr) ? __shInd[j+tidx] : -1;
                const int __keep = (__v > 0) && ((odf[j+tidx]-odf_min) >= compThres);
                const int __msk = __ballot_sync(WMASK, __keep);

//__syncwarp(WMASK); // unnecessary
                if (__keep) {
                        const int myoff = __popc(__msk & lmask);
                        __shInd[n + myoff] = j+tidx;
                }
                n += __popc(__msk);
//__syncwarp(WMASK); // should be unnecessary
        }
        __syncwarp(WMASK);
/*
        if (!tidy && !tidx) {
                for(int j = 0; j < n; j++) {
                        printf("local_max_compact[%d]: %d\n", j, __shInd[tidy][j]);
                }
        }
        __syncwarp(WMASK);
*/

        // sort local maxima indices
        if (n < BDIM_X) {
                REAL_T k = REAL_MIN;
                int    v = 0;
                if (tidx < n) {
                        v = __shInd[tidx];
                        k = odf[v];
                }
                warp_sort<32, BDIM_X, WSORT_DIR_DEC>(&k, &v);
                __syncwarp(WMASK);

                if (tidx < n) {
                        __shInd[tidx] = v;
                }
        } else {
                // ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        }
        __syncwarp(WMASK);

        // __shInd[tidy][] contains the indices in odf correspoding to
        // normalized maxima NOT sorted!
        if (n != 0) {
                // remove_similar_vertices()
                // PRELIMINARY INEFFICIENT, SINGLE TH, IMPLEMENTATION
                if (tidx == 0) {
                        const REAL_T cos_similarity = COS(min_separation_angle);

                        dirs[0] = sphere_vertices[__shInd[0]];

                        int k = 1;
                        for(int i = 1; i < n; i++) {

                                const REAL3_T abc = sphere_vertices[__shInd[i]];

                                int j = 0;
                                for(; j < k; j++) {
                                        const REAL_T cos = FABS(abc.x*dirs[j].x+
                                                                abc.y*dirs[j].y+
                                                                abc.z*dirs[j].z);
                                        if (cos > cos_similarity) {
                                                break;
                                        }
                                }
                                if (j == k) {
                                        dirs[k++] = abc;
                                }
                        }
                        n = k;
                }
                n = __shfl_sync(WMASK, n, 0, BDIM_X);
                __syncwarp(WMASK);

        }
/*
        if (!tidy && !tidx) {
                for(int j = 0; j < n; j++) {
                        printf("local_max_compact_uniq[%d]: %d\n", j, __shInd[tidy][j]);
                }
        }
        __syncwarp(WMASK);
*/
#else
        const int indMax = max_d<BDIM_X, SAMPLM_NR>(__shInd[tidy], -1);
        if (indMax != -1) {
                __ret = MAKE_REAL3(sphere_vertices[indMax][0],
                                   sphere_vertices[indMax][1],
                                   sphere_vertices[indMax][2]);
        }
#endif
        return n;
}

template<int BDIM_X,
         int BDIM_Y,
         typename REAL_T,
         typename REAL3_T>
__device__ int closest_peak_d(const REAL_T max_angle,
			      const REAL3_T  direction, //dir
                              const int npeaks,
                              const REAL3_T *__restrict__ peaks,
                                    REAL3_T *__restrict__ peak) {// dirs,

        const int tidx = threadIdx.x;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        //const REAL_T cos_similarity = COS(MAX_ANGLE_P);
        const REAL_T cos_similarity = COS(max_angle);
#if 0
        if (!threadIdx.y && !tidx) {
                printf("direction: (%f, %f, %f)\n",
                        direction.x, direction.y, direction.z);
        }
        __syncwarp(WMASK);
#endif
        REAL_T cpeak_dot = 0;
        int    cpeak_idx = -1;
        for(int j = 0; j < npeaks; j += BDIM_X) {
                if (j+tidx < npeaks) {
#if 0
                        if (!threadIdx.y && !tidx) {
                                printf("j+tidx: %d, peaks[j+tidx]: (%f, %f, %f)\n",
                                        j+tidx, peaks[j+tidx].x, peaks[j+tidx].y, peaks[j+tidx].z);
                        }
#endif
                        const REAL_T dot = direction.x*peaks[j+tidx].x+
                                           direction.y*peaks[j+tidx].y+
                                           direction.z*peaks[j+tidx].z;

                        if (FABS(dot) > FABS(cpeak_dot)) {
                                cpeak_dot = dot;
                                cpeak_idx = j+tidx;
                        }
                }
        }
#if 0
        if (!threadIdx.y && !tidx) {
                printf("cpeak_idx: %d, cpeak_dot: %f\n", cpeak_idx, cpeak_dot);
        }
        __syncwarp(WMASK);
#endif

        #pragma unroll
        for(int j = BDIM_X/2; j; j /= 2) {

                const REAL_T dot = __shfl_xor_sync(WMASK, cpeak_dot, j, BDIM_X);
                const int    idx = __shfl_xor_sync(WMASK, cpeak_idx, j, BDIM_X);
                if (FABS(dot) > FABS(cpeak_dot)) {
                        cpeak_dot = dot;
                        cpeak_idx = idx;
                }
        }
#if 0
        if (!threadIdx.y && !tidx) {
                printf("cpeak_idx: %d, cpeak_dot: %f, cos_similarity: %f\n", cpeak_idx, cpeak_dot, cos_similarity);
        }
        __syncwarp(WMASK);
#endif
        if (cpeak_idx >= 0) {
                if (cpeak_dot >= cos_similarity) {
                        peak[0] = peaks[cpeak_idx];
                        return 1;
                }
                if (cpeak_dot <= -cos_similarity) {
                        peak[0] = MAKE_REAL3(-peaks[cpeak_idx].x,
                                             -peaks[cpeak_idx].y,
                                             -peaks[cpeak_idx].z);
                        return 1;
                }
        }
        return 0;
}

template<int BDIM_X,
	 typename LEN_T,
	 typename MSK_T,
	 typename VAL_T>
__device__ LEN_T maskGet(const LEN_T n, 
			 const MSK_T *__restrict__ mask,
			 const VAL_T *__restrict__ plain,
			       VAL_T *__restrict__ masked) {

	const int tidx = threadIdx.x;
	
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

	const int __laneMask = (1 << tidx)-1;

	int woff = 0;
	for(int j = 0; j < n; j += BDIM_X) {

		const int __act = (j+tidx < n) ? !mask[j+tidx] : 0;
		const int __msk = __ballot_sync(WMASK, __act);

		const int toff = __popc(__msk & __laneMask);
		if (__act) {
			masked[woff+toff] = plain[j+tidx];
		}
		woff += __popc(__msk);
	}
	return woff;
}

template<int BDIM_X,
	 typename LEN_T,
	 typename MSK_T,
	 typename VAL_T>
__device__ void maskPut(const LEN_T n, 
			const MSK_T *__restrict__ mask,
			const VAL_T *__restrict__ masked,
			      VAL_T *__restrict__ plain) {

	const int tidx = threadIdx.x;
	
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

	const int __laneMask = (1 << tidx)-1;

	int woff = 0;
	for(int j = 0; j < n; j += BDIM_X) {

		const int __act = (j+tidx < n) ? !mask[j+tidx] : 0;
		const int __msk = __ballot_sync(WMASK, __act);

		const int toff = __popc(__msk & __laneMask);
		if (__act) {
			plain[j+tidx] = masked[woff+toff];
		}
		woff += __popc(__msk);
	}
	return;
}

template<typename REAL_T>
__device__ void printArray(const char *name, int ncol, int n, REAL_T *arr) {
	if (!threadIdx.x && !threadIdx.y && !blockIdx.x) {
		printf("%s:\n", name);

		for(int j = 0; j < n; j++) {
			if ((j%ncol)==0) printf("\n");
			printf("%10.8f ", arr[j]);
		}
		printf("\n");
	}
}

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
                                    const int samplm_nr,
                                    const REAL3_T *__restrict__ sphere_vertices,
                                    const int2 *__restrict__ sphere_edges,
                                    const int num_edges,
                                    REAL3_T *__restrict__ dirs) {
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
	
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

	const int n32dimt = ((dimt+31)/32)*32;

	extern REAL_T __shared__ __sh[];

	REAL_T *__pmf_data_sh = reinterpret_cast<REAL_T *>(__sh);
	int *__shInd = reinterpret_cast<int *>(__pmf_data_sh) + BDIM_Y*n32dimt; // only used if IS_START is 1

	__pmf_data_sh += tidy*n32dimt;
	__shInd += tidy*n32dimt;

        // pmf = self.pmf_gen.get_pmf_c(&point[0], pmf)
        const int rv = trilinear_interp_d<BDIM_X>(dimx, dimy, dimz, dimt, pmf, point, __pmf_data_sh);
        if (rv != 0) {
                return 0;
        }

        // for i in range(_len):
        //     if pmf[i] > max_pmf:
        //         max_pmf = pmf[i]
        // absolute_pmf_threshold = pmf_threshold * max_pmf
        const REAL_T absolpmf_thresh = PMF_THRESHOLD_P * max_d<BDIM_X>(samplm_nr, __pmf_data_sh, REAL_MIN);
        __syncwarp(WMASK);

        // for i in range(_len):
        //     if pmf[i] < absolute_pmf_threshold:
        //         pmf[i] = 0.0
        for(int i = tidx; i < samplm_nr; i += BDIM_X) {
                if (__pmf_data_sh[i] < absolpmf_thresh) {
                        __pmf_data_sh[i] = 0.0;
                }
        }
        __syncwarp(WMASK);

        if (IS_START) {
                return peak_directions_d<BDIM_X,
                                         BDIM_Y>(__pmf_data_sh,
                                                 dirs,
                                                 sphere_vertices,
                                                 sphere_edges,
                                                 num_edges,
                                                 samplm_nr,
                                                 __shInd,
                                                 relative_peak_thres,
                                                 min_separation_angle);
        } else {
                REAL_T __tmp;
                #ifdef DEBUG
                        __syncwarp();
                        if (tidx == 0) {
                                printArray("__pmf_data_sh initial", 8, samplm_nr, __pmf_data_sh);
                                printf("absolpmf_thresh %10.8f\n", absolpmf_thresh);
                                printf("--->            dir %10.8f, %10.8f, %10.8f\n", dir.x, dir.y, dir.z);
                                printf("--->            point %10.8f, %10.8f, %10.8f\n", point.x, point.y, point.z);
                                if (sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z) >= 1.05){
                                        printf("ERROR dir %10.8f, %10.8f, %10.8f\n", dir.x, dir.y, dir.z);
                                }
                        }
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

                for(int i = tidx; i < samplm_nr; i += BDIM_X) {
                        const REAL_T dot = dir.x*sphere_vertices[i].x+
                                           dir.y*sphere_vertices[i].y+
                                           dir.z*sphere_vertices[i].z;

                        if (FABS(dot) < cos_similarity) {
                                __pmf_data_sh[i] = 0.0;
                        }
                }
                __syncwarp(WMASK);

                #ifdef DEBUG
                        __syncwarp();
                        if (tidx == 0) {
                                printArray("__pmf_data_sh after filtering", 8, samplm_nr, __pmf_data_sh);
                        }
                #endif

                // cumsum(pmf, pmf, _len)
                for (int j = 0; j < samplm_nr; j += BDIM_X) {
                        if ((tidx == 0) && (j != 0)) {
                                __pmf_data_sh[j] += __pmf_data_sh[j-1];
                        }
                        __syncwarp(WMASK);

                        #pragma unroll
                        for(int i = 1; i < BDIM_X; i *= 2) {
                                if ((tidx >= i) && (j+tidx < samplm_nr)) {
                                        __tmp = __pmf_data_sh[j+tidx-i];
                                }
                                __syncwarp(WMASK);
                                if ((tidx >= i) && (j+tidx < samplm_nr)) {
                                        __pmf_data_sh[j+tidx] += __tmp;
                                }
                                __syncwarp(WMASK);
                        }
                }

                #ifdef DEBUG
                        __syncwarp();
                        if (tidx == 0) {
                                printArray("__pmf_data_sh after cumsum", 8, samplm_nr, __pmf_data_sh);
                        }
                #endif

                // last_cdf = pmf[_len - 1]
                // if last_cdf == 0:
                //         return 1
                if (tidx == 0) {
                        __tmp = __pmf_data_sh[samplm_nr - 1];
                }
                REAL_T last_cdf = __shfl_sync(WMASK, __tmp, 0, BDIM_X);
                if (last_cdf == 0) {
                        return 0;
                }

                // idx = where_to_insert(pmf, random() * last_cdf, _len)
                if (tidx == 0) {
                        __tmp = curand_uniform(st) * last_cdf;
                }
                REAL_T selected_cdf = __shfl_sync(WMASK, __tmp, 0, BDIM_X);
                int indProb = samplm_nr;
                for(int i = samplm_nr - 1 - tidx; i >= 0; i-= BDIM_X) {
                        if (selected_cdf >= __pmf_data_sh[i]) {
                                if ((i < samplm_nr) && (selected_cdf < __pmf_data_sh[i + 1])) {
                                        indProb = i + 1;
                                } else {
                                        indProb = MIN(samplm_nr - 1, i + BDIM_X);
                                }
                                break;
                        }
                }

                #pragma unroll
                for(int i = BDIM_X/2; i; i /= 2) {
                        __tmp = __shfl_xor_sync(WMASK, indProb, i, BDIM_X);
                        indProb = MIN(indProb, __tmp);
                }

                #ifdef DEBUG
                        __syncwarp();
                        if (tidx == 0) {
                                printf("last_cdf %10.8f\n", last_cdf);
                                printf("selected_cdf %10.8f\n", selected_cdf);
                                printf("indProb %i out of %i\n", indProb, samplm_nr);
                        }
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
                                dirs[0] = MAKE_REAL3(sphere_vertices[indProb].x,
                                                     sphere_vertices[indProb].y,
                                                     sphere_vertices[indProb].z);
                        } else {
                                dirs[0] = MAKE_REAL3(-sphere_vertices[indProb].x,
                                                     -sphere_vertices[indProb].y,
                                                     -sphere_vertices[indProb].z);
                        }
                }

                #ifdef DEBUG
                        if (tidx == 0) {
                                if ((dirs[0].x == dir.x) && (dirs[0].y == dir.y) && (dirs[0].z == dir.z)) {
                                printf("ERROR dir %10.8f, %10.8f, %10.8f\n", dirs[0].x, dirs[0].y, dirs[0].z);
                                printf("last_cdf %10.8f\n", last_cdf);
                                printf("selected_cdf %10.8f\n", selected_cdf);
                                printf("indProb %i out of %i\n", indProb, samplm_nr);
                                }
                                if (sqrt(dirs[0].x*dirs[0].x + dirs[0].y*dirs[0].y + dirs[0].z*dirs[0].z) >= 1.1) {
                                        printf("ERROR dir %10.8f, %10.8f, %10.8f\n", dirs[0].x, dirs[0].y, dirs[0].z);
                                }
                        }
                #endif
                return 1;
        }
}

template<int BDIM_X,
         int BDIM_Y,
         int NATTEMPTS,
         typename REAL_T,
         typename REAL3_T>
__device__ int get_direction_boot_d(
                                curandStatePhilox4_32_10_t *st,
                                const ModelType model_type,
                                const REAL_T max_angle,
                                const REAL_T min_signal,
                                const REAL_T relative_peak_thres,
                                const REAL_T min_separation_angle,
                                REAL3_T dir,
                                const int dimx,
                                const int dimy,
                                const int dimz,
                                const int dimt,
                                const REAL_T *__restrict__ dataf,
                                const int *__restrict__ b0s_mask, // not using this (and its opposite, dwi_mask)
                                                                  // but not clear if it will never be needed so
                                                                  // we'll keep it here for now...
                                const REAL3_T point,
                                const REAL_T *__restrict__ H, 
                                const REAL_T *__restrict__ R,
                                // model unused
                                // max_angle, pmf_threshold from global defines
                                // b0s_mask already passed
                                // min_signal from global defines
                                const int delta_nr,
                                const REAL_T *__restrict__ delta_b,
                                const REAL_T *__restrict__ delta_q, // fit_matrix
                                const int samplm_nr,
                                const REAL_T *__restrict__ sampling_matrix,
                                const REAL3_T *__restrict__ sphere_vertices,
                                const int2 *__restrict__ sphere_edges,
                                const int num_edges,
                                REAL3_T *__restrict__ dirs) {

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
	
        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

	const int n32dimt = ((dimt+31)/32)*32;

	extern REAL_T __shared__ __sh[];

	REAL_T *__vox_data_sh = reinterpret_cast<REAL_T *>(__sh);
	REAL_T *__msk_data_sh = __vox_data_sh + BDIM_Y*n32dimt;

	REAL_T *__r_sh = __msk_data_sh + BDIM_Y*n32dimt;
	REAL_T *__h_sh = __r_sh + BDIM_Y*MAX(n32dimt, samplm_nr);

	__vox_data_sh += tidy*n32dimt;
	__msk_data_sh += tidy*n32dimt;

	__r_sh += tidy*MAX(n32dimt, samplm_nr);
	__h_sh += tidy*MAX(n32dimt, samplm_nr);
	
	// compute hr_side (may be passed from python)
	int hr_side = 0;
	for(int j = tidx; j < dimt; j += BDIM_X) {
		hr_side += !b0s_mask[j] ? 1 : 0;
	}
        #pragma unroll
        for(int i = BDIM_X/2; i; i /= 2) {
                hr_side += __shfl_xor_sync(WMASK, hr_side, i, BDIM_X);
        }

        #pragma unroll
        for(int i = 0; i < NATTEMPTS; i++) {

                const int rv = trilinear_interp_d<BDIM_X>(dimx, dimy, dimz, dimt, dataf, point, __vox_data_sh);

		const int nmsk = maskGet<BDIM_X>(dimt, b0s_mask, __vox_data_sh, __msk_data_sh);

		//if (!tidx && !threadIdx.y && !blockIdx.x) {
		//
		//	printf("interp of %f, %f, %f\n", point.x, point.y, point.z);
		//	printf("hr_side: %d\n", hr_side);
		//	printArray("vox_data", 6, dimt, __vox_data_sh[tidy]);
		//	printArray("msk_data", 6, nmsk, __msk_data_sh[tidy]);
		//}
		//break;

                __syncwarp(WMASK);

                if (rv == 0) {

                        ndotp_d<BDIM_X>(hr_side, hr_side, __msk_data_sh, R, __r_sh);
			//__syncwarp();
			//printArray("__r", 5, hr_side*hr_side, R);
			//printArray("__r_sh", 6, hr_side, __r_sh[tidy]);

                        ndotp_d<BDIM_X>(hr_side, hr_side, __msk_data_sh, H, __h_sh);
			//__syncwarp();
			//printArray("__h_sh", 6, hr_side, __h_sh[tidy]);

                        __syncwarp(WMASK);

                        for(int j = 0; j < hr_side; j += BDIM_X) {
                                if (j+tidx < hr_side) {
#ifdef USE_FIXED_PERMUTATION
                                        const int srcPermInd = fixedPerm[j+tidx];
#else
                                        const int srcPermInd = curand(st) % hr_side;
//                                        if (srcPermInd < 0 || srcPermInd >= hr_side) {
//                                                printf("srcPermInd: %d\n", srcPermInd);
//                                        }
#endif
					__h_sh[j+tidx] += __r_sh[srcPermInd];
					//__h_sh[j+tidx] += __r_sh[j+tidx];
                                }
                        }
			__syncwarp(WMASK);

			//printArray("h+perm(r):", 6, hr_side, __h_sh[tidy]);
			//__syncwarp();
		
			// vox_data[dwi_mask] = masked_data
			maskPut<BDIM_X>(dimt, b0s_mask, __h_sh, __vox_data_sh);
			__syncwarp(WMASK);

			//printArray("vox_data[dwi_mask]:", 6, dimt, __vox_data_sh[tidy]);
			//__syncwarp();

			for(int j = tidx; j < dimt; j += BDIM_X) {
				//__vox_data_sh[j] = MAX(MIN_SIGNAL_P, __vox_data_sh[j]);
				__vox_data_sh[j] = MAX(min_signal, __vox_data_sh[j]);
			}
			__syncwarp(WMASK);

			const REAL_T denom = avgMask<BDIM_X>(dimt, b0s_mask, __vox_data_sh);

			for(int j = tidx; j < dimt; j += BDIM_X) {
				__vox_data_sh[j] /= denom;
			}
			__syncwarp();

			//if (!tidx && !threadIdx.y && !blockIdx.x) {
			//	printf("denom: %f\n", denom);
			//}
			////break;
			//if (!tidx && !threadIdx.y && !blockIdx.x) {
			//
			//	printf("__vox_data_sh:\n");
			//	printArray("vox_data", 6, dimt, __vox_data_sh[tidy]);
			//}
			//break;

			maskGet<BDIM_X>(dimt, b0s_mask, __vox_data_sh, __msk_data_sh);
			__syncwarp(WMASK);

                        fit_model_coef<BDIM_X>(model_type, delta_nr, hr_side, delta_q, delta_b, __msk_data_sh, __h_sh, __r_sh);

                        // __r_sh[tidy] <- python 'coef'

                        ndotp_d<BDIM_X>(samplm_nr, delta_nr, __r_sh, sampling_matrix, __h_sh);

                        // __h_sh[tidy] <- python 'pmf'
                } else {
                        #pragma unroll
                        for(int j = tidx; j < samplm_nr; j += BDIM_X) {
				__h_sh[j] = 0;
                        }
                        // __h_sh[tidy] <- python 'pmf'
                }
                __syncwarp(WMASK);
#if 0
                if (!threadIdx.y && threadIdx.x == 0) {
                        for(int j = 0; j < samplm_nr; j++) {
                                printf("pmf[%d]: %f\n", j, __h_sh[tidy][j]);
                        }
                }
                //return;
#endif
                const REAL_T abs_pmf_thr = PMF_THRESHOLD_P*max_d<BDIM_X>(samplm_nr, __h_sh, REAL_MIN);
                __syncwarp(WMASK);

                #pragma unroll
                for(int j = tidx; j < samplm_nr; j += BDIM_X) {
			const REAL_T __v = __h_sh[j];
			if (__v < abs_pmf_thr) {
				__h_sh[j] = 0;
			}
                }
                __syncwarp(WMASK);
#if 0
                if (!threadIdx.y && threadIdx.x == 0) {
                        printf("abs_pmf_thr: %f\n", abs_pmf_thr);
                        for(int j = 0; j < samplm_nr; j++) {
                                printf("pmfNORM[%d]: %f\n", j, __h_sh[tidy][j]);
                        }
                }
                //return;
#endif
#if 0
                if init:
                        directions = peak_directions(pmf, sphere)[0]
                        return directions
                else:
                        peaks = peak_directions(pmf, sphere)[0]
                        if (len(peaks) > 0):
                                return closest_peak(directions, peaks, cos_similarity)
#endif
                const int ndir = peak_directions_d<BDIM_X,
                                                   BDIM_Y>(__h_sh, dirs,
                                                           sphere_vertices,
                                                           sphere_edges,
                                                           num_edges,
							   samplm_nr,
							   reinterpret_cast<int *>(__r_sh), // reuse __r_sh as shInd in func which is large enough
							   relative_peak_thres,
							   min_separation_angle);
                if (NATTEMPTS == 1) { // init=True...
                        return ndir; // and dirs;
                } else { // init=False...
                        if (ndir > 0) {
                                /*
                                if (!threadIdx.y && threadIdx.x == 0 && ndir > 1) {
                                        printf("NATTEMPTS=5 and ndir: %d!!!\n", ndir);
                                }
                                */
                                REAL3_T peak;
                                const int foundPeak = closest_peak_d<BDIM_X, BDIM_Y, REAL_T, REAL3_T>(max_angle, dir, ndir, dirs, &peak);
                                __syncwarp(WMASK);
                                if (foundPeak) {
                                        if (tidx == 0) {
                                                dirs[0] = peak;
                                        }
                                        return 1;
                                }
                        }
                }
        }
        return 0;
}

enum {OUTSIDEIMAGE, INVALIDPOINT, TRACKPOINT, ENDPOINT};

template<int BDIM_X,
         int BDIM_Y,
         typename REAL_T,
         typename REAL3_T>
__device__ int check_point_d(const REAL_T tc_threshold,
			     const REAL3_T point,
                             const int dimx,
                             const int dimy,
                             const int dimz,
                             const REAL_T *__restrict__ metric_map) {

        const int tidy = threadIdx.y;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        __shared__ REAL_T __shInterpOut[BDIM_Y];

        const int rv = trilinear_interp_d<BDIM_X>(dimx, dimy, dimz, 1, metric_map, point, __shInterpOut+tidy);
        __syncwarp(WMASK);
#if 0
        if (threadIdx.y == 1 && threadIdx.x == 0) {
                printf("__shInterpOut[tidy]: %f, TC_THRESHOLD_P: %f\n", __shInterpOut[tidy], TC_THRESHOLD_P);
        }
#endif
        if (rv != 0) {
                return OUTSIDEIMAGE;
        }
        //return (__shInterpOut[tidy] > TC_THRESHOLD_P) ? TRACKPOINT : ENDPOINT;
        return (__shInterpOut[tidy] > tc_threshold) ? TRACKPOINT : ENDPOINT;
}

template<int BDIM_X,
         int BDIM_Y,
         typename REAL_T,
         typename REAL3_T>
__device__ int tracker_d(curandStatePhilox4_32_10_t *st,
			 const ModelType model_type,
			 const REAL_T max_angle,
			 const REAL_T min_signal,
			 const REAL_T tc_threshold,
			 const REAL_T step_size,
			 const REAL_T relative_peak_thres,
			 const REAL_T min_separation_angle,
                         REAL3_T seed,
                         REAL3_T first_step,
                         REAL3_T voxel_size,
                         const int dimx,
                         const int dimy,
                         const int dimz,
                         const int dimt,
                         const REAL_T *__restrict__ dataf,
                         const int *__restrict__ b0s_mask, // not using this (and its opposite, dwi_mask)
                         const REAL_T *__restrict__ H, 
                         const REAL_T *__restrict__ R,
                         // model unused
                         // step_size from global defines
                         // max_angle, pmf_threshold from global defines
                         // b0s_mask already passed
                         // min_signal from global defines
                         // tc_threshold from global defines
                         // pmf_threashold from global defines
                         const REAL_T *__restrict__ metric_map,
			 const int delta_nr,
                         const REAL_T *__restrict__ delta_b,
                         const REAL_T *__restrict__ delta_q, // fit_matrix
			 const int samplm_nr,
                         const REAL_T *__restrict__ sampling_matrix,
                         const REAL3_T *__restrict__ sphere_vertices,
                         const int2 *__restrict__ sphere_edges,
                         const int num_edges,
                               REAL3_T *__restrict__ __shDir,
                               int *__restrict__ nsteps,
                               REAL3_T *__restrict__ streamline) {

        const int tidx = threadIdx.x;
        //const int tidy = threadIdx.y;

        const int lid = (threadIdx.y*BDIM_X + threadIdx.x) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

        int tissue_class = TRACKPOINT;

        REAL3_T point = seed;
        REAL3_T direction = first_step;

        if (tidx == 0) {
                streamline[0] = point;
#if 0
                if (threadIdx.y == 1) {
                        printf("streamline[0]: %f, %f, %f\n", point.x, point.y, point.z);
                }
#endif
        }
        __syncwarp(WMASK);

        int i;
        for(i = 1; i < MAX_SLINE_LEN; i++) {
                int ndir;
                if (model_type == PROB) {
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
                                                        samplm_nr,
                                                        sphere_vertices,
                                                        sphere_edges,
                                                        num_edges,
                                                        __shDir);
                } else {
                        // call get_direction_boot_d() with NATTEMPTS=5
                        ndir = get_direction_boot_d<BDIM_X,
                                                    BDIM_Y,
                                                    5>(
                                                        st,
                                                        model_type,
                                                        max_angle,
                                                        min_signal,
                                                        relative_peak_thres,
                                                        min_separation_angle,
                                                        direction,
                                                        dimx, dimy, dimz, dimt, dataf,
                                                        b0s_mask /* !dwi_mask */,
                                                        point,
                                                        H, R,
                                                        // model unused
                                                        // max_angle, pmf_threshold from global defines
                                                        // b0s_mask already passed
                                                        // min_signal from global defines
                                                        delta_nr,
                                                        delta_b, delta_q, // fit_matrix
                                                        samplm_nr,
                                                        sampling_matrix,
                                                        sphere_vertices,
                                                        sphere_edges,
                                                        num_edges,
                                                        __shDir);
                }
                __syncwarp(WMASK);
                direction = __shDir[0];
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
                //point.x += (direction.x / voxel_size.x) * STEP_SIZE_P;
                //point.y += (direction.y / voxel_size.y) * STEP_SIZE_P;
                //point.z += (direction.z / voxel_size.z) * STEP_SIZE_P;
                point.x += (direction.x / voxel_size.x) * step_size;
                point.y += (direction.y / voxel_size.y) * step_size;
                point.z += (direction.z / voxel_size.z) * step_size;

                if (tidx == 0) {
                        streamline[i] = point;
#if 0
                        if (threadIdx.y == 1) {
                                printf("streamline[%d]: %f, %f, %f\n", i, point.x, point.y, point.z);
                        }
#endif
                }
                __syncwarp(WMASK);

                tissue_class = check_point_d<BDIM_X, BDIM_Y>(tc_threshold, point, dimx, dimy, dimz, metric_map);

                if (tissue_class == ENDPOINT ||
                    tissue_class == INVALIDPOINT ||
                    tissue_class == OUTSIDEIMAGE) {
                        break;
                }
        }
        nsteps[0] = i;

        return tissue_class;
}

template<int BDIM_X,
         int BDIM_Y,
         typename REAL_T,
         typename REAL3_T>
__global__ void getNumStreamlines_k(const ModelType model_type,
                                    const REAL_T max_angle,
				    const REAL_T min_signal,
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
                                    const REAL_T *__restrict__ H,
                                    const REAL_T *__restrict__ R,
				    const int delta_nr,
                                    const REAL_T *__restrict__ delta_b,
                                    const REAL_T *__restrict__ delta_q,
                                    const int  *__restrict__ b0s_mask, // change to int
				    const int samplm_nr,
                                    const REAL_T *__restrict__ sampling_matrix,
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

        REAL3_T seed = seeds[slid]; 
        // seed = lin_mat*seed + offset

        REAL3_T *__restrict__ __shDir = shDir0+slid*samplm_nr;

	// const int hr_side = dimt-1;

        curandStatePhilox4_32_10_t st;
        //curand_init(rndSeed, slid + rndOffset, DIV_UP(hr_side, BDIM_X)*tidx, &st); // each thread uses DIV_UP(hr_side/BDIM_X)
        curand_init(rndSeed, gid, 0, &st); // each thread uses DIV_UP(hr_side/BDIM_X)
                                                                                   // elements of the same sequence
        // python:
        //directions = get_direction(None, dataf, dwi_mask, sphere, s, H, R, model, max_angle,
        //                pmf_threshold, b0s_mask, min_signal, fit_matrix,
        //                sampling_matrix, init=True)

	//if (!tidx && !threadIdx.y && !blockIdx.x) {
	//	printf("seed: %f, %f, %f\n", seed.x, seed.y, seed.z);
	//}

        int ndir;
        if (model_type == PROB) {
                ndir = get_direction_prob_d<BDIM_X,
                                            BDIM_Y,
                                            1>(
                                                &st,
                                                dataf,
                                                max_angle,
                                                relative_peak_thres,
                                                min_separation_angle,
                                                MAKE_REAL3(0,0,0),
                                                dimx, dimy, dimz, dimt,
                                                seed,
                                                samplm_nr,
                                                sphere_vertices,
                                                sphere_edges,
                                                num_edges,
                                                __shDir);
        } else {
                ndir = get_direction_boot_d<BDIM_X,
                                            BDIM_Y,
                                            1>(
                                                &st,
                                                model_type,
                                                max_angle,
                                                min_signal,
                                                relative_peak_thres,
                                                min_separation_angle,
                                                MAKE_REAL3(0,0,0),
                                                dimx, dimy, dimz, dimt, dataf,
                                                b0s_mask /* !dwi_mask */,
                                                seed,
                                                H, R,
                                                // model unused
                                                // max_angle, pmf_threshold from global defines
                                                // b0s_mask already passed
                                                // min_signal from global defines
                                                delta_nr,
                                                delta_b, delta_q, // fit_matrix
                                                samplm_nr,
                                                sampling_matrix,
                                                sphere_vertices,
                                                sphere_edges,
                                                num_edges,
                                                __shDir);
        }
        if (tidx == 0) {
                slineOutOff[slid] = ndir;

		//if (!tidx && !threadIdx.y && !blockIdx.x) {
		//	printf("ndir: %d\n", ndir);
		//	for(int i = 0; i < ndir; i++) {
		//		printf("%f %f %f\n", __shDir[i].x, __shDir[i].y, __shDir[i].z);	
		//	}
		//}

        }

        return;
}

template<int BDIM_X,
         int BDIM_Y,
         typename REAL_T,
         typename REAL3_T>
__global__ void genStreamlinesMerge_k(const ModelType model_type,
				      const REAL_T max_angle,
				      const REAL_T min_signal,
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
                                      const REAL_T *__restrict__ H,
                                      const REAL_T *__restrict__ R,
				      const int delta_nr,
                                      const REAL_T *__restrict__ delta_b,
                                      const REAL_T *__restrict__ delta_q,
                                      const int    *__restrict__ b0s_mask, // change to int
                                      const REAL_T *__restrict__ metric_map,
				      const int samplm_nr,
                                      const REAL_T *__restrict__ sampling_matrix,
                                      const REAL3_T *__restrict__ sphere_vertices,
                                      const int2 *__restrict__ sphere_edges,
                                      const int num_edges,
                                      const int    *__restrict__ slineOutOff,
                                            REAL3_T *__restrict__ shDir0,
                                            REAL3_T *__restrict__ shDir1,
                                            int     *__restrict__ slineSeed,
                                            int     *__restrict__ slineLen,
                                            REAL3_T *__restrict__ sline) {

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int slid = blockIdx.x*blockDim.y + threadIdx.y;

        const int lid = (tidy*BDIM_X + tidx) % 32;
        const unsigned int WMASK = ((1ull << BDIM_X)-1) << (lid & (~(BDIM_X-1)));

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
                int stepsB;
                const int tissue_classB = tracker_d<BDIM_X,
                                                    BDIM_Y>(&st,
							    model_type,
						            max_angle,
						            min_signal,
							    tc_threshold,
							    step_size,
							    relative_peak_thres,
							    min_separation_angle,
                                                            seed,
                                                            MAKE_REAL3(-first_step.x, -first_step.y, -first_step.z),
                                                            MAKE_REAL3(1, 1, 1),
                                                            dimx, dimy, dimz, dimt, dataf,
                                                            b0s_mask,
                                                            H, R,
                                                            metric_map,
							    delta_nr,
                                                            delta_b, delta_q, //fit_matrix
							    samplm_nr,
                                                            sampling_matrix,
                                                            sphere_vertices,
                                                            sphere_edges,
                                                            num_edges,
                                                            shDir1 + slid*samplm_nr,
                                                            &stepsB,
                                                            currSline);
                //if (tidx == 0) {
                //        slineLenB[slineOff] = stepsB;
                //}

                // reverse backward sline
                for(int i = 0; i < stepsB/2; i += BDIM_X) {
                        if (i+tidx < stepsB/2) {
                                const REAL3_T __p = currSline[i+tidx];
                                currSline[i+tidx] = currSline[stepsB-1 - (i+tidx)];
                                currSline[stepsB-1 - (i+tidx)] = __p;
                        }
                }

                int stepsF;
                const int tissue_classF = tracker_d<BDIM_X,
                                                    BDIM_Y>(&st,
							    model_type,
						            max_angle,
						            min_signal,
							    tc_threshold,
							    step_size,
							    relative_peak_thres,
							    min_separation_angle,
                                                            seed,
                                                            first_step,
                                                            MAKE_REAL3(1, 1, 1),
                                                            dimx, dimy, dimz, dimt, dataf,
                                                            b0s_mask,
                                                            H, R,
                                                            metric_map,
							    delta_nr,
                                                            delta_b, delta_q, //fit_matrix
							    samplm_nr,
                                                            sampling_matrix,
                                                            sphere_vertices,
                                                            sphere_edges,
                                                            num_edges,
                                                            shDir1 + slid*samplm_nr,
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
                                    const int ngpus, const std::vector<cudaStream_t> &streams) {

  int nseeds_per_gpu = (nseeds + ngpus - 1) / ngpus;

  std::vector<int*> slinesOffs_d(ngpus, nullptr);
  std::vector<REAL3*> shDirTemp0_d(ngpus, nullptr);
  std::vector<REAL3*> shDirTemp1_d(ngpus, nullptr);

  //#pragma omp parallel for
  for (int n = 0; n < ngpus; ++n) {
    CHECK_CUDA(cudaSetDevice(n));
    int nseeds_gpu = std::min(nseeds_per_gpu, std::max(0, nseeds - n*nseeds_per_gpu));
    dim3 block(THR_X_SL, THR_X_BL/THR_X_SL);
    dim3 grid(DIV_UP(nseeds_gpu, THR_X_BL/THR_X_SL));

    CHECK_CUDA(cudaMalloc(&slinesOffs_d[n], sizeof(*slinesOffs_d[n])*(nseeds_gpu+1)));
    CHECK_CUDA(cudaMalloc(&shDirTemp0_d[n], sizeof(*shDirTemp0_d[n])*samplm_nr*grid.x*block.y));
    CHECK_CUDA(cudaMalloc(&shDirTemp1_d[n], sizeof(*shDirTemp1_d[n])*samplm_nr*grid.x*block.y));
  }

  int n32dimt = ((dimt+31)/32)*32;

  size_t shSizeGNS = sizeof(REAL)*(THR_X_BL/THR_X_SL)*(2*n32dimt + 2*MAX(n32dimt, samplm_nr)) + // for get_direction_boot_d
		     sizeof(int)*samplm_nr;						        // for peak_directions_d	

  //printf("shSizeGNS: %zu\n", shSizeGNS);

  //#pragma omp parallel for
  for (int n = 0; n < ngpus; ++n) {
    CHECK_CUDA(cudaSetDevice(n));
    int nseeds_gpu = std::min(nseeds_per_gpu, std::max(0, nseeds - n*nseeds_per_gpu));
    if (nseeds_gpu == 0) continue;
    dim3 block(THR_X_SL, THR_X_BL/THR_X_SL);
    dim3 grid(DIV_UP(nseeds_gpu, THR_X_BL/THR_X_SL));

    // Precompute number of streamlines before allocating memory
    getNumStreamlines_k<THR_X_SL,
                        THR_X_BL/THR_X_SL>
                        <<<grid, block, shSizeGNS>>>(model_type,
                                                     max_angle,
						     min_signal,
						     relative_peak_thresh,
						     min_separation_angle,
						     rng_seed,
						     rng_offset + n*nseeds_per_gpu,
						     nseeds_gpu,
						     reinterpret_cast<const REAL3 *>(seeds_d[n]),
						     dimx,
						     dimy,
						     dimz,
						     dimt,
						     dataf_d[n],
						     H_d[n],
						     R_d[n],
						     delta_nr,
						     delta_b_d[n],
						     delta_q_d[n],
						     b0s_mask_d[n],
						     samplm_nr,
						     sampling_matrix_d[n],
						     reinterpret_cast<const REAL3 *>(sphere_vertices_d[n]),
						     reinterpret_cast<const int2 *>(sphere_edges_d[n]),
						     nedges,
						     shDirTemp0_d[n],
						     slinesOffs_d[n]);
  }

  std::vector<int> slinesOffs_h;
  //#pragma omp parallel for
  for (int n = 0; n < ngpus; ++n) {
    //std::vector<int> slinesOffs_h;
    int nseeds_gpu = std::min(nseeds_per_gpu, std::max(0, nseeds - n*nseeds_per_gpu));
    if (nseeds_gpu == 0) {
      nSlines_h[n] = 0;
      continue;
    }
    slinesOffs_h.resize(nseeds_gpu+1);
    CHECK_CUDA(cudaMemcpy(slinesOffs_h.data(), slinesOffs_d[n], sizeof(*slinesOffs_h.data())*(nseeds_gpu+1), cudaMemcpyDeviceToHost));

    int __pval = slinesOffs_h[0];
    slinesOffs_h[0] = 0;
    for(int i = 1; i < nseeds_gpu+1; i++) {
      const int __cval = slinesOffs_h[i];
      slinesOffs_h[i] = slinesOffs_h[i-1] + __pval;
      __pval = __cval;
    }
    nSlines_h[n] = slinesOffs_h[nseeds_gpu];
    CHECK_CUDA(cudaMemcpy(slinesOffs_d[n], slinesOffs_h.data(), sizeof(*slinesOffs_d[n])*(nseeds_gpu+1), cudaMemcpyHostToDevice));
  }

  std::vector<int*> slineSeed_d(ngpus, nullptr);

  //#pragma omp parallel for
  for (int n = 0; n < ngpus; ++n) {
    CHECK_CUDA(cudaSetDevice(n));
    int nseeds_gpu = std::min(nseeds_per_gpu, std::max(0, nseeds - n*nseeds_per_gpu));

    CHECK_CUDA(cudaMalloc(&slineSeed_d[n], sizeof(*slineSeed_d[n])*nSlines_h[n]));
    CHECK_CUDA(cudaMemset(slineSeed_d[n], -1, sizeof(*slineSeed_d[n])*nSlines_h[n]));

    // Allocate/reallocate output arrays if necessary
    if (nSlines_h[n] > nSlines_old_h[n]) {
      if(slines_h[n]) cudaFreeHost(slines_h[n]);
      if(slinesLen_h[n]) cudaFreeHost(slinesLen_h[n]);
      slines_h[n] = nullptr;
      slinesLen_h[n] = nullptr;
    }

    if (!slines_h[n]) CHECK_CUDA(cudaMallocHost(&slines_h[n], 2*3*MAX_SLINE_LEN*nSlines_h[n]*sizeof(*slines_h[n])));
    if (!slinesLen_h[n]) CHECK_CUDA(cudaMallocHost(&slinesLen_h[n], nSlines_h[n]*sizeof(*slinesLen_h[n])));
  }

  //if (nSlines_h) {

  std::vector<int*> slineLen_d(ngpus, nullptr);
  std::vector<REAL3*> sline_d(ngpus, nullptr);
  //#pragma omp parallel for
  for (int n = 0; n < ngpus; ++n) {
    CHECK_CUDA(cudaSetDevice(n));
    CHECK_CUDA(cudaMalloc(&slineLen_d[n], sizeof(*slineLen_d[n])*nSlines_h[n]));

    CHECK_CUDA(cudaMalloc(&sline_d[n], sizeof(*sline_d[n])*2*MAX_SLINE_LEN*nSlines_h[n]));

#if 0
    size_t free_mem, total_mem;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    std::cerr << "GPU " << n << ": ";
    std::cerr << "GPU Memory Usage before genStreamlinesMerge_k: ";
    std::cerr << (total_mem-free_mem)/(1024*1024) << " MiB used, ";
    std::cerr << total_mem/(1024*1024) << " MiB total ";
    std::cerr << std::endl;
#endif
  }

  //#pragma omp parallel for
  for (int n = 0; n < ngpus; ++n) {
    CHECK_CUDA(cudaSetDevice(n));
    int nseeds_gpu = std::min(nseeds_per_gpu, std::max(0, nseeds - n*nseeds_per_gpu));
    if (nseeds_gpu == 0) continue;
    dim3 block(THR_X_SL, THR_X_BL/THR_X_SL);
    dim3 grid(DIV_UP(nseeds_gpu, THR_X_BL/THR_X_SL));
#if 0
    std::cerr << "GPU " << n << ": ";
    std::cerr << "Generating " << nSlines_h[n] << " streamlines (from " << nseeds_gpu << " seeds)" << std::endl; 
#endif
    //fprintf(stderr, "Launching kernel with %u blocks of size (%u, %u)\n", grid.x, block.x, block.y);
    genStreamlinesMerge_k<THR_X_SL,
                          THR_X_BL/THR_X_SL>
                          <<<grid, block, shSizeGNS, streams[n]>>>(model_type,
								   max_angle,
								   min_signal,
								   tc_threshold,
								   step_size,
								   relative_peak_thresh,
								   min_separation_angle,
								   rng_seed,
								   rng_offset + n*nseeds_per_gpu,
								   nseeds_gpu,
								   reinterpret_cast<const REAL3 *>(seeds_d[n]),
								   dimx,
								   dimy,
								   dimz,
								   dimt,
								   dataf_d[n],
								   H_d[n],
								   R_d[n],
								   delta_nr,
								   delta_b_d[n],
								   delta_q_d[n],
								   b0s_mask_d[n],
								   metric_map_d[n],
								   samplm_nr,
								   sampling_matrix_d[n],
								   reinterpret_cast<const REAL3 *>(sphere_vertices_d[n]),
								   reinterpret_cast<const int2 *>(sphere_edges_d[n]),
								   nedges,
								   slinesOffs_d[n],
								   shDirTemp0_d[n],
								   shDirTemp1_d[n],
								   slineSeed_d[n],
								   slineLen_d[n],
								   sline_d[n]);
    CHECK_ERROR("genStreamlinesMerge_k");
  }

  //CHECK_CUDA(cudaDeviceSynchronize());

  //#pragma omp parallel for
  for (int n = 0; n < ngpus; ++n) {
    CHECK_CUDA(cudaSetDevice(n));
    CHECK_CUDA(cudaMemcpyAsync(slines_h[n],
                          reinterpret_cast<REAL *>(sline_d[n]),
                          sizeof(*slines_h[n])*2*MAX_SLINE_LEN*nSlines_h[n]*3,
                          cudaMemcpyDeviceToHost, streams[n]));
    CHECK_CUDA(cudaMemcpyAsync(slinesLen_h[n],
                          slineLen_d[n],
                          sizeof(*slinesLen_h[n])*nSlines_h[n],
                          cudaMemcpyDeviceToHost, streams[n]));

  }
  //};

  //#pragma omp parallel for
  for (int n = 0; n < ngpus; ++n) {
    CHECK_CUDA(cudaSetDevice(n));
    CHECK_CUDA(cudaStreamSynchronize(streams[n]));
    CHECK_CUDA(cudaFree(slineSeed_d[n]));
    CHECK_CUDA(cudaFree(slinesOffs_d[n]));
    CHECK_CUDA(cudaFree(shDirTemp0_d[n]));
    CHECK_CUDA(cudaFree(shDirTemp1_d[n]));
    CHECK_CUDA(cudaFree(slineLen_d[n]));
    CHECK_CUDA(cudaFree(sline_d[n]));
  }

}

#if 1
void write_trk(const char *fname,
               const /*short*/ int *dims,
               const REAL *voxel_size,
               const char *voxel_order,
               const REAL *vox_to_ras,
               const int nsline,
               const int *slineLen,
               const REAL3 *sline) {

        FILE *fp = fopen(fname, "w");
        if (!fp) {
                fprintf(stderr, "Cannot open file %s for writing...\n", fname);
                exit(EXIT_FAILURE);
        }

        const char ID_STRING[6] = "TRACK";
        short DIM[3] = {1, 1, 1};
        float VOXEL_SIZE[3] = {1.0f, 1.0f, 1.0f};
        float VOX_TO_RAS[4][4] = {{1.0f, 0.0f, 0.0, 0.0f},
                                  {0.0f, 1.0f, 0.0, 0.0f},
                                  {0.0f, 0.0f, 1.0, 0.0f},
                                  {0.0f, 0.0f, 0.0, 1.0f}};
        //const char VOXEL_ORDER[2][4] = {"RAS", "LAS"};
        const float ORIGIN[3] = {0.0f, 0.0f, 0.0f};
        const float IMAGE_ORIENTATION_PATIENT[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const int VERSION = 2;
        const int HDR_SIZE = 1000;

        // write header
        unsigned char header[1000];
        memset(&header[0], 0, sizeof(header));

        long long int off = 0;

        memcpy(header, ID_STRING, sizeof(ID_STRING));
        off += sizeof(ID_STRING);

        if (dims) {
                DIM[0] = dims[0];
                DIM[1] = dims[1];
                DIM[2] = dims[2];
        }
        memcpy(header+off, DIM, sizeof(DIM));
        off += sizeof(DIM);

        if (voxel_size) {
                VOXEL_SIZE[0] = (float)voxel_size[0];
                VOXEL_SIZE[1] = (float)voxel_size[1];
                VOXEL_SIZE[2] = (float)voxel_size[2];
        }
        memcpy(header+off, VOXEL_SIZE, sizeof(VOXEL_SIZE));
        off += sizeof(VOXEL_SIZE);

        memcpy(header+off, ORIGIN, sizeof(ORIGIN));
        off += sizeof(ORIGIN);

        // skip n_scalaer(2b) + scalar_name(200b) +
        //      n_properties(2b) + property_name(200b)
        off += 404;

        if (vox_to_ras) {
                for(int i = 0; i < 4; i++) {
                        for(int j = 0; j < 4; j++) {
                                VOX_TO_RAS[i][j] = (float)vox_to_ras[i*4+j];
                        }
                }
        }
        memcpy(header+off, VOX_TO_RAS, sizeof(VOX_TO_RAS));
        off += sizeof(VOX_TO_RAS);

        // skip reserved(444b)
        off += 444;

        if (voxel_order) {
                memcpy(header+off, voxel_order, 4);
        } else {
                memcpy(header+off, "LAS", 4);
        }
        off += 4; //sizeof(VOXEL_ORDER[voxel_order]);

        // skip pad2(4b)
        off += 4;

        memcpy(header+off, IMAGE_ORIENTATION_PATIENT, sizeof(IMAGE_ORIENTATION_PATIENT));
        off += sizeof(IMAGE_ORIENTATION_PATIENT);

        // skip pad1(2b)
        off += 2;

        // skip invert_x(1b), invert_y(1b), invert_x(1b), swap_xy(1b), swap_yz(1b), swap_zx(1b)
        off += 6;

        memcpy(header+off, &nsline, sizeof(int));
        off += sizeof(int);

        memcpy(header+off, &VERSION, sizeof(VERSION));
        off += sizeof(VERSION);

        memcpy(header+off, &HDR_SIZE, sizeof(HDR_SIZE));
        off += sizeof(HDR_SIZE);

        //assert(off == 1000);
        if (off != 1000) {
                fprintf(stderr, "%s:%s:%d: heder size = %lld, (!= 1000)!\n", __FILE__, __func__, __LINE__, off);
                exit(EXIT_FAILURE);
        }
        
        size_t nw = fwrite(header, sizeof(header), 1, fp);
        if (nw != 1) {
                fprintf(stderr, "Error while writing to file!\n");
                exit(EXIT_FAILURE);
        }
#if 0
        // write body
        long long maxSlineLen = slineLen[0];
        for(long long i = 1; i < nsline; i++) {
                maxSlineLen = MAX(maxSlineLen, slineLen[i]);
        }

        float *slineData = (float *)Malloc((1+3*maxSlineLen)*sizeof(*slineData));
#else
        float slineData[1 + 3*(2*MAX_SLINE_LEN)];
#endif
        for(int i = 0; i < nsline; i++) {
                reinterpret_cast<int *>(slineData)[0] = slineLen[i];
                for(int j = 0; j < slineLen[i]; j++) {
                        slineData[1+3*j+0] = (float)((sline[i*2*MAX_SLINE_LEN + j].x+0.5)*VOXEL_SIZE[0]);
                        slineData[1+3*j+1] = (float)((sline[i*2*MAX_SLINE_LEN + j].y+0.5)*VOXEL_SIZE[1]);
                        slineData[1+3*j+2] = (float)((sline[i*2*MAX_SLINE_LEN + j].z+0.5)*VOXEL_SIZE[2]);
                }
                nw = fwrite(slineData, (1+3*slineLen[i])*sizeof(*slineData), 1, fp);
                if (nw != 1) {
                        fprintf(stderr, "Error while writing to file!\n");
                        exit(EXIT_FAILURE);
                }
        }
#if 0
        free(slineData);
#endif
        fclose(fp);

        return;
}
#else
void write_trk(const int num_threads,
               const char *fname,
               const /*short*/ int *dims,
               const REAL *voxel_size,
               const char *voxel_order,
               const REAL *vox_to_ras,
               const int nsline,
               const int *slineLen,
               const REAL3 *sline) {

        FILE *fp = fopen(fname, "w");
        if (!fp) {
                fprintf(stderr, "Cannot open file %s for writing...\n", fname);
                exit(EXIT_FAILURE);
        }

        const char ID_STRING[6] = "TRACK";
        short DIM[3] = {1, 1, 1};
        float VOXEL_SIZE[3] = {1.0f, 1.0f, 1.0f};
        float VOX_TO_RAS[4][4] = {{1.0f, 0.0f, 0.0, 0.0f},
                                  {0.0f, 1.0f, 0.0, 0.0f},
                                  {0.0f, 0.0f, 1.0, 0.0f},
                                  {0.0f, 0.0f, 0.0, 1.0f}};
        //const char VOXEL_ORDER[2][4] = {"RAS", "LAS"};
        const float ORIGIN[3] = {0.0f, 0.0f, 0.0f};
        const float IMAGE_ORIENTATION_PATIENT[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const int VERSION = 2;
        const int HDR_SIZE = 1000;

        // write header
        unsigned char header[1000];
        memset(&header[0], 0, sizeof(header));

        long long int off = 0;

        memcpy(header, ID_STRING, sizeof(ID_STRING));
        off += sizeof(ID_STRING);

        if (dims) {
                DIM[0] = dims[0];
                DIM[1] = dims[1];
                DIM[2] = dims[2];
        }
        memcpy(header+off, DIM, sizeof(DIM));
        off += sizeof(DIM);

        if (voxel_size) {
                VOXEL_SIZE[0] = (float)voxel_size[0];
                VOXEL_SIZE[1] = (float)voxel_size[1];
                VOXEL_SIZE[2] = (float)voxel_size[2];
        }
        memcpy(header+off, VOXEL_SIZE, sizeof(VOXEL_SIZE));
        off += sizeof(VOXEL_SIZE);

        memcpy(header+off, ORIGIN, sizeof(ORIGIN));
        off += sizeof(ORIGIN);

        // skip n_scalaer(2b) + scalar_name(200b) +
        //      n_properties(2b) + property_name(200b)
        off += 404;

        if (vox_to_ras) {
                for(int i = 0; i < 4; i++) {
                        for(int j = 0; j < 4; j++) {
                                VOX_TO_RAS[i][j] = (float)vox_to_ras[i*4+j];
                        }
                }
        }
        memcpy(header+off, VOX_TO_RAS, sizeof(VOX_TO_RAS));
        off += sizeof(VOX_TO_RAS);

        // skip reserved(444b)
        off += 444;

        if (voxel_order) {
                memcpy(header+off, voxel_order, 4);
        } else {
                memcpy(header+off, "LAS", 4);
        }
        off += 4; //sizeof(VOXEL_ORDER[voxel_order]);

        // skip pad2(4b)
        off += 4;

        memcpy(header+off, IMAGE_ORIENTATION_PATIENT, sizeof(IMAGE_ORIENTATION_PATIENT));
        off += sizeof(IMAGE_ORIENTATION_PATIENT);

        // skip pad1(2b)
        off += 2;

        // skip invert_x(1b), invert_y(1b), invert_x(1b), swap_xy(1b), swap_yz(1b), swap_zx(1b)
        off += 6;

        memcpy(header+off, &nsline, sizeof(int));
        off += sizeof(int);

        memcpy(header+off, &VERSION, sizeof(VERSION));
        off += sizeof(VERSION);

        memcpy(header+off, &HDR_SIZE, sizeof(HDR_SIZE));
        off += sizeof(HDR_SIZE);

        //assert(off == 1000);
        if (off != 1000) {
                fprintf(stderr, "%s:%s:%d: heder size = %lld, (!= 1000)!\n", __FILE__, __func__, __LINE__, off);
                exit(EXIT_FAILURE);
        }
        
        size_t nw = fwrite(header, sizeof(header), 1, fp);
        if (nw != 1) {
                fprintf(stderr, "Error while writing to file!\n");
                exit(EXIT_FAILURE);
        }

        // write body
        long long maxSlineLen = slineLen[0];
        for(long long i = 1; i < nsline; i++) {
                maxSlineLen = MAX(maxSlineLen, slineLen[i]);
        }

        //omp_set_dynamic(0);
        const int NTHREADS = num_threads > 0 ? num_threads : 1;
        omp_set_num_threads(NTHREADS);

        const int NFLTS_PER_TH = 1 + 2*(3*MAX_SLINE_LEN);
        float *slineData = (float *)Malloc(NFLTS_PER_TH*NTHREADS*sizeof(*slineData));

        #pragma omp parallel 
        {
                const int tid = omp_get_thread_num();
                float *__mySlineData = slineData+tid*NFLTS_PER_TH;
#if 1
                //#pragma omp for schedule(static)
                for(int i = 0; i < nsline; i += NTHREADS) {
                        if (i+tid < nsline) {
                                reinterpret_cast<int *>(__mySlineData)[0] = slineLen[i+tid];
                                for(int j = 0; j < slineLen[i+tid]; j++) {
                                        __mySlineData[1+3*j+0] = (float)((sline[(i+tid)*2*MAX_SLINE_LEN + j].x+0.5)*VOXEL_SIZE[0]);
                                        __mySlineData[1+3*j+1] = (float)((sline[(i+tid)*2*MAX_SLINE_LEN + j].y+0.5)*VOXEL_SIZE[1]);
                                        __mySlineData[1+3*j+2] = (float)((sline[(i+tid)*2*MAX_SLINE_LEN + j].z+0.5)*VOXEL_SIZE[2]);
                                }
                        }
                        #pragma omp barrier
                        if (tid == 0) {
                                for(int j = 0; j < NTHREADS; j++) {
                                        if (i+j >= nsline) {
                                               break;
                                        }
                                        nw = fwrite(slineData+j*NFLTS_PER_TH, (1+3*slineLen[i+j])*sizeof(*slineData), 1, fp);
                                        if (nw != 1) {
                                                fprintf(stderr, "Error while writing to file!\n");
                                                exit(EXIT_FAILURE);
                                        }
                                }
                        }
                        #pragma omp barrier
                }
#else
                // streamlines are not required to be in any specific order inside the trk file...
                #pragma omp for
                for(int i = 0; i < nsline; i++) {
                        reinterpret_cast<int *>(__mySlineData)[0] = slineLen[i];
                        for(int j = 0; j < slineLen[i]; j++) {
                                __mySlineData[1+3*j+0] = (float)((sline[i*2*MAX_SLINE_LEN + j].x+0.5)*VOXEL_SIZE[0]);
                                __mySlineData[1+3*j+1] = (float)((sline[i*2*MAX_SLINE_LEN + j].y+0.5)*VOXEL_SIZE[1]);
                                __mySlineData[1+3*j+2] = (float)((sline[i*2*MAX_SLINE_LEN + j].z+0.5)*VOXEL_SIZE[2]);
                        }
                        nw = fwrite(__mySlineData, (1+3*slineLen[i])*sizeof(*__mySlineData), 1, fp);
                        if (nw != 1) {
                                fprintf(stderr, "Error while writing to file!\n");
                                exit(EXIT_FAILURE);
                        }
                }
#endif
        }
        free(slineData);
        fclose(fp);

        return;
}
#endif
