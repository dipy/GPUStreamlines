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

#ifndef __CUWSORT_H__
#define __CUWSORT_H__

#define __CWSORT_MIN(a,b)        (((a)<(b))?(a):(b))
#define __CWSORT_MAX(a,b)        (((a)>(b))?(a):(b))

namespace cuwsort {

__device__ __constant__
int swap32[15][32] = {{16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
                      { 8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23},
                      { 4,  5,  6,  7,  0,  1,  2,  3, 16, 17, 18, 19, 20, 21, 22, 23,  8,  9, 10, 11, 12, 13, 14, 15, 28, 29, 30, 31, 24, 25, 26, 27},
                      { 2,  3,  0,  1,  4,  5,  6,  7, 12, 13, 14, 15,  8,  9, 10, 11, 20, 21, 22, 23, 16, 17, 18, 19, 24, 25, 26, 27, 30, 31, 28, 29},
                      { 1,  0,  2,  3, 16, 17, 18, 19,  8,  9, 10, 11, 24, 25, 26, 27,  4,  5,  6,  7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 31, 30},
                      { 0,  1,  2,  3,  8,  9, 10, 11,  4,  5,  6,  7, 16, 17, 18, 19, 12, 13, 14, 15, 24, 25, 26, 27, 20, 21, 22, 23, 28, 29, 30, 31},
                      { 0,  1,  2,  3,  6,  7,  4,  5, 10, 11,  8,  9, 14, 15, 12, 13, 18, 19, 16, 17, 22, 23, 20, 21, 26, 27, 24, 25, 28, 29, 30, 31},
                      { 0,  1, 16, 17,  4,  5, 20, 21,  8,  9, 24, 25, 12, 13, 28, 29,  2,  3, 18, 19,  6,  7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31},
                      { 0,  1,  8,  9,  4,  5, 12, 13,  2,  3, 16, 17,  6,  7, 20, 21, 10, 11, 24, 25, 14, 15, 28, 29, 18, 19, 26, 27, 22, 23, 30, 31},
                      { 0,  1,  4,  5,  2,  3,  8,  9,  6,  7, 12, 13, 10, 11, 16, 17, 14, 15, 20, 21, 18, 19, 24, 25, 22, 23, 28, 29, 26, 27, 30, 31},
                      { 0,  1,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 30, 31},
                      { 0, 16,  2, 18,  4, 20,  6, 22,  8, 24, 10, 26, 12, 28, 14, 30,  1, 17,  3, 19,  5, 21,  7, 23,  9, 25, 11, 27, 13, 29, 15, 31},
                      { 0,  8,  2, 10,  4, 12,  6, 14,  1, 16,  3, 18,  5, 20,  7, 22,  9, 24, 11, 26, 13, 28, 15, 30, 17, 25, 19, 27, 21, 29, 23, 31},
                      { 0,  4,  2,  6,  1,  8,  3, 10,  5, 12,  7, 14,  9, 16, 11, 18, 13, 20, 15, 22, 17, 24, 19, 26, 21, 28, 23, 30, 25, 29, 27, 31},
                      { 0,  2,  1,  4,  3,  6,  5,  8,  7, 10,  9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 31}};

__device__ __constant__
int swap16[10][16] = {{ 8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7},
                      { 4,  5,  6,  7,  0,  1,  2,  3, 12, 13, 14, 15,  8,  9, 10, 11},
                      { 2,  3,  0,  1,  8,  9, 10, 11,  4,  5,  6,  7, 14, 15, 12, 13},
                      { 1,  0,  2,  3,  6,  7,  4,  5, 10, 11,  8,  9, 12, 13, 15, 14},
                      { 0,  1,  8,  9,  4,  5, 12, 13,  2,  3, 10, 11,  6,  7, 14, 15},
                      { 0,  1,  4,  5,  2,  3,  8,  9,  6,  7, 12, 13, 10, 11, 14, 15},
                      { 0,  1,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 14, 15},
                      { 0,  8,  2, 10,  4, 12,  6, 14,  1,  9,  3, 11,  5, 13,  7, 15},
                      { 0,  4,  2,  6,  1,  8,  3, 10,  5, 12,  7, 14,  9, 13, 11, 15},
                      { 0,  2,  1,  4,  3,  6,  5,  8,  7, 10,  9, 12, 11, 14, 13, 15}};

__device__ __constant__
int swap8[6][8] = {{ 4,  5,  6,  7,  0,  1,  2,  3},
                   { 2,  3,  0,  1,  6,  7,  4,  5},
                   { 1,  0,  4,  5,  2,  3,  7,  6},
                   { 0,  1,  3,  2,  5,  4,  6,  7},
                   { 0,  4,  2,  6,  1,  5,  3,  7},
                   { 0,  2,  1,  4,  3,  6,  5,  7}};

__device__ __constant__
int swap4[3][4] = {{ 2,  3,  0,  1},
                   { 1,  0,  3,  2},
                   { 0,  2,  1,  3}};

__device__ __constant__
int swap2[1][2] = {{ 1,  0}};

__device__ __constant__ const int *__swaps[] = {NULL,
						reinterpret_cast<const int *>(&swap2[0][0]),
						reinterpret_cast<const int *>(&swap4[0][0]),
						reinterpret_cast<const int *>(&swap8[0][0]),
						reinterpret_cast<const int *>(&swap16[0][0]),
						reinterpret_cast<const int *>(&swap32[0][0])};

template<int X>
struct STATIC_LOG2 {
        enum {value = 1+STATIC_LOG2<X/2>::value};
};

template<>
struct STATIC_LOG2<1> {
        enum {value = 0};
};

enum {WSORT_DIR_DEC, WSORT_DIR_INC};

template<int WSIZE,
	 int GSIZE, // power-pf-2 <= WSIZE
	 int DIRECTION,
	 typename KEY_T>
__device__  KEY_T warp_sort(KEY_T v) {

	const int NET_LEN[] = {0, 1, 3, 6, 10, 15};
	const int LOG2_GSIZE = STATIC_LOG2<GSIZE>::value;
	const int NSWAP = NET_LEN[LOG2_GSIZE];

	const int lid = (threadIdx.y*blockDim.x + threadIdx.x) % WSIZE;
        const unsigned int WMASK = ((1ull << GSIZE)-1) << (lid & (~(GSIZE-1)));

	const int gid = lid % GSIZE;

	const int (*swap)[GSIZE] = reinterpret_cast<const int (*)[GSIZE]>(__swaps[LOG2_GSIZE]);

        #pragma unroll
        for(int i = 0; i < NSWAP; i++) {
                const int srclane = swap[i][gid];
                const KEY_T a = __shfl_sync(WMASK, v, srclane, GSIZE);
                v = (gid < srclane == DIRECTION) ? __CWSORT_MIN(a, v) : __CWSORT_MAX(a, v);
        }
        return v;
}

template<int WSIZE,
	 int GSIZE, // power-pf-2 <= WSIZE
	 int DIRECTION,
	 typename KEY_T,
	 typename VAL_T>
__device__  void warp_sort(KEY_T *__restrict__ k, VAL_T *__restrict__ v) {
	
	const int NET_LEN[] = {0, 1, 3, 6, 10, 15};
	const int LOG2_GSIZE = STATIC_LOG2<GSIZE>::value;
	const int NSWAP = NET_LEN[LOG2_GSIZE];

	const int lid = (threadIdx.y*blockDim.x + threadIdx.x) % WSIZE;
        const unsigned int WMASK = ((1ull << GSIZE)-1) << (lid & (~(GSIZE-1)));
	
	const int gid = lid % GSIZE;

	const int (*swap)[GSIZE] = reinterpret_cast<const int (*)[GSIZE]>(__swaps[LOG2_GSIZE]);

        #pragma unroll
        for(int i = 0; i < NSWAP; i++) {
                const int srclane = swap[i][gid];

                const KEY_T a = __shfl_sync(WMASK, k[0], srclane, GSIZE);
                const VAL_T b = __shfl_sync(WMASK, v[0], srclane, GSIZE);

                if (gid < srclane == DIRECTION) {
                        if (a < k[0]) {
                                k[0] = a;
                                v[0] = b;
                        }
                } else {
                        if (a > k[0]) {
                                k[0] = a;
                                v[0] = b;
                        }
                }
        }
        return;
}

}
#endif
