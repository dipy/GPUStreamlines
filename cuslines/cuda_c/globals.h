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

#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#define REAL_SIZE 4

#if REAL_SIZE == 4

#define REAL		float
#define REAL3		float3
#define MAKE_REAL3	make_float3
#define RCONV		"%f"
#define FLOOR		floorf
#define LOG		__logf
#define EXP		__expf
#define REAL_MAX	__int_as_float(0x7f7fffffU)
#define REAL_MIN	(-REAL_MAX)
#define COS		__cosf
#define SIN		__sinf
#define FABS		fabsf
#define SQRT		sqrtf
#define RSQRT		rsqrtf
#define ACOS		acosf

#elif REAL_SIZE == 8

#define REAL		double
#define REAL3		double3
#define MAKE_REAL3	make_double3
#define RCONV		"%lf"
#define FLOOR		floor
#define LOG		log
#define EXP		exp
#define REAL_MAX	__longlong_as_double(0x7fefffffffffffffLL)
#define REAL_MIN	(-REAL_MAX)
#define COS		cos
#define SIN		sin
#define FABS		fabs
#define SQRT		sqrt
#define RSQRT		rsqrt
#define ACOS		acos

#endif
// TODO: half this in when WMGMI seeding
#define MAX_SLINE_LEN	(501)
#define PMF_THRESHOLD_P	((REAL)0.05)

#define THR_X_BL (64)
#define THR_X_SL (32)

#define MAX_SLINES_PER_SEED (10)

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))
#define POW2(n) (1 << (n))

#define DIV_UP(a,b) (((a)+((b)-1))/(b))

#define EXCESS_ALLOC_FACT 2

#define NORM_EPS ((REAL)1e-8)

#if 0
  #define DEBUG
#endif

enum ModelType {
  OPDT = 0,
  CSA = 1,
  PROB = 2,
  PTT = 3,
};

enum {OUTSIDEIMAGE, INVALIDPOINT, TRACKPOINT, ENDPOINT};

#endif
