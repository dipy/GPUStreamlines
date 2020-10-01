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

#define REAL_SIZE 8

#if REAL_SIZE == 4

#define REAL		float
#define REAL3		float3
#define MAKE_REAL3	make_float3
#define RCONV		"%f"
#define FLOOR		floorf
#define LOG		__logf
#define EXP		__expf
#define REAL_MAX	(FLT_MAX)
#define REAL_MIN	(-REAL_MAX)
#define COS		__cosf
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
#define REAL_MAX	(DBL_MAX)
#define REAL_MIN	(-REAL_MAX)
#define COS		cos
#define FABS		fabs
#define SQRT		sqrt
#define RSQRT		rsqrt
#define ACOS		acos

#endif

#define MAX_SLINE_LEN	(501)
#define PMF_THRESHOLD_P	((REAL)0.1)

//#define TC_THRESHOLD_P	((REAL)0.1)
//#define STEP_SIZE_P	((REAL)0.5)  // only for TRK generation
//#define MAX_ANGLE_P	((REAL)1.0471975511965976) // 60 deg in radians
//#define MIN_SIGNAL_P	((REAL)1.0)

#define MAX_SLINES_PER_SEED (10)

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

#define DIV_UP(a,b) (((a)+((b)-1))/(b))

#endif
