#ifndef __PTT_CUH__
#define __PTT_CUH__

#include "disc.h"
#include "globals.h"

#define STEP_FRAC 20 // divides output step size (usually 0.5) into this many internal steps
#define PROBE_FRAC 2 // divides output step size (usually 0.5) to find probe length
#define PROBE_QUALITY 4
#define SAMPLING_QUALITY 4 // can be 2-7, should be at least 4 to take advantage of GPUs
#define PROBABILISTIC_BIAS 1 // 1 looks good. can be 0-log_2(N_WARPS) (typically 0-5). 0 is fully probabilistic, 4 is close to deterministic.
#define ALLOW_WEAK_LINK 1
#define INIT_FRAME_TRIES 16
#define TRIES_PER_REJECTION_SAMPLING 1024
#define DEFAULT_PTT_MINDATASUPPORT 0.05
#define K_SMALL 0.0001

#define NORM_MIN_SUPPORT (DEFAULT_PTT_MINDATASUPPORT * PROBE_QUALITY)
#define PROBABILISTIC_GROUP_SZ POW2(PROBABILISTIC_BIAS)

#if SAMPLING_QUALITY == 2
#define DISC_VERT_CNT DISC_2_VERT_CNT
#define DISC_FACE_CNT DISC_2_FACE_CNT
__device__ __constant__ REAL DISC_VERT[DISC_VERT_CNT*2] = DISC_2_VERT;
__device__ __constant__ int DISC_FACE[DISC_FACE_CNT*3] = DISC_2_FACE;
#elif SAMPLING_QUALITY == 3
#define DISC_VERT_CNT DISC_3_VERT_CNT
#define DISC_FACE_CNT DISC_3_FACE_CNT
__device__ __constant__ REAL DISC_VERT[DISC_VERT_CNT*2] = DISC_3_VERT;
__device__ __constant__ int DISC_FACE[DISC_FACE_CNT*3] = DISC_3_FACE;
#elif SAMPLING_QUALITY == 4
#define DISC_VERT_CNT DISC_4_VERT_CNT
#define DISC_FACE_CNT DISC_4_FACE_CNT
__device__ __constant__ REAL DISC_VERT[DISC_VERT_CNT*2] = DISC_4_VERT;
__device__ __constant__ int DISC_FACE[DISC_FACE_CNT*3] = DISC_4_FACE;
#elif SAMPLING_QUALITY == 5
#define DISC_VERT_CNT DISC_5_VERT_CNT
#define DISC_FACE_CNT DISC_5_FACE_CNT
__device__ __constant__ REAL DISC_VERT[DISC_VERT_CNT*2] = DISC_5_VERT;
__device__ __constant__ int DISC_FACE[DISC_FACE_CNT*3] = DISC_5_FACE;
#elif SAMPLING_QUALITY == 6
#define DISC_VERT_CNT DISC_6_VERT_CNT
#define DISC_FACE_CNT DISC_6_FACE_CNT
__device__ __constant__ REAL DISC_VERT[DISC_VERT_CNT*2] = DISC_6_VERT;
__device__ __constant__ int DISC_FACE[DISC_FACE_CNT*3] = DISC_6_FACE;
#elif SAMPLING_QUALITY == 7
#define DISC_VERT_CNT DISC_7_VERT_CNT
#define DISC_FACE_CNT DISC_7_FACE_CNT
__device__ __constant__ REAL DISC_VERT[DISC_VERT_CNT*2] = DISC_7_VERT; // TODO: check if removing __constant__ helps or hurts
__device__ __constant__ int DISC_FACE[DISC_FACE_CNT*3] = DISC_7_FACE;
#endif

#endif