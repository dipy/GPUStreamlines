#ifndef __PTT_CUH__
#define __PTT_CUH__

#include "disc.h"
#include "globals.h"

#define STEP_FRAC 20 // divides output step size (usually 0.5) into this many steps
#define PROBE_LENGTH 0.25
#define PROBE_QUALITY 4 // can be 4-7
#define ALLOW_WEAK_LINK 1
#define TRIES_PER_REJECTION_SAMPLING 1024
#define DEFAULT_PTT_MINDATASUPPORT 0.05
#define K_SMALL 0.0001
#define PURE_PROBABILISTIC 0


#define NORM_MIN_SUPPORT (DEFAULT_PTT_MINDATASUPPORT * PROBE_QUALITY)
#define PROBE_STEP (PROBE_LENGTH / (PROBE_QUALITY - 1))

#if PROBE_QUALITY == 4
#define DISC_VERT_CNT DISC_4_VERT_CNT
#define DISC_FACE_CNT DISC_4_FACE_CNT
__device__ __constant__ REAL DISC_VERT[DISC_VERT_CNT*2] = DISC_4_VERT;
__device__ __constant__ int DISC_FACE[DISC_FACE_CNT*3] = DISC_4_FACE;
#elif PROBE_QUALITY == 5
#define DISC_VERT_CNT DISC_5_VERT_CNT
#define DISC_FACE_CNT DISC_5_FACE_CNT
__device__ __constant__ REAL DISC_VERT[DISC_VERT_CNT*2] = DISC_5_VERT;
__device__ __constant__ int DISC_FACE[DISC_FACE_CNT*3] = DISC_5_FACE;
#elif PROBE_QUALITY == 6
#define DISC_VERT_CNT DISC_6_VERT_CNT
#define DISC_FACE_CNT DISC_6_FACE_CNT
__device__ __constant__ REAL DISC_VERT[DISC_VERT_CNT*2] = DISC_6_VERT;
__device__ __constant__ int DISC_FACE[DISC_FACE_CNT*3] = DISC_6_FACE;
#elif PROBE_QUALITY == 7
#define DISC_VERT_CNT DISC_7_VERT_CNT
#define DISC_FACE_CNT DISC_7_FACE_CNT
__device__ __constant__ REAL DISC_VERT[DISC_VERT_CNT*2] = DISC_7_VERT; // TODO: check if removing __constant__ helps or hurts
__device__ __constant__ int DISC_FACE[DISC_FACE_CNT*3] = DISC_7_FACE;
#endif

#endif