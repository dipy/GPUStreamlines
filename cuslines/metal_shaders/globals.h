/* Metal-adapted globals — mirrors cuslines/cuda_c/globals.h.
 * Metal only supports float (no double), so REAL_SIZE is always 4.
 */

#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#include <metal_stdlib>
using namespace metal;

// ── precision ────────────────────────────────────────────────────────
#define REAL_SIZE 4

#define REAL        float
#define FLOOR       floor
#define LOG         fast::log
#define EXP         fast::exp
#define COS         fast::cos
#define SIN         fast::sin
#define FABS        abs
#define SQRT        sqrt
#define RSQRT       rsqrt
#define ACOS        acos
#define REAL_MAX    FLT_MAX
#define REAL_MIN    (-FLT_MAX)

// ── geometry constants ───────────────────────────────────────────────
#define MAX_SLINE_LEN   (501)
#define PMF_THRESHOLD_P ((REAL)0.05)

#define THR_X_BL (64)
#define THR_X_SL (32)
#define BLOCK_Y  (THR_X_BL / THR_X_SL)  // = 2
#define MAX_N32DIMT 512

#define MAX_SLINES_PER_SEED (10)

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))
#define POW2(n)  (1 << (n))

#define DIV_UP(a,b) (((a)+((b)-1))/(b))

// simd_ballot returns simd_vote; extract bits via ulong then truncate to uint
#define SIMD_BALLOT_MASK(pred) uint(ulong(simd_ballot(pred)))

#define EXCESS_ALLOC_FACT 2

#define NORM_EPS ((REAL)1e-8)

// ── model types ──────────────────────────────────────────────────────
enum ModelType {
    OPDT = 0,
    CSA  = 1,
    PROB = 2,
    PTT  = 3,
};

enum { OUTSIDEIMAGE, INVALIDPOINT, TRACKPOINT, ENDPOINT };

#endif
