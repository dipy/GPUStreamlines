/* Metal port of cuslines/cuda_c/utils.cu — reduction and prefix-sum primitives.
 *
 * CUDA warp operations → Metal SIMD group operations:
 *   __shfl_xor_sync(WMASK, v, delta, BDIM_X) → simd_shuffle_xor(v, delta)
 *   __shfl_up_sync(WMASK, v, delta, BDIM_X)  → simd_shuffle_up(v, delta)
 *   __syncwarp(WMASK)                         → simdgroup_barrier(mem_flags::mem_threadgroup)
 *
 * Since BDIM_X == THR_X_SL == 32 == Apple GPU SIMD width, the custom
 * WMASK always covers the full SIMD group so no masking is needed.
 */

#include "globals.h"

// ── max reduction across SIMD group ──────────────────────────────────

inline float simd_max_reduce(int n, const threadgroup float* src, float minVal,
                             uint tidx) {
    float m = minVal;
    for (int i = tidx; i < n; i += THR_X_SL) {
        m = MAX(m, src[i]);
    }
    for (int i = THR_X_SL / 2; i > 0; i /= 2) {
        float tmp = simd_shuffle_xor(m, ushort(i));
        m = MAX(m, tmp);
    }
    return m;
}

// ── min reduction across SIMD group ──────────────────────────────────

inline float simd_min_reduce(int n, const threadgroup float* src, float maxVal,
                             uint tidx) {
    float m = maxVal;
    for (int i = tidx; i < n; i += THR_X_SL) {
        m = MIN(m, src[i]);
    }
    for (int i = THR_X_SL / 2; i > 0; i /= 2) {
        float tmp = simd_shuffle_xor(m, ushort(i));
        m = MIN(m, tmp);
    }
    return m;
}

// ── max-with-mask reduction ──────────────────────────────────────────
// Only considers entries where srcMsk[i] > 0, applies offset to value.

inline float simd_max_mask_transl(int n,
                                  const threadgroup int* srcMsk,
                                  const threadgroup float* srcVal,
                                  float offset, float minVal,
                                  uint tidx) {
    float m = minVal;
    for (int i = tidx; i < n; i += THR_X_SL) {
        int sel = srcMsk[i];
        if (sel > 0) {
            m = MAX(m, srcVal[i] + offset);
        }
    }
    for (int i = THR_X_SL / 2; i > 0; i /= 2) {
        float tmp = simd_shuffle_xor(m, ushort(i));
        m = MAX(m, tmp);
    }
    return m;
}

// ── max from device buffer ───────────────────────────────────────────

inline float simd_max_reduce_dev(int n, const device float* src, float minVal,
                                 uint tidx) {
    float m = minVal;
    for (int i = tidx; i < n; i += THR_X_SL) {
        m = MAX(m, src[i]);
    }
    for (int i = THR_X_SL / 2; i > 0; i /= 2) {
        float tmp = simd_shuffle_xor(m, ushort(i));
        m = MAX(m, tmp);
    }
    return m;
}

// ── inclusive prefix sum in threadgroup memory ────────────────────────
// Operates on threadgroup float array of length __len.
// All threads in the SIMD group participate.

inline void prefix_sum_sh(threadgroup float* num_sh, int len, uint tidx) {
    for (int j = 0; j < len; j += THR_X_SL) {
        if ((tidx == 0) && (j != 0)) {
            num_sh[j] += num_sh[j - 1];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float t_pmf = 0.0f;
        if (j + int(tidx) < len) {
            t_pmf = num_sh[j + tidx];
        }
        for (int i = 1; i < THR_X_SL; i *= 2) {
            float tmp = simd_shuffle_up(t_pmf, ushort(i));
            if ((int(tidx) >= i) && (j + int(tidx) < len)) {
                t_pmf += tmp;
            }
        }
        if (j + int(tidx) < len) {
            num_sh[j + tidx] = t_pmf;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
