/* Metal port of cuslines/cuda_c/tracking_helpers.cu
 *
 * Trilinear interpolation, tissue checking, and peak direction finding.
 */

#include "globals.h"
#include "types.h"

// ── trilinear interpolation helper (inner loop) ──────────────────────

inline float interpolation_helper(const device float* dataf,
                                  const float wgh[3][2],
                                  const long coo[3][2],
                                  int dimy, int dimz, int dimt, int t) {
    float tmp = 0.0f;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                tmp += wgh[0][i] * wgh[1][j] * wgh[2][k] *
                       dataf[coo[0][i] * dimy * dimz * dimt +
                             coo[1][j] * dimz * dimt +
                             coo[2][k] * dimt +
                             t];
            }
        }
    }
    return tmp;
}

// ── trilinear interpolation ──────────────────────────────────────────
// All threads in the SIMD group compute boundary checks together.
// Thread-parallel loop over the dimt dimension.

inline int trilinear_interp(const int dimx, const int dimy, const int dimz,
                            const int dimt, int dimt_idx,
                            const device float* dataf,
                            const float3 point,
                            threadgroup float* vox_data,
                            uint tidx) {
    const float HALF = 0.5f;

    if (point.x < -HALF || point.x + HALF >= float(dimx) ||
        point.y < -HALF || point.y + HALF >= float(dimy) ||
        point.z < -HALF || point.z + HALF >= float(dimz)) {
        return -1;
    }

    long coo[3][2];  // 64-bit to avoid overflow in index computation (CUDA uses long long)
    float wgh[3][2];

    const float3 fl = floor(point);

    wgh[0][1] = point.x - fl.x;
    wgh[0][0] = 1.0f - wgh[0][1];
    coo[0][0] = MAX(0, int(fl.x));
    coo[0][1] = MIN(int(dimx - 1), coo[0][0] + 1);

    wgh[1][1] = point.y - fl.y;
    wgh[1][0] = 1.0f - wgh[1][1];
    coo[1][0] = MAX(0, int(fl.y));
    coo[1][1] = MIN(int(dimy - 1), coo[1][0] + 1);

    wgh[2][1] = point.z - fl.z;
    wgh[2][0] = 1.0f - wgh[2][1];
    coo[2][0] = MAX(0, int(fl.z));
    coo[2][1] = MIN(int(dimz - 1), coo[2][0] + 1);

    if (dimt_idx == -1) {
        for (int t = int(tidx); t < dimt; t += THR_X_SL) {
            vox_data[t] = interpolation_helper(dataf, wgh, coo, dimy, dimz, dimt, t);
        }
    } else {
        *vox_data = interpolation_helper(dataf, wgh, coo, dimy, dimz, dimt, dimt_idx);
    }
    return 0;
}

// ── tissue check at a point ──────────────────────────────────────────

inline int check_point(const float tc_threshold,
                       const float3 point,
                       const int dimx, const int dimy, const int dimz,
                       const device float* metric_map,
                       threadgroup float* interp_out,  // length BLOCK_Y
                       uint tidx, uint tidy) {

    const int rv = trilinear_interp(dimx, dimy, dimz, 1, 0,
                                    metric_map, point,
                                    interp_out + tidy, tidx);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (rv != 0) {
        return OUTSIDEIMAGE;
    }
    return (interp_out[tidy] > tc_threshold) ? TRACKPOINT : ENDPOINT;
}

// ── peak direction finding ───────────────────────────────────────────
// Finds local maxima on the ODF sphere, filters by relative threshold
// and minimum separation angle.

inline int peak_directions(const threadgroup float* odf,
                           threadgroup float3* dirs,
                           const device packed_float3* sphere_vertices,
                           const device int2* sphere_edges,
                           const int num_edges,
                           int samplm_nr,
                           threadgroup int* shInd,
                           const float relative_peak_thres,
                           const float min_separation_angle,
                           uint tidx) {
    // Initialize index array
    for (int j = int(tidx); j < samplm_nr; j += THR_X_SL) {
        shInd[j] = 0;
    }

    float odf_min = simd_min_reduce(samplm_nr, odf, REAL_MAX, tidx);
    odf_min = MAX(0.0f, odf_min);

    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Local maxima detection using sphere edges
    // atomics on threadgroup memory for benign race conditions
    for (int j = 0; j < num_edges; j += THR_X_SL) {
        if (j + int(tidx) < num_edges) {
            const int u_ind = sphere_edges[j + tidx].x;
            const int v_ind = sphere_edges[j + tidx].y;

            const float u_val = odf[u_ind];
            const float v_val = odf[v_ind];

            if (u_val < v_val) {
                atomic_store_explicit(
                    (volatile threadgroup atomic_int*)(shInd + u_ind), -1,
                    memory_order_relaxed);
                atomic_fetch_or_explicit(
                    (volatile threadgroup atomic_int*)(shInd + v_ind), 1,
                    memory_order_relaxed);
            } else if (v_val < u_val) {
                atomic_store_explicit(
                    (volatile threadgroup atomic_int*)(shInd + v_ind), -1,
                    memory_order_relaxed);
                atomic_fetch_or_explicit(
                    (volatile threadgroup atomic_int*)(shInd + u_ind), 1,
                    memory_order_relaxed);
            }
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const float compThres = relative_peak_thres *
        simd_max_mask_transl(samplm_nr, shInd, odf, -odf_min, REAL_MIN, tidx);

    // Compact indices of positive values (local maxima above threshold)
    int n = 0;
    const uint lmask = (1u << tidx) - 1u;  // lanes below me

    for (int j = 0; j < samplm_nr; j += THR_X_SL) {
        const int v = (j + int(tidx) < samplm_nr) ? shInd[j + tidx] : -1;
        const bool keep = (v > 0) && ((odf[j + tidx] - odf_min) >= compThres);

        // simd_ballot returns a simd_vote on Metal; we can extract the uint mask
        uint msk = SIMD_BALLOT_MASK(keep);

        if (keep) {
            const int myoff = popcount(msk & lmask);
            shInd[n + myoff] = j + int(tidx);
        }
        n += popcount(msk);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Sort local maxima by ODF value (descending)
    if (n > 0 && n < THR_X_SL) {
        float k = REAL_MIN;
        int val = 0;
        if (int(tidx) < n) {
            val = shInd[tidx];
            k = odf[val];
        }
        warp_sort_kv<THR_X_SL, WSORT_DIR_DEC>(k, val, tidx);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (int(tidx) < n) {
            shInd[tidx] = val;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Remove similar vertices (single-threaded)
    if (n != 0) {
        if (tidx == 0) {
            const float cos_similarity = COS(min_separation_angle);

            dirs[0] = load_f3(sphere_vertices, uint(shInd[0]));

            int k = 1;
            for (int i = 1; i < n; i++) {
                const float3 abc = load_f3(sphere_vertices, uint(shInd[i]));

                int j = 0;
                for (; j < k; j++) {
                    const float cs = FABS(abc.x * dirs[j].x +
                                          abc.y * dirs[j].y +
                                          abc.z * dirs[j].z);
                    if (cs > cos_similarity) {
                        break;
                    }
                }
                if (j == k) {
                    dirs[k++] = abc;
                }
            }
            n = k;
        }
        n = simd_broadcast_first(n);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    return n;
}
