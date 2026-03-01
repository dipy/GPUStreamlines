/* Metal port of cuslines/cuda_c/boot.cu — bootstrap streamline generation.
 *
 * Translation notes:
 *   - CUDA __device__ functions → plain inline functions
 *   - CUDA __global__ kernels  → kernel functions
 *   - CUDA templates removed; concrete float types used throughout
 *   - __shared__ → threadgroup
 *   - Warp intrinsics → SIMD group intrinsics (Apple GPU SIMD width == 32)
 *   - curandStatePhilox4_32_10_t → PhiloxState (from philox_rng.h)
 *   - REAL_T → float, REAL3_T → float3 (packed_float3 for device buffers)
 *   - All #ifdef DEBUG / #if 0 blocks removed
 *   - USE_FIXED_PERMUTATION block removed
 */

#include "globals.h"
#include "types.h"
#include "philox_rng.h"

// ── params struct for kernel arguments ──────────────────────────────

struct BootTrackingParams {
    float max_angle;
    float tc_threshold;
    float step_size;
    float relative_peak_thresh;
    float min_separation_angle;
    float min_signal;
    int rng_seed_lo;
    int rng_seed_hi;
    int rng_offset;
    int nseed;
    int dimx, dimy, dimz, dimt;
    int samplm_nr;
    int num_edges;
    int delta_nr;
    int model_type;
};

// ── raw uint from Philox (equivalent to CUDA curand(&st)) ──────────

inline uint philox_uint(thread PhiloxState& s) {
    if (s.idx >= 4) {
        philox_next(s);
    }
    uint bits;
    switch (s.idx) {
        case 0: bits = s.output.x; break;
        case 1: bits = s.output.y; break;
        case 2: bits = s.output.z; break;
        default: bits = s.output.w; break;
    }
    s.idx++;
    return bits;
}

// ── avgMask — SIMD-parallel masked average ──────────────────────────

inline float avgMask(const int mskLen,
                     const device int* mask,
                     const threadgroup float* data,
                     uint tidx) {

    int   myCnt = 0;
    float mySum = 0.0f;

    for (int i = int(tidx); i < mskLen; i += THR_X_SL) {
        if (mask[i]) {
            myCnt++;
            mySum += data[i];
        }
    }

    for (int i = THR_X_SL / 2; i > 0; i /= 2) {
        mySum += simd_shuffle_xor(mySum, ushort(i));
        myCnt += simd_shuffle_xor(myCnt, ushort(i));
    }

    return mySum / float(myCnt);
}

// ── maskGet — compact non-masked entries ────────────────────────────

inline int maskGet(const int n,
                   const device int* mask,
                   const threadgroup float* plain,
                   threadgroup float* masked,
                   uint tidx) {

    const uint laneMask = (1u << tidx) - 1u;

    int woff = 0;
    for (int j = 0; j < n; j += THR_X_SL) {

        const int act = (j + int(tidx) < n) ? (!mask[j + int(tidx)]) : 0;
        const uint msk = SIMD_BALLOT_MASK(bool(act));

        const int toff = popcount(msk & laneMask);
        if (act) {
            masked[woff + toff] = plain[j + int(tidx)];
        }
        woff += popcount(msk);
    }
    return woff;
}

// ── maskPut — scatter masked entries back ───────────────────────────

inline void maskPut(const int n,
                    const device int* mask,
                    const threadgroup float* masked,
                    threadgroup float* plain,
                    uint tidx) {

    const uint laneMask = (1u << tidx) - 1u;

    int woff = 0;
    for (int j = 0; j < n; j += THR_X_SL) {

        const int act = (j + int(tidx) < n) ? (!mask[j + int(tidx)]) : 0;
        const uint msk = SIMD_BALLOT_MASK(bool(act));

        const int toff = popcount(msk & laneMask);
        if (act) {
            plain[j + int(tidx)] = masked[woff + toff];
        }
        woff += popcount(msk);
    }
}

// ── closest_peak_d — find closest peak to current direction ─────────

inline int closest_peak_d(const float max_angle,
                           const float3 direction,
                           const int npeaks,
                           const threadgroup float3* peaks,
                           threadgroup float3* peak,
                           uint tidx) {

    const float cos_similarity = COS(max_angle);

    float cpeak_dot = 0.0f;
    int   cpeak_idx = -1;
    for (int j = 0; j < npeaks; j += THR_X_SL) {
        if (j + int(tidx) < npeaks) {
            const float dot = direction.x * peaks[j + int(tidx)].x +
                              direction.y * peaks[j + int(tidx)].y +
                              direction.z * peaks[j + int(tidx)].z;

            if (FABS(dot) > FABS(cpeak_dot)) {
                cpeak_dot = dot;
                cpeak_idx = j + int(tidx);
            }
        }
    }

    for (int j = THR_X_SL / 2; j > 0; j /= 2) {
        const float dot = simd_shuffle_xor(cpeak_dot, ushort(j));
        const int   idx = simd_shuffle_xor(cpeak_idx, ushort(j));
        if (FABS(dot) > FABS(cpeak_dot)) {
            cpeak_dot = dot;
            cpeak_idx = idx;
        }
    }

    if (cpeak_idx >= 0) {
        if (cpeak_dot >= cos_similarity) {
            peak[0] = peaks[cpeak_idx];
            return 1;
        }
        if (cpeak_dot <= -cos_similarity) {
            peak[0] = float3(-peaks[cpeak_idx].x,
                             -peaks[cpeak_idx].y,
                             -peaks[cpeak_idx].z);
            return 1;
        }
    }
    return 0;
}

// ── ndotp_d — matrix-vector dot product ─────────────────────────────

inline void ndotp_d(const int N,
                    const int M,
                    const threadgroup float* srcV,
                    const device float* srcM,
                    threadgroup float* dstV,
                    uint tidx) {

    for (int i = 0; i < N; i++) {

        float tmp = 0.0f;

        for (int j = 0; j < M; j += THR_X_SL) {
            if (j + int(tidx) < M) {
                tmp += srcV[j + int(tidx)] * srcM[i * M + j + int(tidx)];
            }
        }
        for (int j = THR_X_SL / 2; j > 0; j /= 2) {
            tmp += simd_shuffle_down(tmp, ushort(j));
        }

        if (tidx == 0) {
            dstV[i] = tmp;
        }
    }
}

// ── ndotp_log_opdt_d — OPDT log-weighted dot product ────────────────

inline void ndotp_log_opdt_d(const int N,
                              const int M,
                              const threadgroup float* srcV,
                              const device float* srcM,
                              threadgroup float* dstV,
                              uint tidx) {

    const float ONEP5 = 1.5f;

    for (int i = 0; i < N; i++) {

        float tmp = 0.0f;

        for (int j = 0; j < M; j += THR_X_SL) {
            if (j + int(tidx) < M) {
                const float v = srcV[j + int(tidx)];
                tmp += -LOG(v) * (ONEP5 + LOG(v)) * v * srcM[i * M + j + int(tidx)];
            }
        }
        for (int j = THR_X_SL / 2; j > 0; j /= 2) {
            tmp += simd_shuffle_down(tmp, ushort(j));
        }

        if (tidx == 0) {
            dstV[i] = tmp;
        }
    }
}

// ── ndotp_log_csa_d — CSA log-log-weighted dot product ──────────────

inline void ndotp_log_csa_d(const int N,
                             const int M,
                             const threadgroup float* srcV,
                             const device float* srcM,
                             threadgroup float* dstV,
                             uint tidx) {

    const float csa_min = 0.001f;
    const float csa_max = 0.999f;

    for (int i = 0; i < N; i++) {

        float tmp = 0.0f;

        for (int j = 0; j < M; j += THR_X_SL) {
            if (j + int(tidx) < M) {
                const float v = MIN(MAX(srcV[j + int(tidx)], csa_min), csa_max);
                tmp += LOG(-LOG(v)) * srcM[i * M + j + int(tidx)];
            }
        }
        for (int j = THR_X_SL / 2; j > 0; j /= 2) {
            tmp += simd_shuffle_down(tmp, ushort(j));
        }

        if (tidx == 0) {
            dstV[i] = tmp;
        }
    }
}

// ── fit_opdt — OPDT model fitting ───────────────────────────────────

inline void fit_opdt(const int delta_nr,
                     const int hr_side,
                     const device float* delta_q,
                     const device float* delta_b,
                     const threadgroup float* msk_data_sh,
                     threadgroup float* h_sh,
                     threadgroup float* r_sh,
                     uint tidx) {

    ndotp_log_opdt_d(delta_nr, hr_side, msk_data_sh, delta_q, r_sh, tidx);
    ndotp_d(delta_nr, hr_side, msk_data_sh, delta_b, h_sh, tidx);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = int(tidx); j < delta_nr; j += THR_X_SL) {
        r_sh[j] -= h_sh[j];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// ── fit_csa — CSA model fitting ─────────────────────────────────────

inline void fit_csa(const int delta_nr,
                    const int hr_side,
                    const device float* fit_matrix,
                    const threadgroup float* msk_data_sh,
                    threadgroup float* r_sh,
                    uint tidx) {

    const float n0_const = 0.28209479177387814f;
    ndotp_log_csa_d(delta_nr, hr_side, msk_data_sh, fit_matrix, r_sh, tidx);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (tidx == 0) {
        r_sh[0] = n0_const;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// ── fit_model_coef — dispatch to OPDT or CSA ────────────────────────

inline void fit_model_coef(const int model_type,
                            const int delta_nr,
                            const int hr_side,
                            const device float* delta_q,
                            const device float* delta_b,
                            const threadgroup float* msk_data_sh,
                            threadgroup float* h_sh,
                            threadgroup float* r_sh,
                            uint tidx) {
    switch (model_type) {
        case OPDT:
            fit_opdt(delta_nr, hr_side, delta_q, delta_b, msk_data_sh, h_sh, r_sh, tidx);
            break;
        case CSA:
            fit_csa(delta_nr, hr_side, delta_q, msk_data_sh, r_sh, tidx);
            break;
        default:
            break;
    }
}

// ── get_direction_boot_d — bootstrap direction getter ───────────────

inline int get_direction_boot_d(
        thread PhiloxState& st,
        const int nattempts,
        const int model_type,
        const float max_angle,
        const float min_signal,
        const float relative_peak_thres,
        const float min_separation_angle,
        float3 dir,
        const int dimx,
        const int dimy,
        const int dimz,
        const int dimt,
        const device float* dataf,
        const device int* b0s_mask,
        const float3 point,
        const device float* H,
        const device float* R,
        const int delta_nr,
        const device float* delta_b,
        const device float* delta_q,
        const int samplm_nr,
        const device float* sampling_matrix,
        const device packed_float3* sphere_vertices,
        const device int2* sphere_edges,
        const int num_edges,
        threadgroup float3* dirs,
        threadgroup float* sh_mem,
        threadgroup float3* scratch_f3,
        uint tidx,
        uint tidy) {

    const int n32dimt = ((dimt + 31) / 32) * 32;

    // Partition shared memory — mirrors the CUDA layout
    threadgroup float* vox_data_sh = sh_mem;
    threadgroup float* msk_data_sh = vox_data_sh + n32dimt;

    threadgroup float* r_sh = msk_data_sh + n32dimt;
    threadgroup float* h_sh = r_sh + MAX(n32dimt, samplm_nr);

    // Compute hr_side (number of non-b0 volumes)
    int hr_side = 0;
    for (int j = int(tidx); j < dimt; j += THR_X_SL) {
        hr_side += (!b0s_mask[j]) ? 1 : 0;
    }
    for (int i = THR_X_SL / 2; i > 0; i /= 2) {
        hr_side += simd_shuffle_xor(hr_side, ushort(i));
    }

    for (int attempt = 0; attempt < nattempts; attempt++) {

        const int rv = trilinear_interp(dimx, dimy, dimz, dimt, -1,
                                        dataf, point, vox_data_sh, tidx);

        maskGet(dimt, b0s_mask, vox_data_sh, msk_data_sh, tidx);

        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (rv == 0) {

            // Multiply masked data by R and H matrices
            ndotp_d(hr_side, hr_side, msk_data_sh, R, r_sh, tidx);
            ndotp_d(hr_side, hr_side, msk_data_sh, H, h_sh, tidx);

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Bootstrap: add permuted residuals
            for (int j = 0; j < hr_side; j += THR_X_SL) {
                if (j + int(tidx) < hr_side) {
                    const int srcPermInd = int(philox_uint(st) % uint(hr_side));
                    h_sh[j + int(tidx)] += r_sh[srcPermInd];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // vox_data[dwi_mask] = masked_data
            maskPut(dimt, b0s_mask, h_sh, vox_data_sh, tidx);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (int j = int(tidx); j < dimt; j += THR_X_SL) {
                vox_data_sh[j] = MAX(min_signal, vox_data_sh[j]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            const float denom = avgMask(dimt, b0s_mask, vox_data_sh, tidx);

            for (int j = int(tidx); j < dimt; j += THR_X_SL) {
                vox_data_sh[j] /= denom;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            maskGet(dimt, b0s_mask, vox_data_sh, msk_data_sh, tidx);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            fit_model_coef(model_type, delta_nr, hr_side,
                           delta_q, delta_b, msk_data_sh, h_sh, r_sh, tidx);

            // r_sh <- coef; compute pmf = sampling_matrix * coef
            ndotp_d(samplm_nr, delta_nr, r_sh, sampling_matrix, h_sh, tidx);

            // h_sh <- pmf
        } else {
            for (int j = int(tidx); j < samplm_nr; j += THR_X_SL) {
                h_sh[j] = 0.0f;
            }
            // h_sh <- pmf (all zeros)
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Optional soft angular weighting: boost PMF values near the
        // current trajectory direction BEFORE thresholding.  At fiber
        // crossings (e.g. corona radiata), the commissural peak may be
        // weaker than the dominant projection peak.  Without weighting,
        // the aligned peak can fall below the 5% absolute or 25%
        // relative threshold and be zeroed out.  By weighting first,
        // the aligned peak is preserved and the perpendicular peak is
        // suppressed.
        // Controlled by ANGULAR_WEIGHT (0.0 = disabled, default).
        // Typical value: 0.5 → weight = 0.5 + 0.5*|cos(angle)|.
#if ENABLE_ANGULAR_WEIGHT
        if (nattempts > 1) {
            for (int j = int(tidx); j < samplm_nr; j += THR_X_SL) {
                const float3 sv = load_f3(sphere_vertices, uint(j));
                const float cos_sim = FABS(dir.x * sv.x +
                                           dir.y * sv.y +
                                           dir.z * sv.z);
                h_sh[j] *= ((1.0f - ANGULAR_WEIGHT) + ANGULAR_WEIGHT * cos_sim);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
#endif

        const float abs_pmf_thr = PMF_THRESHOLD_P *
            simd_max_reduce(samplm_nr, h_sh, REAL_MIN, tidx);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (int j = int(tidx); j < samplm_nr; j += THR_X_SL) {
            const float v = h_sh[j];
            if (v < abs_pmf_thr) {
                h_sh[j] = 0.0f;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const int ndir = peak_directions(h_sh, dirs,
                                         sphere_vertices,
                                         sphere_edges,
                                         num_edges,
                                         samplm_nr,
                                         reinterpret_cast<threadgroup int*>(r_sh),
                                         relative_peak_thres,
                                         min_separation_angle,
                                         tidx);
        if (nattempts == 1) { // init=True
            return ndir;
        } else { // init=False
            if (ndir > 0) {
                const int foundPeak = closest_peak_d(max_angle, dir, ndir, dirs, scratch_f3, tidx);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if (foundPeak) {
                    if (tidx == 0) {
                        dirs[0] = *scratch_f3;
                    }
                    return 1;
                }
            }
        }
    }
    return 0;
}

// ── tracker_boot_d — single-direction streamline tracker ────────────

inline int tracker_boot_d(
        thread PhiloxState& st,
        const int model_type,
        const float max_angle,
        const float tc_threshold,
        const float step_size,
        const float relative_peak_thres,
        const float min_separation_angle,
        float3 seed,
        float3 first_step,
        float3 voxel_size,
        const int dimx,
        const int dimy,
        const int dimz,
        const int dimt,
        const device float* dataf,
        const device float* metric_map,
        const int samplm_nr,
        const device packed_float3* sphere_vertices,
        const device int2* sphere_edges,
        const int num_edges,
        const float min_signal,
        const int delta_nr,
        const device float* H,
        const device float* R,
        const device float* delta_b,
        const device float* delta_q,
        const device float* sampling_matrix,
        const device int* b0s_mask,
        threadgroup int* nsteps,
        device packed_float3* streamline,
        threadgroup float* sh_mem,
        threadgroup float3* sh_dirs,
        threadgroup float* sh_interp,
        threadgroup float3* scratch_f3,
        uint tidx,
        uint tidy) {

    int tissue_class = TRACKPOINT;

    float3 point = seed;
    float3 direction = first_step;

    if (tidx == 0) {
        store_f3(streamline, 0, point);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const int step_frac = 1;

    int i;
    for (i = 1; i < MAX_SLINE_LEN * step_frac; i++) {
        int ndir = get_direction_boot_d(
                st,
                5,  // NATTEMPTS
                model_type,
                max_angle,
                min_signal,
                relative_peak_thres,
                min_separation_angle,
                direction,
                dimx, dimy, dimz, dimt, dataf,
                b0s_mask,
                point,
                H, R,
                delta_nr,
                delta_b, delta_q,
                samplm_nr,
                sampling_matrix,
                sphere_vertices,
                sphere_edges,
                num_edges,
                sh_dirs,
                sh_mem,
                scratch_f3,
                tidx, tidy);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        direction = *scratch_f3;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (ndir == 0) {
            break;
        }

        point.x += (direction.x / voxel_size.x) * (step_size / float(step_frac));
        point.y += (direction.y / voxel_size.y) * (step_size / float(step_frac));
        point.z += (direction.z / voxel_size.z) * (step_size / float(step_frac));

        if ((tidx == 0) && ((i % step_frac) == 0)) {
            store_f3(streamline, uint(i / step_frac), point);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        tissue_class = check_point(tc_threshold, point,
                                   dimx, dimy, dimz,
                                   metric_map,
                                   sh_interp,
                                   tidx, tidy);

        if (tissue_class == ENDPOINT ||
            tissue_class == INVALIDPOINT ||
            tissue_class == OUTSIDEIMAGE) {
            break;
        }
    }
    nsteps[0] = i / step_frac;
    if (((i % step_frac) != 0) && i < step_frac * (MAX_SLINE_LEN - 1)) {
        nsteps[0]++;
        if (tidx == 0) {
            store_f3(streamline, uint(nsteps[0]), point);
        }
    }

    return tissue_class;
}

// ── getNumStreamlinesBoot_k — count streamlines per seed (kernel) ───

kernel void getNumStreamlinesBoot_k(
        constant BootTrackingParams& params      [[buffer(0)]],
        const device packed_float3* seeds        [[buffer(1)]],
        const device float* dataf                [[buffer(2)]],
        const device float* H                    [[buffer(3)]],
        const device float* R                    [[buffer(4)]],
        const device float* delta_b              [[buffer(5)]],
        const device float* delta_q              [[buffer(6)]],
        const device int*   b0s_mask             [[buffer(7)]],
        const device float* sampling_matrix      [[buffer(8)]],
        const device packed_float3* sphere_vertices [[buffer(9)]],
        const device int2*  sphere_edges         [[buffer(10)]],
        device packed_float3* shDir0             [[buffer(11)]],
        device int*         slineOutOff          [[buffer(12)]],
        threadgroup float*  sh_pool              [[threadgroup(0)]],
        uint3 tgpig   [[threadgroup_position_in_grid]],
        uint3 tptg    [[threads_per_threadgroup]],
        uint3 tid_in_tg [[thread_position_in_threadgroup]],
        uint simd_lane [[thread_index_in_simdgroup]]) {

    const uint tidx = tid_in_tg.x;
    const uint tidy = tid_in_tg.y;
    const uint BDIM_Y = tptg.y;

    const int slid = int(tgpig.x) * int(BDIM_Y) + int(tidy);
    const uint gid = tgpig.x * tptg.y * tptg.x + tptg.x * tidy + tidx;

    if (slid >= params.nseed) {
        return;
    }

    float3 seed = load_f3(seeds, uint(slid));

    PhiloxState st = philox_init(uint(params.rng_seed_lo), uint(params.rng_seed_hi), gid, 0);

    // Shared memory layout:
    // Per-thread-row shared memory for get_direction_boot_d
    const int n32dimt = ((params.dimt + 31) / 32) * 32;
    const int sh_per_row = 2 * n32dimt + 2 * MAX(n32dimt, params.samplm_nr);

    // sh_pool is dynamically sized via setThreadgroupMemoryLength (CUDA extern __shared__ equivalent)
    threadgroup float3 sh_dirs[BLOCK_Y * MAX_SLINES_PER_SEED]; // per-tidy dirs
    threadgroup float3 scratch_f3[BLOCK_Y]; // per-tidy scratch for closest_peak_d
    threadgroup float* sh_mem = sh_pool + tidy * sh_per_row;

    int ndir;
    switch (params.model_type) {
        case OPDT:
        case CSA:
            ndir = get_direction_boot_d(
                    st,
                    1,  // NATTEMPTS=1 (init=True)
                    params.model_type,
                    params.max_angle,
                    params.min_signal,
                    params.relative_peak_thresh,
                    params.min_separation_angle,
                    float3(0.0f, 0.0f, 0.0f),
                    params.dimx, params.dimy, params.dimz, params.dimt,
                    dataf, b0s_mask,
                    seed,
                    H, R,
                    params.delta_nr,
                    delta_b, delta_q,
                    params.samplm_nr,
                    sampling_matrix,
                    sphere_vertices,
                    sphere_edges,
                    params.num_edges,
                    sh_dirs + tidy * MAX_SLINES_PER_SEED,
                    sh_mem,
                    scratch_f3 + tidy,
                    tidx, tidy);
            break;
        default:
            ndir = 0;
            break;
    }

    // Copy directions to output buffer
    device packed_float3* dirOut = shDir0 + slid * params.samplm_nr;
    for (int j = int(tidx); j < ndir; j += THR_X_SL) {
        store_f3(dirOut, uint(j), sh_dirs[tidy * MAX_SLINES_PER_SEED + j]);
    }

    if (tidx == 0) {
        slineOutOff[slid] = ndir;
    }
}

// ── genStreamlinesMergeBoot_k — main bootstrap streamline kernel ────

kernel void genStreamlinesMergeBoot_k(
        constant BootTrackingParams& params      [[buffer(0)]],
        const device packed_float3* seeds        [[buffer(1)]],
        const device float* dataf                [[buffer(2)]],
        const device float* metric_map           [[buffer(3)]],
        const device packed_float3* sphere_vertices [[buffer(4)]],
        const device int2*  sphere_edges         [[buffer(5)]],
        const device float* H                    [[buffer(6)]],
        const device float* R                    [[buffer(7)]],
        const device float* delta_b              [[buffer(8)]],
        const device float* delta_q              [[buffer(9)]],
        const device float* sampling_matrix      [[buffer(10)]],
        const device int*   b0s_mask             [[buffer(11)]],
        const device int*   slineOutOff          [[buffer(12)]],
        device packed_float3* shDir0             [[buffer(13)]],
        device int*         slineSeed            [[buffer(14)]],
        device int*         slineLen             [[buffer(15)]],
        device packed_float3* sline              [[buffer(16)]],
        threadgroup float*  sh_pool              [[threadgroup(0)]],
        uint3 tgpig   [[threadgroup_position_in_grid]],
        uint3 tptg    [[threads_per_threadgroup]],
        uint3 tid_in_tg [[thread_position_in_threadgroup]],
        uint simd_lane [[thread_index_in_simdgroup]]) {

    const uint tidx = tid_in_tg.x;
    const uint tidy = tid_in_tg.y;
    const uint BDIM_Y = tptg.y;

    const int slid = int(tgpig.x) * int(BDIM_Y) + int(tidy);

    const uint gid = tgpig.x * tptg.y * tptg.x + tptg.x * tidy + tidx;
    PhiloxState st = philox_init(uint(params.rng_seed_lo), uint(params.rng_seed_hi), gid + 1, 0);

    if (slid >= params.nseed) {
        return;
    }

    float3 seed = load_f3(seeds, uint(slid));

    int ndir = slineOutOff[slid + 1] - slineOutOff[slid];

    simdgroup_barrier(mem_flags::mem_threadgroup);

    int slineOff = slineOutOff[slid];

    // Shared memory layout for this thread row
    const int n32dimt = ((params.dimt + 31) / 32) * 32;
    const int sh_per_row = 2 * n32dimt + 2 * MAX(n32dimt, params.samplm_nr);

    // sh_pool is dynamically sized via setThreadgroupMemoryLength (CUDA extern __shared__ equivalent)
    threadgroup float3 sh_dirs[BLOCK_Y * MAX_SLINES_PER_SEED]; // per-tidy dirs
    threadgroup float sh_interp[BLOCK_Y]; // for check_point (indexed by tidy)
    threadgroup int sh_nsteps[BLOCK_Y]; // per-tidy step counts
    threadgroup float3 scratch_f3[BLOCK_Y]; // per-tidy scratch for closest_peak_d
    threadgroup float* sh_mem = sh_pool + tidy * sh_per_row;

    for (int i = 0; i < ndir; i++) {
        float3 first_step = load_f3(shDir0, uint(slid * params.samplm_nr + i));

        device packed_float3* currSline = sline + slineOff * MAX_SLINE_LEN * 2;

        if (tidx == 0) {
            slineSeed[slineOff] = slid;
        }

        // Track backward
        int stepsB;
        tracker_boot_d(
                st,
                params.model_type,
                params.max_angle,
                params.tc_threshold,
                params.step_size,
                params.relative_peak_thresh,
                params.min_separation_angle,
                seed,
                float3(-first_step.x, -first_step.y, -first_step.z),
                float3(1.0f, 1.0f, 1.0f),
                params.dimx, params.dimy, params.dimz, params.dimt,
                dataf,
                metric_map,
                params.samplm_nr,
                sphere_vertices,
                sphere_edges,
                params.num_edges,
                params.min_signal,
                params.delta_nr,
                H, R,
                delta_b, delta_q,
                sampling_matrix,
                b0s_mask,
                sh_nsteps + tidy,
                currSline,
                sh_mem,
                sh_dirs + tidy * MAX_SLINES_PER_SEED,
                sh_interp,
                scratch_f3 + tidy,
                tidx, tidy);
        stepsB = sh_nsteps[tidy];

        // Reverse backward streamline
        for (int j = 0; j < stepsB / 2; j += THR_X_SL) {
            if (j + int(tidx) < stepsB / 2) {
                const float3 p = load_f3(currSline, uint(j + int(tidx)));
                const float3 q = load_f3(currSline, uint(stepsB - 1 - (j + int(tidx))));
                store_f3(currSline, uint(j + int(tidx)), q);
                store_f3(currSline, uint(stepsB - 1 - (j + int(tidx))), p);
            }
        }

        // Track forward
        int stepsF;
        tracker_boot_d(
                st,
                params.model_type,
                params.max_angle,
                params.tc_threshold,
                params.step_size,
                params.relative_peak_thresh,
                params.min_separation_angle,
                seed,
                first_step,
                float3(1.0f, 1.0f, 1.0f),
                params.dimx, params.dimy, params.dimz, params.dimt,
                dataf,
                metric_map,
                params.samplm_nr,
                sphere_vertices,
                sphere_edges,
                params.num_edges,
                params.min_signal,
                params.delta_nr,
                H, R,
                delta_b, delta_q,
                sampling_matrix,
                b0s_mask,
                sh_nsteps + tidy,
                currSline + stepsB - 1,
                sh_mem,
                sh_dirs + tidy * MAX_SLINES_PER_SEED,
                sh_interp,
                scratch_f3 + tidy,
                tidx, tidy);
        stepsF = sh_nsteps[tidy];

        if (tidx == 0) {
            slineLen[slineOff] = stepsB - 1 + stepsF;
        }

        slineOff += 1;
    }
}
