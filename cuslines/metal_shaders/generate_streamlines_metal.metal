/* Metal port of cuslines/cuda_c/generate_streamlines_cuda.cu
 *
 * Main streamline generation kernels for probabilistic and PTT tracking.
 * Bootstrap kernels are in boot.metal.
 */

#include "globals.h"
#include "types.h"
#include "philox_rng.h"

// Forward declarations from tracking_helpers.metal and utils.metal
inline int trilinear_interp(const int dimx, const int dimy, const int dimz,
                            const int dimt, int dimt_idx,
                            const device float* dataf,
                            const float3 point,
                            threadgroup float* vox_data,
                            uint tidx);

inline int check_point(const float tc_threshold,
                       const float3 point,
                       const int dimx, const int dimy, const int dimz,
                       const device float* metric_map,
                       threadgroup float* interp_out,
                       uint tidx, uint tidy);

inline int peak_directions(const threadgroup float* odf,
                           threadgroup float3* dirs,
                           const device packed_float3* sphere_vertices,
                           const device int2* sphere_edges,
                           const int num_edges,
                           int samplm_nr,
                           threadgroup int* shInd,
                           const float relative_peak_thres,
                           const float min_separation_angle,
                           uint tidx);

inline float simd_max_reduce(int n, const threadgroup float* src, float minVal, uint tidx);

inline void prefix_sum_sh(threadgroup float* num_sh, int len, uint tidx);

// ── Parameter struct for Prob/PTT kernels ────────────────────────────
// Guarded: may already be defined by ptt.metal (compiled first).

#ifndef PROB_TRACKING_PARAMS_DEFINED
#define PROB_TRACKING_PARAMS_DEFINED
struct ProbTrackingParams {
    float max_angle;
    float tc_threshold;
    float step_size;
    float relative_peak_thresh;
    float min_separation_angle;
    int   rng_seed_lo;
    int   rng_seed_hi;
    int   rng_offset;
    int   nseed;
    int   dimx;
    int   dimy;
    int   dimz;
    int   dimt;
    int   samplm_nr;
    int   num_edges;
    int   model_type;  // PROB=2 or PTT=3
};
#endif

// ── max threadgroup memory dimensions ────────────────────────────────
// BLOCK_Y and MAX_N32DIMT are defined in globals.h

// ── probabilistic direction getter ───────────────────────────────────

inline int get_direction_prob(thread PhiloxState& st,
                              const device float* pmf,
                              const float max_angle,
                              const float relative_peak_thres,
                              const float min_separation_angle,
                              float3 dir,
                              const int dimx, const int dimy,
                              const int dimz, const int dimt,
                              const float3 point,
                              const device packed_float3* sphere_vertices,
                              const device int2* sphere_edges,
                              const int num_edges,
                              threadgroup float3* out_dirs,
                              threadgroup float* sh_mem,
                              threadgroup int* sh_ind,
                              bool is_start,
                              uint tidx, uint tidy) {

    const int n32dimt = ((dimt + 31) / 32) * 32;
    threadgroup float* pmf_data_sh = sh_mem + tidy * n32dimt;

    // pmf = trilinear interpolation at point
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const int rv = trilinear_interp(dimx, dimy, dimz, dimt, -1, pmf, point, pmf_data_sh, tidx);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (rv != 0) {
        return 0;
    }

    // absolute pmf threshold
    const float absolpmf_thresh = PMF_THRESHOLD_P * simd_max_reduce(dimt, pmf_data_sh, REAL_MIN, tidx);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // zero out entries below threshold
    for (int i = int(tidx); i < dimt; i += THR_X_SL) {
        if (pmf_data_sh[i] < absolpmf_thresh) {
            pmf_data_sh[i] = 0.0f;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (is_start) {
        return peak_directions(pmf_data_sh,
                               out_dirs,
                               sphere_vertices,
                               sphere_edges,
                               num_edges,
                               dimt,
                               sh_ind,
                               relative_peak_thres,
                               min_separation_angle,
                               tidx);
    } else {
        // Filter by angle similarity
        const float cos_similarity = COS(max_angle);

        for (int i = int(tidx); i < dimt; i += THR_X_SL) {
            float3 sv = load_f3(sphere_vertices, uint(i));
            const float dot = dir.x * sv.x + dir.y * sv.y + dir.z * sv.z;
            if (FABS(dot) < cos_similarity) {
                pmf_data_sh[i] = 0.0f;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Prefix sum for CDF
        prefix_sum_sh(pmf_data_sh, dimt, tidx);

        float last_cdf = pmf_data_sh[dimt - 1];
        if (last_cdf == 0.0f) {
            return 0;
        }

        // Sample from CDF
        float tmp;
        if (tidx == 0) {
            tmp = philox_uniform(st) * last_cdf;
        }
        float selected_cdf = simd_broadcast_first(tmp);

        // Binary search + ballot for insertion point
        int low = 0;
        int high = dimt - 1;
        while ((high - low) >= THR_X_SL) {
            const int mid = (low + high) / 2;
            if (pmf_data_sh[mid] < selected_cdf) {
                low = mid;
            } else {
                high = mid;
            }
        }
        const bool ballot_pred = (low + int(tidx) <= high) ? (selected_cdf < pmf_data_sh[low + tidx]) : false;
        const uint msk = SIMD_BALLOT_MASK(ballot_pred);
        const int indProb = (msk != 0) ? (low + int(ctz(msk))) : (dimt - 1);

        // Select direction, flip if needed
        if (tidx == 0) {
            float3 sv = load_f3(sphere_vertices, uint(indProb));
            if ((dir.x * sv.x + dir.y * sv.y + dir.z * sv.z) > 0) {
                *out_dirs = sv;
            } else {
                *out_dirs = -sv;
            }
        }

        return 1;
    }
}

// ── tracker — step along streamline ──────────────────────────────────

inline int tracker_prob(thread PhiloxState& st,
                        const float max_angle,
                        const float tc_threshold,
                        const float step_size,
                        const float relative_peak_thres,
                        const float min_separation_angle,
                        float3 seed,
                        float3 first_step,
                        const float3 voxel_size,
                        const int dimx, const int dimy,
                        const int dimz, const int dimt,
                        const device float* dataf,
                        const device float* metric_map,
                        const int samplm_nr,
                        const device packed_float3* sphere_vertices,
                        const device int2* sphere_edges,
                        const int num_edges,
                        threadgroup int* nsteps,
                        device packed_float3* streamline,
                        threadgroup float3* sh_new_dir,
                        threadgroup float* sh_mem,
                        threadgroup float* interp_out,
                        threadgroup int* sh_ind,
                        uint tidx, uint tidy) {

    int tissue_class = TRACKPOINT;
    float3 point = seed;
    float3 direction = first_step;

    if (tidx == 0) {
        store_f3(streamline, 0, point);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    int i;
    for (i = 1; i < MAX_SLINE_LEN; i++) {
        int ndir = get_direction_prob(st, dataf, max_angle,
                                      relative_peak_thres, min_separation_angle,
                                      direction, dimx, dimy, dimz, dimt,
                                      point, sphere_vertices, sphere_edges,
                                      num_edges, sh_new_dir + tidy,
                                      sh_mem, sh_ind, false, tidx, tidy);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        direction = sh_new_dir[tidy];
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (ndir == 0) {
            break;
        }

        point.x += (direction.x / voxel_size.x) * step_size;
        point.y += (direction.y / voxel_size.y) * step_size;
        point.z += (direction.z / voxel_size.z) * step_size;

        if (tidx == 0) {
            store_f3(streamline, uint(i), point);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        tissue_class = check_point(tc_threshold, point, dimx, dimy, dimz,
                                   metric_map, interp_out, tidx, tidy);

        if (tissue_class == ENDPOINT ||
            tissue_class == INVALIDPOINT ||
            tissue_class == OUTSIDEIMAGE) {
            break;
        }
    }
    nsteps[0] = i;
    return tissue_class;
}

// ── getNumStreamlinesProb_k ──────────────────────────────────────────

kernel void getNumStreamlinesProb_k(
    constant ProbTrackingParams& params [[buffer(0)]],
    const device packed_float3* seeds   [[buffer(1)]],
    const device float* dataf           [[buffer(2)]],
    const device packed_float3* sphere_vertices [[buffer(3)]],
    const device int2* sphere_edges     [[buffer(4)]],
    device packed_float3* shDir0        [[buffer(5)]],
    device int* slineOutOff             [[buffer(6)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint tidx = tid.x;
    const uint tidy = tid.y;
    const uint slid = gid.x * BLOCK_Y + tidy;

    if (int(slid) >= params.nseed) return;

    const uint global_id = gid.x * BLOCK_Y * THR_X_SL + THR_X_SL * tidy + tidx;
    PhiloxState st = philox_init(uint(params.rng_seed_lo), uint(params.rng_seed_hi), global_id, 0);

    const int n32dimt = ((params.dimt + 31) / 32) * 32;

    // Threadgroup memory
    threadgroup float sh_mem[BLOCK_Y * MAX_N32DIMT];
    threadgroup int sh_ind[BLOCK_Y * MAX_N32DIMT];
    threadgroup float3 dirs_sh[BLOCK_Y * MAX_SLINES_PER_SEED];

    threadgroup float* my_sh = sh_mem + tidy * n32dimt;
    threadgroup int* my_ind = sh_ind + tidy * n32dimt;

    float3 seed = load_f3(seeds, slid);
    device packed_float3* my_shDir = shDir0 + slid * params.dimt;

    int ndir = get_direction_prob(st, dataf, params.max_angle,
                                  params.relative_peak_thresh,
                                  params.min_separation_angle,
                                  float3(0, 0, 0),
                                  params.dimx, params.dimy, params.dimz, params.dimt,
                                  seed, sphere_vertices, sphere_edges,
                                  params.num_edges,
                                  dirs_sh + tidy * MAX_SLINES_PER_SEED,
                                  my_sh, my_ind, true, tidx, tidy);

    // Copy found directions to global memory
    if (tidx == 0) {
        for (int d = 0; d < ndir; d++) {
            store_f3(my_shDir, uint(d), dirs_sh[tidy * MAX_SLINES_PER_SEED + d]);
        }
        slineOutOff[slid] = ndir;
    }
}

// ── genStreamlinesMergeProb_k ────────────────────────────────────────

kernel void genStreamlinesMergeProb_k(
    constant ProbTrackingParams& params [[buffer(0)]],
    const device packed_float3* seeds   [[buffer(1)]],
    const device float* dataf           [[buffer(2)]],
    const device float* metric_map      [[buffer(3)]],
    const device packed_float3* sphere_vertices [[buffer(4)]],
    const device int2* sphere_edges     [[buffer(5)]],
    const device int* slineOutOff       [[buffer(6)]],
    device packed_float3* shDir0        [[buffer(7)]],
    device int* slineSeed               [[buffer(8)]],
    device int* slineLen                [[buffer(9)]],
    device packed_float3* sline         [[buffer(10)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint tidx = tid.x;
    const uint tidy = tid.y;
    const uint slid = gid.x * BLOCK_Y + tidy;

    if (int(slid) >= params.nseed) return;

    const uint global_id = gid.x * BLOCK_Y * THR_X_SL + THR_X_SL * tidy + tidx;
    PhiloxState st = philox_init(uint(params.rng_seed_lo), uint(params.rng_seed_hi), global_id + 1, 0);

    const int n32dimt = ((params.dimt + 31) / 32) * 32;

    // Threadgroup memory
    threadgroup float sh_mem[BLOCK_Y * MAX_N32DIMT];
    threadgroup int sh_ind[BLOCK_Y * MAX_N32DIMT];
    threadgroup float3 sh_new_dir[BLOCK_Y];
    threadgroup float interp_out[BLOCK_Y];
    threadgroup int stepsB_sh[BLOCK_Y];
    threadgroup int stepsF_sh[BLOCK_Y];

    float3 seed = load_f3(seeds, slid);

    int ndir = slineOutOff[slid + 1] - slineOutOff[slid];
    simdgroup_barrier(mem_flags::mem_threadgroup);

    int slineOff = slineOutOff[slid];

    for (int i = 0; i < ndir; i++) {
        float3 first_step = load_f3(shDir0, uint(int(slid) * params.samplm_nr + i));

        device packed_float3* currSline = sline + slineOff * MAX_SLINE_LEN * 2;

        if (tidx == 0) {
            slineSeed[slineOff] = int(slid);
        }

        // Backward tracking
        tracker_prob(st, params.max_angle, params.tc_threshold,
                     params.step_size, params.relative_peak_thresh,
                     params.min_separation_angle,
                     seed, float3(-first_step.x, -first_step.y, -first_step.z),
                     float3(1, 1, 1),
                     params.dimx, params.dimy, params.dimz, params.dimt,
                     dataf, metric_map, params.samplm_nr,
                     sphere_vertices, sphere_edges, params.num_edges,
                     stepsB_sh + tidy, currSline,
                     sh_new_dir, sh_mem, interp_out,
                     sh_ind + tidy * n32dimt, tidx, tidy);

        int stepsB = stepsB_sh[tidy];

        // Reverse backward streamline
        for (int j = int(tidx); j < stepsB / 2; j += THR_X_SL) {
            float3 p = load_f3(currSline, uint(j));
            store_f3(currSline, uint(j), load_f3(currSline, uint(stepsB - 1 - j)));
            store_f3(currSline, uint(stepsB - 1 - j), p);
        }

        // Forward tracking
        tracker_prob(st, params.max_angle, params.tc_threshold,
                     params.step_size, params.relative_peak_thresh,
                     params.min_separation_angle,
                     seed, first_step, float3(1, 1, 1),
                     params.dimx, params.dimy, params.dimz, params.dimt,
                     dataf, metric_map, params.samplm_nr,
                     sphere_vertices, sphere_edges, params.num_edges,
                     stepsF_sh + tidy, currSline + (stepsB - 1),
                     sh_new_dir, sh_mem, interp_out,
                     sh_ind + tidy * n32dimt, tidx, tidy);

        if (tidx == 0) {
            slineLen[slineOff] = stepsB - 1 + stepsF_sh[tidy];
        }

        slineOff += 1;
    }
}
