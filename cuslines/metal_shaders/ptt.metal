/* Metal port of cuslines/cuda_c/ptt.cu — Parallel Transport Tractography.
 *
 * Aydogan DB, Shi Y. Parallel Transport Tractography. IEEE Trans Med Imaging.
 * 2021 Feb;40(2):635-647. doi: 10.1109/TMI.2020.3034038.
 *
 * Translation rules applied:
 *   __device__                         -> inline functions
 *   threadIdx.x / threadIdx.y          -> tidx / tidy parameters
 *   __syncwarp(WMASK)                  -> simdgroup_barrier(mem_flags::mem_threadgroup)
 *   __shfl_xor_sync(WMASK, v, d, BDX) -> simd_shuffle_xor(v, ushort(d))
 *   __shfl_sync(WMASK, v, l, BDX)     -> simd_shuffle(v, ushort(l))
 *   curandStatePhilox4_32_10_t         -> PhiloxState
 *   curand_init / uniform / normal     -> philox_init / philox_uniform / philox_normal
 *   __shared__                         -> threadgroup (at kernel scope only)
 *   REAL_T                             -> float
 *   REAL3_T                            -> float3 (registers) / packed_float3 (device)
 *   MAKE_REAL3(x,y,z)                 -> float3(x,y,z)
 *   Templates removed — concrete float types throughout.
 */

#include "globals.h"
#include "types.h"
#include "philox_rng.h"

// ── disc data ────────────────────────────────────────────────────────
// Include the raw disc vertex/face macros,
// then declare Metal constant-address-space arrays for SAMPLING_QUALITY == 2.

#include "disc.h"

// ── PTT constants (from ptt.cuh) ─────────────────────────────────────
#define STEP_FRAC       (20)
#define PROBE_FRAC      (2)
#define PROBE_QUALITY   (4)
#define SAMPLING_QUALITY (2)
#define ALLOW_WEAK_LINK (0)
#define TRIES_PER_REJECTION_SAMPLING (1024)
#define K_SMALL         (0.0001f)

#define DISC_VERT_CNT DISC_2_VERT_CNT
#define DISC_FACE_CNT DISC_2_FACE_CNT

constant float DISC_VERT[DISC_VERT_CNT * 2] = DISC_2_VERT;
constant int   DISC_FACE[DISC_FACE_CNT * 3] = DISC_2_FACE;

// ── forward declarations of helpers defined in other .metal files ────
// (These are compiled together into a single Metal library.)

inline float simd_max_reduce_dev(int n, const device float* src, float minVal,
                                 uint tidx);

inline void prefix_sum_sh(threadgroup float* num_sh, int len, uint tidx);

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

// ── norm3 ────────────────────────────────────────────────────────────
// Normalise a 3-vector in place.  On degenerate input set axis fail_ind to 1.

inline void norm3(thread float* num, int fail_ind) {
    const float scale = SQRT(num[0] * num[0] + num[1] * num[1] + num[2] * num[2]);

    if (scale > NORM_EPS) {
        num[0] /= scale;
        num[1] /= scale;
        num[2] /= scale;
    } else {
        num[0] = num[1] = num[2] = 0;
        num[fail_ind] = 1.0f;
    }
}

// threadgroup overload
inline void norm3(threadgroup float* num, int fail_ind) {
    const float scale = SQRT(num[0] * num[0] + num[1] * num[1] + num[2] * num[2]);

    if (scale > NORM_EPS) {
        num[0] /= scale;
        num[1] /= scale;
        num[2] /= scale;
    } else {
        num[0] = num[1] = num[2] = 0;
        num[fail_ind] = 1.0f;
    }
}

// ── crossnorm3 ──────────────────────────────────────────────────────
// dest = normalise(src1 x src2)

inline void crossnorm3(threadgroup float* dest,
                        const threadgroup float* src1,
                        const threadgroup float* src2,
                        int fail_ind) {
    dest[0] = src1[1] * src2[2] - src1[2] * src2[1];
    dest[1] = src1[2] * src2[0] - src1[0] * src2[2];
    dest[2] = src1[0] * src2[1] - src1[1] * src2[0];

    norm3(dest, fail_ind);
}

// ── interp4 ─────────────────────────────────────────────────────────
// Find the ODF sphere vertex closest to `frame` direction, then
// trilinearly interpolate the PMF at that vertex index.

inline float interp4(const float3 pos,
                      const threadgroup float* frame,
                      const device float* pmf,
                      const int dimx, const int dimy,
                      const int dimz, const int dimt,
                      const device packed_float3* odf_sphere_vertices,
                      threadgroup float* interp_scratch,
                      uint tidx) {

    int closest_odf_idx = 0;
    float max_cos = 0.0f;

    for (int ii = int(tidx); ii < dimt; ii += THR_X_SL) {
        float3 sv = load_f3(odf_sphere_vertices, uint(ii));
        float cos_sim = FABS(sv.x * frame[0] +
                             sv.y * frame[1] +
                             sv.z * frame[2]);
        if (cos_sim > max_cos) {
            max_cos = cos_sim;
            closest_odf_idx = ii;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across the SIMD group
    for (int i = THR_X_SL / 2; i > 0; i /= 2) {
        const float tmp = simd_shuffle_xor(max_cos, ushort(i));
        const int tmp_idx = simd_shuffle_xor(closest_odf_idx, ushort(i));
        if (tmp > max_cos ||
           (tmp == max_cos && tmp_idx < closest_odf_idx)) {
            max_cos = tmp;
            closest_odf_idx = tmp_idx;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Trilinear interpolation at the closest ODF vertex
    const int rv = trilinear_interp(dimx, dimy, dimz, dimt,
                                    closest_odf_idx, pmf, pos,
                                    interp_scratch, tidx);

    if (rv != 0) {
        return 0.0f;  // No support
    } else {
        return *interp_scratch;
    }
}

// ── prepare_propagator ──────────────────────────────────────────────
// Build 3x3 propagator matrix from curvatures k1, k2 and arclength.

inline void prepare_propagator(float k1, float k2, float arclength,
                                threadgroup float* propagator) {
    if ((FABS(k1) < K_SMALL) && (FABS(k2) < K_SMALL)) {
        propagator[0] = arclength;
        propagator[1] = 0;
        propagator[2] = 0;
        propagator[3] = 1;
        propagator[4] = 0;
        propagator[5] = 0;
        propagator[6] = 0;
        propagator[7] = 0;
        propagator[8] = 1;
    } else {
        if (FABS(k1) < K_SMALL) {
            k1 = K_SMALL;
        }
        if (FABS(k2) < K_SMALL) {
            k2 = K_SMALL;
        }
        const float k     = SQRT(k1 * k1 + k2 * k2);
        const float sinkt = SIN(k * arclength);
        const float coskt = COS(k * arclength);
        const float kk    = 1.0f / (k * k);

        propagator[0] = sinkt / k;
        propagator[1] = k1 * (1.0f - coskt) * kk;
        propagator[2] = k2 * (1.0f - coskt) * kk;
        propagator[3] = coskt;
        propagator[4] = k1 * sinkt / k;
        propagator[5] = k2 * sinkt / k;
        propagator[6] = -propagator[5];
        propagator[7] = k1 * k2 * (coskt - 1.0f) * kk;
        propagator[8] = (k1 * k1 + k2 * k2 * coskt) * kk;
    }
}

// ── random_normal_ptt ───────────────────────────────────────────────
// Generate a random normal vector perpendicular to probing_frame[0..2].

inline void random_normal_ptt(thread PhiloxState& st,
                               threadgroup float* probing_frame) {
    probing_frame[3] = philox_normal(st);
    probing_frame[4] = philox_normal(st);
    probing_frame[5] = philox_normal(st);

    float dot = probing_frame[3] * probing_frame[0]
              + probing_frame[4] * probing_frame[1]
              + probing_frame[5] * probing_frame[2];

    probing_frame[3] -= dot * probing_frame[0];
    probing_frame[4] -= dot * probing_frame[1];
    probing_frame[5] -= dot * probing_frame[2];

    float n2 = probing_frame[3] * probing_frame[3]
             + probing_frame[4] * probing_frame[4]
             + probing_frame[5] * probing_frame[5];

    if (n2 < NORM_EPS) {
        float abs_x = FABS(probing_frame[0]);
        float abs_y = FABS(probing_frame[1]);
        float abs_z = FABS(probing_frame[2]);

        if (abs_x <= abs_y && abs_x <= abs_z) {
            probing_frame[3] = 0.0f;
            probing_frame[4] = probing_frame[2];
            probing_frame[5] = -probing_frame[1];
        }
        else if (abs_y <= abs_z) {
            probing_frame[3] = -probing_frame[2];
            probing_frame[4] = 0.0f;
            probing_frame[5] = probing_frame[0];
        }
        else {
            probing_frame[3] = probing_frame[1];
            probing_frame[4] = -probing_frame[0];
            probing_frame[5] = 0.0f;
        }
    }
}

// ── get_probing_frame ───────────────────────────────────────────────
// IS_INIT variant: build a fresh probing frame from the tangent direction.
// Non-init variant: just copy the existing frame.

inline void get_probing_frame_init(const threadgroup float* frame,
                                    thread PhiloxState& st,
                                    threadgroup float* probing_frame) {
    for (int ii = 0; ii < 3; ii++) {
        probing_frame[ii] = frame[ii];
    }
    norm3(probing_frame, 0);

    random_normal_ptt(st, probing_frame);
    norm3(probing_frame + 3, 1);

    // binorm = tangent x normal
    crossnorm3(probing_frame + 2 * 3, probing_frame, probing_frame + 3, 2);
}

inline void get_probing_frame_noinit(const threadgroup float* frame,
                                      threadgroup float* probing_frame) {
    for (int ii = 0; ii < 9; ii++) {
        probing_frame[ii] = frame[ii];
    }
}

// ── propagate_frame ─────────────────────────────────────────────────
// Apply propagator matrix to the frame, re-orthonormalise, and output direction.

inline void propagate_frame(threadgroup float* propagator,
                             threadgroup float* frame,
                             threadgroup float* direc) {
    float tmp[3];

    for (int ii = 0; ii < 3; ii++) {
        direc[ii]       = propagator[0] * frame[ii] + propagator[1] * frame[3 + ii] + propagator[2] * frame[6 + ii];
        tmp[ii]         = propagator[3] * frame[ii] + propagator[4] * frame[3 + ii] + propagator[5] * frame[6 + ii];
        frame[2*3 + ii] = propagator[6] * frame[ii] + propagator[7] * frame[3 + ii] + propagator[8] * frame[6 + ii];
    }

    norm3(tmp, 0);  // normalise tangent

    // Write normalised tangent back to frame[0..2] so crossnorm3 can
    // operate on threadgroup pointers (Metal requires address-space-qualified args).
    for (int ii = 0; ii < 3; ii++) {
        frame[ii] = tmp[ii];
    }

    crossnorm3(frame + 3, frame + 2 * 3, frame, 1);      // normal = cross(binorm, tangent)
    crossnorm3(frame + 2 * 3, frame, frame + 3, 2);       // binorm = cross(tangent, normal)
}

// ── calculate_data_support ──────────────────────────────────────────
// Probe forward along a candidate curve and accumulate FOD amplitudes.

inline float calculate_data_support(
    float support,
    const float3 pos,
    const device float* pmf,
    const int dimx, const int dimy, const int dimz, const int dimt,
    const float probe_step_size,
    const float absolpmf_thresh,
    const device packed_float3* odf_sphere_vertices,
    threadgroup float* probing_prop_sh,
    threadgroup float* direc_sh,
    threadgroup float3* probing_pos_sh,
    threadgroup float* k1_sh,
    threadgroup float* k2_sh,
    threadgroup float* probing_frame_sh,
    threadgroup float* interp_scratch,
    uint tidx) {

    if (tidx == 0) {
        prepare_propagator(*k1_sh, *k2_sh, probe_step_size, probing_prop_sh);
        *probing_pos_sh = pos;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (int ii = 0; ii < PROBE_QUALITY; ii++) {
        if (tidx == 0) {
            propagate_frame(probing_prop_sh, probing_frame_sh, direc_sh);

            float3 pp = *probing_pos_sh;
            pp.x += direc_sh[0];
            pp.y += direc_sh[1];
            pp.z += direc_sh[2];
            *probing_pos_sh = pp;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float fod_amp = interp4(
            *probing_pos_sh, probing_frame_sh, pmf,
            dimx, dimy, dimz, dimt,
            odf_sphere_vertices, interp_scratch, tidx);

        if (!ALLOW_WEAK_LINK && (fod_amp < absolpmf_thresh)) {
            return 0.0f;
        }
        support += fod_amp;
    }
    return support;
}

// ── get_direction_ptt (IS_INIT == true) ─────────────────────────────
// Workspace threadgroup arrays are declared at kernel scope and passed
// as pre-offset (by tidy) pointers.

inline int get_direction_ptt_init(
    thread PhiloxState& st,
    const device float* pmf,
    const float max_angle,
    const float step_size,
    float3 dir,
    threadgroup float* frame_sh,
    const int dimx, const int dimy, const int dimz, const int dimt,
    float3 pos,
    const device packed_float3* odf_sphere_vertices,
    threadgroup packed_float3* dirs,
    // PTT workspace (pre-offset by tidy from kernel scope)
    threadgroup float* my_face_cdf_sh,
    threadgroup float* my_vert_pdf_sh,
    threadgroup float* my_probing_frame_sh,
    threadgroup float* my_k1_probe_sh,
    threadgroup float* my_k2_probe_sh,
    threadgroup float* my_probing_prop_sh,
    threadgroup float* my_direc_sh,
    threadgroup float3* my_probing_pos_sh,
    threadgroup float* my_interp_scratch,
    uint tidx) {

    const float probe_step_size = ((step_size / PROBE_FRAC) / (PROBE_QUALITY - 1));
    const float max_curvature = 2.0f * SIN(max_angle / 2.0f) / (step_size / PROBE_FRAC);
    const float absolpmf_thresh = PMF_THRESHOLD_P * simd_max_reduce_dev(dimt, pmf, REAL_MIN, tidx);

    simdgroup_barrier(mem_flags::mem_threadgroup);

    // IS_INIT: set frame tangent from dir
    if (tidx == 0) {
        frame_sh[0] = dir.x;
        frame_sh[1] = dir.y;
        frame_sh[2] = dir.z;
    }

    const float first_val = interp4(
        pos, frame_sh, pmf,
        dimx, dimy, dimz, dimt,
        odf_sphere_vertices, my_interp_scratch, tidx);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Calculate vert_pdf_sh
    bool support_found = false;
    for (int ii = 0; ii < DISC_VERT_CNT; ii++) {
        if (tidx == 0) {
            *my_k1_probe_sh = DISC_VERT[ii * 2] * max_curvature;
            *my_k2_probe_sh = DISC_VERT[ii * 2 + 1] * max_curvature;
            get_probing_frame_init(frame_sh, st, my_probing_frame_sh);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float this_support = calculate_data_support(
            first_val,
            pos, pmf, dimx, dimy, dimz, dimt,
            probe_step_size,
            absolpmf_thresh,
            odf_sphere_vertices,
            my_probing_prop_sh, my_direc_sh, my_probing_pos_sh,
            my_k1_probe_sh, my_k2_probe_sh,
            my_probing_frame_sh, my_interp_scratch, tidx);

        if (this_support < PROBE_QUALITY * absolpmf_thresh) {
            if (tidx == 0) {
                my_vert_pdf_sh[ii] = 0;
            }
        } else {
            if (tidx == 0) {
                my_vert_pdf_sh[ii] = this_support;
            }
            support_found = true;
        }
    }
    if (!support_found) {
        return 0;
    }

    // Initialise face_cdf_sh
    for (int ii = int(tidx); ii < DISC_FACE_CNT; ii += THR_X_SL) {
        my_face_cdf_sh[ii] = 0;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Move vert PDF to face PDF
    for (int ii = int(tidx); ii < DISC_FACE_CNT; ii += THR_X_SL) {
        bool all_verts_valid = true;
        for (int jj = 0; jj < 3; jj++) {
            float vert_val = my_vert_pdf_sh[DISC_FACE[ii * 3 + jj]];
            if (vert_val == 0) {
                all_verts_valid = true; // IS_INIT: even go with faces that are not fully supported
            }
            my_face_cdf_sh[ii] += vert_val;
        }
        if (!all_verts_valid) {
            my_face_cdf_sh[ii] = 0;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Prefix sum and check for zero total
    prefix_sum_sh(my_face_cdf_sh, DISC_FACE_CNT, tidx);
    float last_cdf = my_face_cdf_sh[DISC_FACE_CNT - 1];

    if (last_cdf == 0) {
        return 0;
    }

    // Rejection sampling
    for (int ii = 0; ii < TRIES_PER_REJECTION_SAMPLING; ii++) {
        float tmp_sample;
        if (tidx == 0) {
            float r1 = philox_uniform(st);
            float r2 = philox_uniform(st);
            if (r1 + r2 > 1.0f) {
                r1 = 1.0f - r1;
                r2 = 1.0f - r2;
            }

            tmp_sample = philox_uniform(st) * last_cdf;
            int jj;
            for (jj = 0; jj < DISC_FACE_CNT; jj++) {
                if (my_face_cdf_sh[jj] >= tmp_sample)
                    break;
            }

            const float vx0 = max_curvature * DISC_VERT[DISC_FACE[jj * 3] * 2];
            const float vx1 = max_curvature * DISC_VERT[DISC_FACE[jj * 3 + 1] * 2];
            const float vx2 = max_curvature * DISC_VERT[DISC_FACE[jj * 3 + 2] * 2];

            const float vy0 = max_curvature * DISC_VERT[DISC_FACE[jj * 3] * 2 + 1];
            const float vy1 = max_curvature * DISC_VERT[DISC_FACE[jj * 3 + 1] * 2 + 1];
            const float vy2 = max_curvature * DISC_VERT[DISC_FACE[jj * 3 + 2] * 2 + 1];

            *my_k1_probe_sh = vx0 + r1 * (vx1 - vx0) + r2 * (vx2 - vx0);
            *my_k2_probe_sh = vy0 + r1 * (vy1 - vy0) + r2 * (vy2 - vy0);
            get_probing_frame_init(frame_sh, st, my_probing_frame_sh);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float this_support = calculate_data_support(
            first_val,
            pos, pmf, dimx, dimy, dimz, dimt,
            probe_step_size,
            absolpmf_thresh,
            odf_sphere_vertices,
            my_probing_prop_sh, my_direc_sh, my_probing_pos_sh,
            my_k1_probe_sh, my_k2_probe_sh,
            my_probing_frame_sh, my_interp_scratch, tidx);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (this_support < PROBE_QUALITY * absolpmf_thresh) {
            continue;
        }

        // IS_INIT: just store the original direction
        if (tidx == 0) {
            store_f3(dirs, 0, dir);
        }

        if (tidx < 9) {
            frame_sh[tidx] = my_probing_frame_sh[tidx];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        return 1;
    }
    return 0;
}

// ── get_direction_ptt (IS_INIT == false) ────────────────────────────
// Workspace threadgroup arrays are declared at kernel scope and passed
// as pre-offset (by tidy) pointers.

inline int get_direction_ptt_noinit(
    thread PhiloxState& st,
    const device float* pmf,
    const float max_angle,
    const float step_size,
    float3 dir,
    threadgroup float* frame_sh,
    const int dimx, const int dimy, const int dimz, const int dimt,
    float3 pos,
    const device packed_float3* odf_sphere_vertices,
    threadgroup packed_float3* dirs,
    // PTT workspace (pre-offset by tidy from kernel scope)
    threadgroup float* my_face_cdf_sh,
    threadgroup float* my_vert_pdf_sh,
    threadgroup float* my_probing_frame_sh,
    threadgroup float* my_k1_probe_sh,
    threadgroup float* my_k2_probe_sh,
    threadgroup float* my_probing_prop_sh,
    threadgroup float* my_direc_sh,
    threadgroup float3* my_probing_pos_sh,
    threadgroup float* my_interp_scratch,
    uint tidx) {

    const float probe_step_size = ((step_size / PROBE_FRAC) / (PROBE_QUALITY - 1));
    const float max_curvature = 2.0f * SIN(max_angle / 2.0f) / (step_size / PROBE_FRAC);
    const float absolpmf_thresh = PMF_THRESHOLD_P * simd_max_reduce_dev(dimt, pmf, REAL_MIN, tidx);

    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Non-init: frame_sh is already populated

    const float first_val = interp4(
        pos, frame_sh, pmf,
        dimx, dimy, dimz, dimt,
        odf_sphere_vertices, my_interp_scratch, tidx);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Calculate vert_pdf_sh
    bool support_found = false;
    for (int ii = 0; ii < DISC_VERT_CNT; ii++) {
        if (tidx == 0) {
            *my_k1_probe_sh = DISC_VERT[ii * 2] * max_curvature;
            *my_k2_probe_sh = DISC_VERT[ii * 2 + 1] * max_curvature;
            get_probing_frame_noinit(frame_sh, my_probing_frame_sh);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float this_support = calculate_data_support(
            first_val,
            pos, pmf, dimx, dimy, dimz, dimt,
            probe_step_size,
            absolpmf_thresh,
            odf_sphere_vertices,
            my_probing_prop_sh, my_direc_sh, my_probing_pos_sh,
            my_k1_probe_sh, my_k2_probe_sh,
            my_probing_frame_sh, my_interp_scratch, tidx);

        if (this_support < PROBE_QUALITY * absolpmf_thresh) {
            if (tidx == 0) {
                my_vert_pdf_sh[ii] = 0;
            }
        } else {
            if (tidx == 0) {
                my_vert_pdf_sh[ii] = this_support;
            }
            support_found = true;
        }
    }
    if (!support_found) {
        return 0;
    }

    // Initialise face_cdf_sh
    for (int ii = int(tidx); ii < DISC_FACE_CNT; ii += THR_X_SL) {
        my_face_cdf_sh[ii] = 0;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Move vert PDF to face PDF
    for (int ii = int(tidx); ii < DISC_FACE_CNT; ii += THR_X_SL) {
        bool all_verts_valid = true;
        for (int jj = 0; jj < 3; jj++) {
            float vert_val = my_vert_pdf_sh[DISC_FACE[ii * 3 + jj]];
            if (vert_val == 0) {
                all_verts_valid = false; // Non-init: reject faces with unsupported vertices
            }
            my_face_cdf_sh[ii] += vert_val;
        }
        if (!all_verts_valid) {
            my_face_cdf_sh[ii] = 0;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Prefix sum and check for zero total
    prefix_sum_sh(my_face_cdf_sh, DISC_FACE_CNT, tidx);
    float last_cdf = my_face_cdf_sh[DISC_FACE_CNT - 1];

    if (last_cdf == 0) {
        return 0;
    }

    // Rejection sampling
    for (int ii = 0; ii < TRIES_PER_REJECTION_SAMPLING; ii++) {
        float tmp_sample;
        if (tidx == 0) {
            float r1 = philox_uniform(st);
            float r2 = philox_uniform(st);
            if (r1 + r2 > 1.0f) {
                r1 = 1.0f - r1;
                r2 = 1.0f - r2;
            }

            tmp_sample = philox_uniform(st) * last_cdf;
            int jj;
            for (jj = 0; jj < DISC_FACE_CNT; jj++) {
                if (my_face_cdf_sh[jj] >= tmp_sample)
                    break;
            }

            const float vx0 = max_curvature * DISC_VERT[DISC_FACE[jj * 3] * 2];
            const float vx1 = max_curvature * DISC_VERT[DISC_FACE[jj * 3 + 1] * 2];
            const float vx2 = max_curvature * DISC_VERT[DISC_FACE[jj * 3 + 2] * 2];

            const float vy0 = max_curvature * DISC_VERT[DISC_FACE[jj * 3] * 2 + 1];
            const float vy1 = max_curvature * DISC_VERT[DISC_FACE[jj * 3 + 1] * 2 + 1];
            const float vy2 = max_curvature * DISC_VERT[DISC_FACE[jj * 3 + 2] * 2 + 1];

            *my_k1_probe_sh = vx0 + r1 * (vx1 - vx0) + r2 * (vx2 - vx0);
            *my_k2_probe_sh = vy0 + r1 * (vy1 - vy0) + r2 * (vy2 - vy0);
            get_probing_frame_noinit(frame_sh, my_probing_frame_sh);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float this_support = calculate_data_support(
            first_val,
            pos, pmf, dimx, dimy, dimz, dimt,
            probe_step_size,
            absolpmf_thresh,
            odf_sphere_vertices,
            my_probing_prop_sh, my_direc_sh, my_probing_pos_sh,
            my_k1_probe_sh, my_k2_probe_sh,
            my_probing_frame_sh, my_interp_scratch, tidx);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (this_support < PROBE_QUALITY * absolpmf_thresh) {
            continue;
        }

        // Non-init: propagate 1/STEP_FRAC of a step and output direction
        if (tidx == 0) {
            prepare_propagator(
                *my_k1_probe_sh, *my_k2_probe_sh,
                step_size / STEP_FRAC, my_probing_prop_sh);
            get_probing_frame_noinit(frame_sh, my_probing_frame_sh);
            propagate_frame(my_probing_prop_sh, my_probing_frame_sh, my_direc_sh);

            // norm3 on threadgroup memory
            norm3(my_direc_sh, 0);

            store_f3(dirs, 0, float3(my_direc_sh[0], my_direc_sh[1], my_direc_sh[2]));
        }

        if (tidx < 9) {
            frame_sh[tidx] = my_probing_frame_sh[tidx];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        return 1;
    }
    return 0;
}

// ── init_frame_ptt ──────────────────────────────────────────────────
// Initialise the parallel transport frame for a new streamline.
// Tries the negative direction first, then the positive, and flips if needed.

inline bool init_frame_ptt(
    thread PhiloxState& st,
    const device float* pmf,
    const float max_angle,
    const float step_size,
    float3 first_step,
    const int dimx, const int dimy, const int dimz, const int dimt,
    float3 seed,
    const device packed_float3* sphere_vertices,
    threadgroup float* frame,
    threadgroup packed_float3* tmp_dir,
    // PTT workspace (pre-offset by tidy from kernel scope)
    threadgroup float* my_face_cdf_sh,
    threadgroup float* my_vert_pdf_sh,
    threadgroup float* my_probing_frame_sh,
    threadgroup float* my_k1_probe_sh,
    threadgroup float* my_k2_probe_sh,
    threadgroup float* my_probing_prop_sh,
    threadgroup float* my_direc_sh,
    threadgroup float3* my_probing_pos_sh,
    threadgroup float* my_interp_scratch,
    uint tidx) {

    bool init_norm_success;

    // Try with negated direction first
    init_norm_success = (bool)get_direction_ptt_init(
        st,
        pmf,
        max_angle,
        step_size,
        float3(-first_step.x, -first_step.y, -first_step.z),
        frame,
        dimx, dimy, dimz, dimt,
        seed,
        sphere_vertices,
        tmp_dir,
        my_face_cdf_sh, my_vert_pdf_sh,
        my_probing_frame_sh,
        my_k1_probe_sh, my_k2_probe_sh,
        my_probing_prop_sh, my_direc_sh,
        my_probing_pos_sh, my_interp_scratch,
        tidx);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (!init_norm_success) {
        // Try the other direction
        init_norm_success = (bool)get_direction_ptt_init(
            st,
            pmf,
            max_angle,
            step_size,
            float3(first_step.x, first_step.y, first_step.z),
            frame,
            dimx, dimy, dimz, dimt,
            seed,
            sphere_vertices,
            tmp_dir,
            my_face_cdf_sh, my_vert_pdf_sh,
            my_probing_frame_sh,
            my_k1_probe_sh, my_k2_probe_sh,
            my_probing_prop_sh, my_direc_sh,
            my_probing_pos_sh, my_interp_scratch,
            tidx);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (!init_norm_success) {
            return false;
        } else {
            if (tidx == 0) {
                for (int ii = 0; ii < 9; ii++) {
                    frame[ii] = -frame[ii];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Save flipped frame for second run
    if (tidx == 0) {
        for (int ii = 0; ii < 9; ii++) {
            frame[9 + ii] = -frame[ii];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    return true;
}

// ── ProbTrackingParams struct ────────────────────────────────────────
// Shared with generate_streamlines_metal.metal.  Guard against
// duplicate definitions since both files are compiled into one library.

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

// ── tracker_ptt — step along streamline with parallel transport ─────
// Mirrors tracker_d<PTT> from CUDA: takes fractional steps (STEP_FRAC
// sub-steps per full step), only stores every STEP_FRAC'th point.

inline int tracker_ptt(thread PhiloxState& st,
                       const float max_angle,
                       const float tc_threshold,
                       const float step_size,
                       float3 seed,
                       float3 first_step,
                       const float3 voxel_size,
                       const int dimx, const int dimy,
                       const int dimz, const int dimt,
                       const device float* dataf,
                       const device float* metric_map,
                       const device packed_float3* sphere_vertices,
                       threadgroup int* nsteps,
                       device packed_float3* streamline,
                       threadgroup float* frame_sh,
                       threadgroup float* interp_out,
                       // PTT workspace (pre-offset by tidy)
                       threadgroup packed_float3* ptt_dirs,
                       threadgroup float* my_face_cdf_sh,
                       threadgroup float* my_vert_pdf_sh,
                       threadgroup float* my_probing_frame_sh,
                       threadgroup float* my_k1_probe_sh,
                       threadgroup float* my_k2_probe_sh,
                       threadgroup float* my_probing_prop_sh,
                       threadgroup float* my_direc_sh,
                       threadgroup float3* my_probing_pos_sh,
                       threadgroup float* my_interp_scratch,
                       uint tidx, uint tidy) {

    int tissue_class = TRACKPOINT;
    float3 point = seed;
    float3 direction = first_step;

    if (tidx == 0) {
        store_f3(streamline, 0, point);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    int i;
    for (i = 1; i < MAX_SLINE_LEN * STEP_FRAC; i++) {
        int ndir = get_direction_ptt_noinit(st, dataf, max_angle, step_size,
                                            direction, frame_sh,
                                            dimx, dimy, dimz, dimt,
                                            point, sphere_vertices,
                                            ptt_dirs,
                                            my_face_cdf_sh, my_vert_pdf_sh,
                                            my_probing_frame_sh,
                                            my_k1_probe_sh, my_k2_probe_sh,
                                            my_probing_prop_sh, my_direc_sh,
                                            my_probing_pos_sh, my_interp_scratch,
                                            tidx);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        direction = load_f3(ptt_dirs, 0);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (ndir == 0) {
            break;
        }

        point.x += (direction.x / voxel_size.x) * (step_size / float(STEP_FRAC));
        point.y += (direction.y / voxel_size.y) * (step_size / float(STEP_FRAC));
        point.z += (direction.z / voxel_size.z) * (step_size / float(STEP_FRAC));

        if ((tidx == 0) && ((i % STEP_FRAC) == 0)) {
            store_f3(streamline, uint(i / STEP_FRAC), point);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if ((i % STEP_FRAC) == 0) {
            tissue_class = check_point(tc_threshold, point, dimx, dimy, dimz,
                                       metric_map, interp_out, tidx, tidy);

            if (tissue_class == ENDPOINT ||
                tissue_class == INVALIDPOINT ||
                tissue_class == OUTSIDEIMAGE) {
                break;
            }
        }
    }

    nsteps[0] = i / STEP_FRAC;
    // If stopped mid-fraction, store the final point
    if (((i % STEP_FRAC) != 0) && (i < STEP_FRAC * (MAX_SLINE_LEN - 1))) {
        nsteps[0] += 1;
        if (tidx == 0) {
            store_f3(streamline, uint(nsteps[0]), point);
        }
    }
    return tissue_class;
}

// ── genStreamlinesMergePtt_k ─────────────────────────────────────────
// PTT generation kernel.  Uses the same buffer layout as the Prob kernel
// so the Python dispatch code is shared.  PTT reuses Prob's getNum kernel
// for initial direction finding.

kernel void genStreamlinesMergePtt_k(
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

    // ── PTT-specific threadgroup memory ─────────────────────────────
    threadgroup float frame_sh[BLOCK_Y * 18];          // 9 backward + 9 forward
    threadgroup packed_float3 tmp_dir_sh[BLOCK_Y];     // for init_frame_ptt
    threadgroup packed_float3 ptt_dirs_sh[BLOCK_Y];    // direction output
    threadgroup float interp_out[BLOCK_Y];
    threadgroup int stepsB_sh[BLOCK_Y];
    threadgroup int stepsF_sh[BLOCK_Y];

    // PTT workspace arrays
    threadgroup float face_cdf[BLOCK_Y * DISC_FACE_CNT];
    threadgroup float vert_pdf[BLOCK_Y * DISC_VERT_CNT];
    threadgroup float probing_frame[BLOCK_Y * 9];
    threadgroup float k1_probe[BLOCK_Y];
    threadgroup float k2_probe[BLOCK_Y];
    threadgroup float probing_prop[BLOCK_Y * 9];
    threadgroup float direc[BLOCK_Y * 3];
    threadgroup float3 probing_pos[BLOCK_Y];
    threadgroup float interp_scratch[BLOCK_Y];

    // Pre-offset pointers for this tidy
    threadgroup float* my_frame          = frame_sh + tidy * 18;
    threadgroup packed_float3* my_tmpdir = tmp_dir_sh + tidy;
    threadgroup packed_float3* my_dirs   = ptt_dirs_sh + tidy;

    threadgroup float*  my_face_cdf  = face_cdf + tidy * DISC_FACE_CNT;
    threadgroup float*  my_vert_pdf  = vert_pdf + tidy * DISC_VERT_CNT;
    threadgroup float*  my_pfr       = probing_frame + tidy * 9;
    threadgroup float*  my_k1        = k1_probe + tidy;
    threadgroup float*  my_k2        = k2_probe + tidy;
    threadgroup float*  my_pprop     = probing_prop + tidy * 9;
    threadgroup float*  my_direc     = direc + tidy * 3;
    threadgroup float3* my_ppos      = probing_pos + tidy;
    threadgroup float*  my_iscratch  = interp_scratch + tidy;

    // ── per-seed loop ───────────────────────────────────────────────
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

        // PTT frame initialization
        if (!init_frame_ptt(st, dataf, params.max_angle, params.step_size,
                            first_step,
                            params.dimx, params.dimy, params.dimz, params.dimt,
                            seed, sphere_vertices,
                            my_frame,
                            my_tmpdir,
                            my_face_cdf, my_vert_pdf,
                            my_pfr, my_k1, my_k2,
                            my_pprop, my_direc,
                            my_ppos, my_iscratch,
                            tidx)) {
            // Init failed — store single-point streamline
            if (tidx == 0) {
                slineLen[slineOff] = 1;
                store_f3(currSline, 0, seed);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            slineOff += 1;
            continue;
        }

        // Backward tracking (using frame[0:9])
        tracker_ptt(st, params.max_angle, params.tc_threshold,
                    params.step_size,
                    seed, float3(-first_step.x, -first_step.y, -first_step.z),
                    float3(1, 1, 1),
                    params.dimx, params.dimy, params.dimz, params.dimt,
                    dataf, metric_map, sphere_vertices,
                    stepsB_sh + tidy, currSline,
                    my_frame,       // backward frame = first 9 elements
                    interp_out,
                    my_dirs,
                    my_face_cdf, my_vert_pdf,
                    my_pfr, my_k1, my_k2,
                    my_pprop, my_direc,
                    my_ppos, my_iscratch,
                    tidx, tidy);

        int stepsB = stepsB_sh[tidy];

        // Reverse backward streamline
        for (int j = int(tidx); j < stepsB / 2; j += THR_X_SL) {
            float3 p = load_f3(currSline, uint(j));
            store_f3(currSline, uint(j), load_f3(currSline, uint(stepsB - 1 - j)));
            store_f3(currSline, uint(stepsB - 1 - j), p);
        }

        // Forward tracking (using frame[9:18])
        tracker_ptt(st, params.max_angle, params.tc_threshold,
                    params.step_size,
                    seed, first_step, float3(1, 1, 1),
                    params.dimx, params.dimy, params.dimz, params.dimt,
                    dataf, metric_map, sphere_vertices,
                    stepsF_sh + tidy, currSline + (stepsB - 1),
                    my_frame + 9,   // forward frame = last 9 elements
                    interp_out,
                    my_dirs,
                    my_face_cdf, my_vert_pdf,
                    my_pfr, my_k1, my_k2,
                    my_pprop, my_direc,
                    my_ppos, my_iscratch,
                    tidx, tidy);

        if (tidx == 0) {
            slineLen[slineOff] = stepsB - 1 + stepsF_sh[tidy];
        }

        slineOff += 1;
    }
}
