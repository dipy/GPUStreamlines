// ptt.wgsl — Parallel Transport Tractography direction getter and kernel.
// Translated from cuslines/metal_shaders/ptt.metal.
//
// Aydogan DB, Shi Y. Parallel Transport Tractography. IEEE Trans Med Imaging.
// 2021 Feb;40(2):635-647. doi: 10.1109/TMI.2020.3034038.
//
// Concatenation order (within the Prob shader module):
//   globals.wgsl, types.wgsl, philox_rng.wgsl, utils.wgsl, warp_sort.wgsl,
//   tracking_helpers.wgsl, disc.wgsl, **ptt.wgsl**, generate_streamlines.wgsl
//
// WGSL declarations are order-independent at module scope, so this file can
// reference buffer bindings, workgroup arrays, and helper functions declared
// in generate_streamlines.wgsl (concatenated after this file).
//
// Key WGSL adaptations vs Metal:
//   - PhiloxState is pass-by-value; every function returns modified state
//   - No threadgroup pointers as function params; use workgroup array offsets
//   - packed_float3 -> flat f32 arrays with 3-element stride
//   - simdgroup_barrier -> subgroupBarrier()
//   - simd_shuffle_xor -> subgroupShuffleXor
//   - Workgroup arrays declared at module scope with PTT-specific names

// ── PTT constants ──────────────────────────────────────────────────
const STEP_FRAC: i32 = 20;
const PROBE_FRAC: i32 = 2;
const PROBE_QUALITY: i32 = 4;
const ALLOW_WEAK_LINK: bool = false;
const TRIES_PER_REJECTION_SAMPLING: i32 = 1024;
const K_SMALL: f32 = 0.0001;

// ── PTT-specific workgroup memory ──────────────────────────────────
// These must not conflict with generate_streamlines.wgsl's wg_* arrays.
// Sizes account for BLOCK_Y=2 concurrent SIMD groups.

var<workgroup> ptt_frame_sh: array<f32, 36>;          // BLOCK_Y * 18
var<workgroup> ptt_dirs: array<f32, 6>;                 // BLOCK_Y * 3
var<workgroup> ptt_stepsB: array<i32, 2>;               // BLOCK_Y
var<workgroup> ptt_stepsF: array<i32, 2>;              // BLOCK_Y
var<workgroup> ptt_face_cdf: array<f32, 62>;           // BLOCK_Y * DISC_FACE_CNT
var<workgroup> ptt_vert_pdf: array<f32, 48>;           // BLOCK_Y * DISC_VERT_CNT
var<workgroup> ptt_probing_frame: array<f32, 18>;      // BLOCK_Y * 9
var<workgroup> ptt_k1_probe: array<f32, 2>;            // BLOCK_Y
var<workgroup> ptt_k2_probe: array<f32, 2>;            // BLOCK_Y
var<workgroup> ptt_probing_prop: array<f32, 18>;       // BLOCK_Y * 9
var<workgroup> ptt_direc: array<f32, 6>;               // BLOCK_Y * 3
var<workgroup> ptt_probing_pos: array<f32, 6>;         // BLOCK_Y * 3
var<workgroup> ptt_interp_scratch: array<f32, 2>;      // BLOCK_Y

// ── max reduction reading directly from dataf storage buffer ───────
// Finds max of dataf[0..n-1] across the subgroup (32 lanes).
// Used to compute the absolute PMF threshold in PTT.
fn sg_max_reduce_dataf(n: i32, min_val: f32, tidx: u32) -> f32 {
    var m = min_val;
    for (var i = i32(tidx); i < n; i += i32(THR_X_SL)) {
        m = max(m, dataf[u32(i)]);
    }
    m = max(m, subgroupShuffleXor(m, 16u));
    m = max(m, subgroupShuffleXor(m, 8u));
    m = max(m, subgroupShuffleXor(m, 4u));
    m = max(m, subgroupShuffleXor(m, 2u));
    m = max(m, subgroupShuffleXor(m, 1u));
    return m;
}

// ── single-channel trilinear interpolation from dataf ──────────────
// Interpolates a single dimt channel at a given point, writing result
// to ptt_interp_scratch[tidy]. Returns -1 if outside image, 0 otherwise.
// Only lane 0 gets the meaningful result; all lanes participate in setup.
fn trilinear_interp_dataf_single(
    dimx: i32, dimy: i32, dimz: i32, dimt: i32,
    dimt_idx: i32, point: vec3<f32>, tidy: u32
) -> i32 {
    let setup = trilinear_setup(dimx, dimy, dimz, point);
    if (setup.status != 0) { return -1; }

    ptt_interp_scratch[tidy] =
        interpolation_helper_dataf(setup.wgh, setup.coo, dimy, dimz, dimt, dimt_idx);
    return 0;
}

// ── prefix sum on ptt_face_cdf ─────────────────────────────────────
// Inclusive prefix sum operating on ptt_face_cdf[base_offset..base_offset+len].
fn prefix_sum_ptt_face_cdf(base_offset: u32, len: i32, tidx: u32) {
    for (var j = 0; j < len; j += i32(THR_X_SL)) {
        if (tidx == 0u && j != 0) {
            ptt_face_cdf[base_offset + u32(j)] += ptt_face_cdf[base_offset + u32(j - 1)];
        }
        subgroupBarrier();

        var t_pmf: f32 = 0.0;
        if (j + i32(tidx) < len) {
            t_pmf = ptt_face_cdf[base_offset + u32(j) + tidx];
        }
        for (var i = 1u; i < THR_X_SL; i *= 2u) {
            let tmp = subgroupShuffleUp(t_pmf, i);
            if (tidx >= i && j + i32(tidx) < len) {
                t_pmf += tmp;
            }
        }
        if (j + i32(tidx) < len) {
            ptt_face_cdf[base_offset + u32(j) + tidx] = t_pmf;
        }
        subgroupBarrier();
    }
}

// ── norm3 / crossnorm3 on workgroup arrays ─────────────────────────
// PTT uses multiple workgroup arrays. Since WGSL cannot pass workgroup
// pointers to functions, we provide array-specific norm3/crossnorm3
// variants for each workgroup array that needs them.

// norm3 on ptt_probing_frame
fn norm3_probing_frame(base: u32, fail_ind: i32) {
    let x = ptt_probing_frame[base];
    let y = ptt_probing_frame[base + 1u];
    let z = ptt_probing_frame[base + 2u];
    let scale = sqrt(x * x + y * y + z * z);

    if (scale > NORM_EPS) {
        ptt_probing_frame[base] = x / scale;
        ptt_probing_frame[base + 1u] = y / scale;
        ptt_probing_frame[base + 2u] = z / scale;
    } else {
        ptt_probing_frame[base] = 0.0;
        ptt_probing_frame[base + 1u] = 0.0;
        ptt_probing_frame[base + 2u] = 0.0;
        ptt_probing_frame[base + u32(fail_ind)] = 1.0;
    }
}

// Direct norm3 on ptt_frame_sh
fn norm3_frame(base: u32, fail_ind: i32) {
    let x = ptt_frame_sh[base];
    let y = ptt_frame_sh[base + 1u];
    let z = ptt_frame_sh[base + 2u];
    let scale = sqrt(x * x + y * y + z * z);

    if (scale > NORM_EPS) {
        ptt_frame_sh[base] = x / scale;
        ptt_frame_sh[base + 1u] = y / scale;
        ptt_frame_sh[base + 2u] = z / scale;
    } else {
        ptt_frame_sh[base] = 0.0;
        ptt_frame_sh[base + 1u] = 0.0;
        ptt_frame_sh[base + 2u] = 0.0;
        ptt_frame_sh[base + u32(fail_ind)] = 1.0;
    }
}

// Direct norm3 on ptt_direc
fn norm3_direc(base: u32, fail_ind: i32) {
    let x = ptt_direc[base];
    let y = ptt_direc[base + 1u];
    let z = ptt_direc[base + 2u];
    let scale = sqrt(x * x + y * y + z * z);

    if (scale > NORM_EPS) {
        ptt_direc[base] = x / scale;
        ptt_direc[base + 1u] = y / scale;
        ptt_direc[base + 2u] = z / scale;
    } else {
        ptt_direc[base] = 0.0;
        ptt_direc[base + 1u] = 0.0;
        ptt_direc[base + 2u] = 0.0;
        ptt_direc[base + u32(fail_ind)] = 1.0;
    }
}

// ── crossnorm3 on ptt_probing_frame ────────────────────────────────
// dest = normalise(src1 x src2), all offsets into ptt_probing_frame.
fn crossnorm3_probing_frame(dest: u32, src1: u32, src2: u32, fail_ind: i32) {
    ptt_probing_frame[dest] =
        ptt_probing_frame[src1 + 1u] * ptt_probing_frame[src2 + 2u] -
        ptt_probing_frame[src1 + 2u] * ptt_probing_frame[src2 + 1u];
    ptt_probing_frame[dest + 1u] =
        ptt_probing_frame[src1 + 2u] * ptt_probing_frame[src2] -
        ptt_probing_frame[src1]      * ptt_probing_frame[src2 + 2u];
    ptt_probing_frame[dest + 2u] =
        ptt_probing_frame[src1]      * ptt_probing_frame[src2 + 1u] -
        ptt_probing_frame[src1 + 1u] * ptt_probing_frame[src2];

    norm3_probing_frame(dest, fail_ind);
}

// ── crossnorm3 on ptt_frame_sh ─────────────────────────────────────
fn crossnorm3_frame(dest: u32, src1: u32, src2: u32, fail_ind: i32) {
    ptt_frame_sh[dest] =
        ptt_frame_sh[src1 + 1u] * ptt_frame_sh[src2 + 2u] -
        ptt_frame_sh[src1 + 2u] * ptt_frame_sh[src2 + 1u];
    ptt_frame_sh[dest + 1u] =
        ptt_frame_sh[src1 + 2u] * ptt_frame_sh[src2] -
        ptt_frame_sh[src1]      * ptt_frame_sh[src2 + 2u];
    ptt_frame_sh[dest + 2u] =
        ptt_frame_sh[src1]      * ptt_frame_sh[src2 + 1u] -
        ptt_frame_sh[src1 + 1u] * ptt_frame_sh[src2];

    norm3_frame(dest, fail_ind);
}

// ── interp4 — find closest ODF vertex, trilinear interp ────────────
// Returns the interpolated FOD amplitude along the probing frame tangent.
fn interp4_ptt(
    pos: vec3<f32>,
    frame_base: u32,   // offset into ptt_probing_frame for tangent direction [0..2]
    dimx: i32, dimy: i32, dimz: i32, dimt: i32,
    tidy: u32, tidx: u32
) -> f32 {
    var closest_odf_idx: i32 = 0;
    var max_cos: f32 = 0.0;

    for (var ii = i32(tidx); ii < dimt; ii += i32(THR_X_SL)) {
        let sv = load_sphere_verts_f3(u32(ii));
        let cos_sim = abs(
            sv.x * ptt_probing_frame[frame_base] +
            sv.y * ptt_probing_frame[frame_base + 1u] +
            sv.z * ptt_probing_frame[frame_base + 2u]);
        if (cos_sim > max_cos) {
            max_cos = cos_sim;
            closest_odf_idx = ii;
        }
    }
    subgroupBarrier();

    // Reduce across the subgroup
    for (var i = i32(THR_X_SL) / 2; i > 0; i /= 2) {
        let tmp = subgroupShuffleXor(max_cos, u32(i));
        let tmp_idx = subgroupShuffleXor(closest_odf_idx, u32(i));
        if (tmp > max_cos || (tmp == max_cos && tmp_idx < closest_odf_idx)) {
            max_cos = tmp;
            closest_odf_idx = tmp_idx;
        }
    }
    subgroupBarrier();

    // Trilinear interpolation of dataf at the closest ODF vertex
    let rv = trilinear_interp_dataf_single(
        dimx, dimy, dimz, dimt, closest_odf_idx, pos, tidy);

    if (rv != 0) {
        return 0.0;
    } else {
        return ptt_interp_scratch[tidy];
    }
}

// Variant reading frame direction from ptt_frame_sh instead of ptt_probing_frame.
fn interp4_ptt_frame(
    pos: vec3<f32>,
    frame_base: u32,   // offset into ptt_frame_sh for tangent direction [0..2]
    dimx: i32, dimy: i32, dimz: i32, dimt: i32,
    tidy: u32, tidx: u32
) -> f32 {
    var closest_odf_idx: i32 = 0;
    var max_cos: f32 = 0.0;

    for (var ii = i32(tidx); ii < dimt; ii += i32(THR_X_SL)) {
        let sv = load_sphere_verts_f3(u32(ii));
        let cos_sim = abs(
            sv.x * ptt_frame_sh[frame_base] +
            sv.y * ptt_frame_sh[frame_base + 1u] +
            sv.z * ptt_frame_sh[frame_base + 2u]);
        if (cos_sim > max_cos) {
            max_cos = cos_sim;
            closest_odf_idx = ii;
        }
    }
    subgroupBarrier();

    for (var i = i32(THR_X_SL) / 2; i > 0; i /= 2) {
        let tmp = subgroupShuffleXor(max_cos, u32(i));
        let tmp_idx = subgroupShuffleXor(closest_odf_idx, u32(i));
        if (tmp > max_cos || (tmp == max_cos && tmp_idx < closest_odf_idx)) {
            max_cos = tmp;
            closest_odf_idx = tmp_idx;
        }
    }
    subgroupBarrier();

    let rv = trilinear_interp_dataf_single(
        dimx, dimy, dimz, dimt, closest_odf_idx, pos, tidy);

    if (rv != 0) {
        return 0.0;
    } else {
        return ptt_interp_scratch[tidy];
    }
}

// ── prepare_propagator ─────────────────────────────────────────────
// Build 3x3 propagator matrix from curvatures k1, k2 and arclength.
// Writes 9 floats to ptt_probing_prop[prop_base..prop_base+8].
fn prepare_propagator_ptt(k1_in: f32, k2_in: f32, arclength: f32, prop_base: u32) {
    var k1 = k1_in;
    var k2 = k2_in;

    if (abs(k1) < K_SMALL && abs(k2) < K_SMALL) {
        ptt_probing_prop[prop_base]     = arclength;
        ptt_probing_prop[prop_base + 1u] = 0.0;
        ptt_probing_prop[prop_base + 2u] = 0.0;
        ptt_probing_prop[prop_base + 3u] = 1.0;
        ptt_probing_prop[prop_base + 4u] = 0.0;
        ptt_probing_prop[prop_base + 5u] = 0.0;
        ptt_probing_prop[prop_base + 6u] = 0.0;
        ptt_probing_prop[prop_base + 7u] = 0.0;
        ptt_probing_prop[prop_base + 8u] = 1.0;
    } else {
        if (abs(k1) < K_SMALL) { k1 = K_SMALL; }
        if (abs(k2) < K_SMALL) { k2 = K_SMALL; }
        let k     = sqrt(k1 * k1 + k2 * k2);
        let sinkt = sin(k * arclength);
        let coskt = cos(k * arclength);
        let kk    = 1.0 / (k * k);

        ptt_probing_prop[prop_base]     = sinkt / k;
        ptt_probing_prop[prop_base + 1u] = k1 * (1.0 - coskt) * kk;
        ptt_probing_prop[prop_base + 2u] = k2 * (1.0 - coskt) * kk;
        ptt_probing_prop[prop_base + 3u] = coskt;
        ptt_probing_prop[prop_base + 4u] = k1 * sinkt / k;
        ptt_probing_prop[prop_base + 5u] = k2 * sinkt / k;
        ptt_probing_prop[prop_base + 6u] = -ptt_probing_prop[prop_base + 5u];
        ptt_probing_prop[prop_base + 7u] = k1 * k2 * (coskt - 1.0) * kk;
        ptt_probing_prop[prop_base + 8u] = (k1 * k1 + k2 * k2 * coskt) * kk;
    }
}

// ── random_normal_ptt ──────────────────────────────────────────────
// Generate a random normal vector perpendicular to ptt_probing_frame[pf_base..pf_base+2].
// Writes result to ptt_probing_frame[pf_base+3..pf_base+5].
// Returns updated PhiloxState.
fn random_normal_ptt_fn(st: PhiloxState, pf_base: u32) -> PhiloxState {
    var rng = st;

    let nr1 = philox_normal(rng); rng = nr1.state;
    let nr2 = philox_normal(rng); rng = nr2.state;
    let nr3 = philox_normal(rng); rng = nr3.state;

    ptt_probing_frame[pf_base + 3u] = nr1.value;
    ptt_probing_frame[pf_base + 4u] = nr2.value;
    ptt_probing_frame[pf_base + 5u] = nr3.value;

    let dot_val = ptt_probing_frame[pf_base + 3u] * ptt_probing_frame[pf_base] +
                  ptt_probing_frame[pf_base + 4u] * ptt_probing_frame[pf_base + 1u] +
                  ptt_probing_frame[pf_base + 5u] * ptt_probing_frame[pf_base + 2u];

    ptt_probing_frame[pf_base + 3u] -= dot_val * ptt_probing_frame[pf_base];
    ptt_probing_frame[pf_base + 4u] -= dot_val * ptt_probing_frame[pf_base + 1u];
    ptt_probing_frame[pf_base + 5u] -= dot_val * ptt_probing_frame[pf_base + 2u];

    let n2 = ptt_probing_frame[pf_base + 3u] * ptt_probing_frame[pf_base + 3u] +
             ptt_probing_frame[pf_base + 4u] * ptt_probing_frame[pf_base + 4u] +
             ptt_probing_frame[pf_base + 5u] * ptt_probing_frame[pf_base + 5u];

    if (n2 < NORM_EPS) {
        let abs_x = abs(ptt_probing_frame[pf_base]);
        let abs_y = abs(ptt_probing_frame[pf_base + 1u]);
        let abs_z = abs(ptt_probing_frame[pf_base + 2u]);

        if (abs_x <= abs_y && abs_x <= abs_z) {
            ptt_probing_frame[pf_base + 3u] = 0.0;
            ptt_probing_frame[pf_base + 4u] = ptt_probing_frame[pf_base + 2u];
            ptt_probing_frame[pf_base + 5u] = -ptt_probing_frame[pf_base + 1u];
        } else if (abs_y <= abs_z) {
            ptt_probing_frame[pf_base + 3u] = -ptt_probing_frame[pf_base + 2u];
            ptt_probing_frame[pf_base + 4u] = 0.0;
            ptt_probing_frame[pf_base + 5u] = ptt_probing_frame[pf_base];
        } else {
            ptt_probing_frame[pf_base + 3u] = ptt_probing_frame[pf_base + 1u];
            ptt_probing_frame[pf_base + 4u] = -ptt_probing_frame[pf_base];
            ptt_probing_frame[pf_base + 5u] = 0.0;
        }
    }
    return rng;
}

// ── get_probing_frame_init ─────────────────────────────────────────
// Build a fresh probing frame from the tangent direction in ptt_frame_sh.
// frame_base: offset into ptt_frame_sh for the tangent [0..2].
// pf_base: offset into ptt_probing_frame for the output 9-element frame.
// Returns updated PhiloxState.
fn get_probing_frame_init_fn(frame_base: u32, st: PhiloxState, pf_base: u32) -> PhiloxState {
    // Copy tangent from frame_sh to probing_frame
    for (var ii = 0u; ii < 3u; ii++) {
        ptt_probing_frame[pf_base + ii] = ptt_frame_sh[frame_base + ii];
    }
    norm3_probing_frame(pf_base, 0);

    let rng = random_normal_ptt_fn(st, pf_base);
    norm3_probing_frame(pf_base + 3u, 1);

    // binorm = tangent x normal
    crossnorm3_probing_frame(pf_base + 6u, pf_base, pf_base + 3u, 2);

    return rng;
}

// ── get_probing_frame_noinit ───────────────────────────────────────
// Copy existing frame from ptt_frame_sh to ptt_probing_frame.
fn get_probing_frame_noinit_fn(frame_base: u32, pf_base: u32) {
    for (var ii = 0u; ii < 9u; ii++) {
        ptt_probing_frame[pf_base + ii] = ptt_frame_sh[frame_base + ii];
    }
}

// ── propagate_frame ────────────────────────────────────────────────
// Apply propagator matrix to the probing frame, re-orthonormalise,
// and output direction. All arrays are at tidy-offset within their
// respective workgroup arrays.
// prop_base: into ptt_probing_prop (9 floats)
// pf_base: into ptt_probing_frame (9 floats: tangent, normal, binormal)
// direc_base: into ptt_direc (3 floats)
fn propagate_frame_ptt(prop_base: u32, pf_base: u32, direc_base: u32) {
    var tmp: array<f32, 3>;

    for (var ii = 0u; ii < 3u; ii++) {
        ptt_direc[direc_base + ii] =
            ptt_probing_prop[prop_base]      * ptt_probing_frame[pf_base + ii] +
            ptt_probing_prop[prop_base + 1u] * ptt_probing_frame[pf_base + 3u + ii] +
            ptt_probing_prop[prop_base + 2u] * ptt_probing_frame[pf_base + 6u + ii];
        tmp[ii] =
            ptt_probing_prop[prop_base + 3u] * ptt_probing_frame[pf_base + ii] +
            ptt_probing_prop[prop_base + 4u] * ptt_probing_frame[pf_base + 3u + ii] +
            ptt_probing_prop[prop_base + 5u] * ptt_probing_frame[pf_base + 6u + ii];
        ptt_probing_frame[pf_base + 6u + ii] =
            ptt_probing_prop[prop_base + 6u] * ptt_probing_frame[pf_base + ii] +
            ptt_probing_prop[prop_base + 7u] * ptt_probing_frame[pf_base + 3u + ii] +
            ptt_probing_prop[prop_base + 8u] * ptt_probing_frame[pf_base + 6u + ii];
    }

    // Normalise tangent (in tmp), write back to probing_frame[0..2]
    let scale_t = sqrt(tmp[0] * tmp[0] + tmp[1] * tmp[1] + tmp[2] * tmp[2]);
    if (scale_t > NORM_EPS) {
        ptt_probing_frame[pf_base]      = tmp[0] / scale_t;
        ptt_probing_frame[pf_base + 1u] = tmp[1] / scale_t;
        ptt_probing_frame[pf_base + 2u] = tmp[2] / scale_t;
    } else {
        ptt_probing_frame[pf_base]      = 0.0;
        ptt_probing_frame[pf_base + 1u] = 0.0;
        ptt_probing_frame[pf_base + 2u] = 0.0;
        ptt_probing_frame[pf_base]      = 1.0;
    }

    // normal = cross(binorm, tangent), binorm = cross(tangent, normal)
    crossnorm3_probing_frame(pf_base + 3u, pf_base + 6u, pf_base, 1);
    crossnorm3_probing_frame(pf_base + 6u, pf_base, pf_base + 3u, 2);
}

// ── calculate_data_support ─────────────────────────────────────────
// Probe forward along a candidate curve and accumulate FOD amplitudes.
fn calculate_data_support_ptt(
    support_in: f32,
    pos: vec3<f32>,
    dimx: i32, dimy: i32, dimz: i32, dimt: i32,
    probe_step_size: f32,
    absolpmf_thresh: f32,
    prop_base: u32,    // into ptt_probing_prop
    direc_base: u32,   // into ptt_direc
    pos_base: u32,     // into ptt_probing_pos
    k1_idx: u32,       // into ptt_k1_probe
    k2_idx: u32,       // into ptt_k2_probe
    pf_base: u32,      // into ptt_probing_frame
    tidy: u32, tidx: u32
) -> f32 {
    var support = support_in;

    if (tidx == 0u) {
        prepare_propagator_ptt(ptt_k1_probe[k1_idx], ptt_k2_probe[k2_idx],
                               probe_step_size, prop_base);
        ptt_probing_pos[pos_base]      = pos.x;
        ptt_probing_pos[pos_base + 1u] = pos.y;
        ptt_probing_pos[pos_base + 2u] = pos.z;
    }
    subgroupBarrier();

    for (var ii = 0; ii < PROBE_QUALITY; ii++) {
        if (tidx == 0u) {
            propagate_frame_ptt(prop_base, pf_base, direc_base);

            ptt_probing_pos[pos_base]      += ptt_direc[direc_base];
            ptt_probing_pos[pos_base + 1u] += ptt_direc[direc_base + 1u];
            ptt_probing_pos[pos_base + 2u] += ptt_direc[direc_base + 2u];
        }
        subgroupBarrier();

        let probe_pos = vec3<f32>(
            ptt_probing_pos[pos_base],
            ptt_probing_pos[pos_base + 1u],
            ptt_probing_pos[pos_base + 2u]);

        let fod_amp = interp4_ptt(
            probe_pos, pf_base,
            dimx, dimy, dimz, dimt,
            tidy, tidx);

        if (!ALLOW_WEAK_LINK && (fod_amp < absolpmf_thresh)) {
            return 0.0;
        }
        support += fod_amp;
    }
    return support;
}

// ── Result types for PTT functions ─────────────────────────────────

struct GetDirPttResult {
    ndir: i32,
    state: PhiloxState,
}

struct PttInitResult {
    success: bool,
    state: PhiloxState,
}

struct TrackerPttResult {
    tissue_class: i32,
    state: PhiloxState,
}

// ── get_direction_ptt_init ─────────────────────────────────────────
// IS_INIT variant: set frame tangent from dir, sample disc, find supported
// curvature. Writes the original direction to ptt_dirs and the probing
// frame to ptt_frame_sh.
fn get_direction_ptt_init_fn(
    st: PhiloxState,
    max_angle: f32,
    step_size: f32,
    dir: vec3<f32>,
    frame_base: u32,   // offset into ptt_frame_sh
    dimx: i32, dimy: i32, dimz: i32, dimt: i32,
    pos: vec3<f32>,
    dirs_base: u32,    // offset into ptt_dirs (tidy * 3)
    // PTT workspace offsets
    face_cdf_base: u32,
    vert_pdf_base: u32,
    pf_base: u32,      // into ptt_probing_frame
    k1_idx: u32,       // into ptt_k1_probe
    k2_idx: u32,       // into ptt_k2_probe
    prop_base: u32,    // into ptt_probing_prop
    direc_base: u32,   // into ptt_direc
    pos_base: u32,     // into ptt_probing_pos
    tidy: u32, tidx: u32
) -> GetDirPttResult {
    var rng = st;

    let probe_step_size = (step_size / f32(PROBE_FRAC)) / f32(PROBE_QUALITY - 1);
    let max_curvature = 2.0 * sin(max_angle / 2.0) / (step_size / f32(PROBE_FRAC));
    let absolpmf_thresh = PMF_THRESHOLD_P * sg_max_reduce_dataf(dimt, REAL_MIN, tidx);

    subgroupBarrier();

    // IS_INIT: set frame tangent from dir
    if (tidx == 0u) {
        ptt_frame_sh[frame_base]      = dir.x;
        ptt_frame_sh[frame_base + 1u] = dir.y;
        ptt_frame_sh[frame_base + 2u] = dir.z;
    }

    let first_val = interp4_ptt_frame(
        pos, frame_base,
        dimx, dimy, dimz, dimt,
        tidy, tidx);
    subgroupBarrier();

    // Calculate vert_pdf
    var support_found = false;
    for (var ii = 0; ii < i32(DISC_VERT_CNT); ii++) {
        if (tidx == 0u) {
            ptt_k1_probe[k1_idx] = DISC_VERT[u32(ii) * 2u] * max_curvature;
            ptt_k2_probe[k2_idx] = DISC_VERT[u32(ii) * 2u + 1u] * max_curvature;
            rng = get_probing_frame_init_fn(frame_base, rng, pf_base);
        }
        subgroupBarrier();

        let this_support = calculate_data_support_ptt(
            first_val, pos,
            dimx, dimy, dimz, dimt,
            probe_step_size, absolpmf_thresh,
            prop_base, direc_base, pos_base,
            k1_idx, k2_idx, pf_base,
            tidy, tidx);

        if (this_support < f32(PROBE_QUALITY) * absolpmf_thresh) {
            if (tidx == 0u) {
                ptt_vert_pdf[vert_pdf_base + u32(ii)] = 0.0;
            }
        } else {
            if (tidx == 0u) {
                ptt_vert_pdf[vert_pdf_base + u32(ii)] = this_support;
            }
            support_found = true;
        }
    }
    if (!support_found) {
        return GetDirPttResult(0, rng);
    }

    // Initialise face_cdf
    for (var ii = i32(tidx); ii < i32(DISC_FACE_CNT); ii += i32(THR_X_SL)) {
        ptt_face_cdf[face_cdf_base + u32(ii)] = 0.0;
    }
    subgroupBarrier();

    // Move vert PDF to face PDF
    for (var ii = i32(tidx); ii < i32(DISC_FACE_CNT); ii += i32(THR_X_SL)) {
        // IS_INIT: even go with faces that are not fully supported
        for (var jj = 0; jj < 3; jj++) {
            let vert_val = ptt_vert_pdf[vert_pdf_base + u32(DISC_FACE[u32(ii) * 3u + u32(jj)])];
            ptt_face_cdf[face_cdf_base + u32(ii)] += vert_val;
        }
    }
    subgroupBarrier();

    // Prefix sum and check for zero total
    prefix_sum_ptt_face_cdf(face_cdf_base, i32(DISC_FACE_CNT), tidx);
    let last_cdf = ptt_face_cdf[face_cdf_base + DISC_FACE_CNT - 1u];

    if (last_cdf == 0.0) {
        return GetDirPttResult(0, rng);
    }

    // Rejection sampling
    for (var ii = 0; ii < TRIES_PER_REJECTION_SAMPLING; ii++) {
        if (tidx == 0u) {
            let ur1 = philox_uniform(rng); rng = ur1.state;
            let ur2 = philox_uniform(rng); rng = ur2.state;
            var r1 = ur1.value;
            var r2 = ur2.value;
            if (r1 + r2 > 1.0) {
                r1 = 1.0 - r1;
                r2 = 1.0 - r2;
            }

            let ur3 = philox_uniform(rng); rng = ur3.state;
            let tmp_sample = ur3.value * last_cdf;
            var jj: i32 = 0;
            for (; jj < i32(DISC_FACE_CNT); jj++) {
                if (ptt_face_cdf[face_cdf_base + u32(jj)] >= tmp_sample) {
                    break;
                }
            }

            let vx0 = max_curvature * DISC_VERT[u32(DISC_FACE[u32(jj) * 3u]) * 2u];
            let vx1 = max_curvature * DISC_VERT[u32(DISC_FACE[u32(jj) * 3u + 1u]) * 2u];
            let vx2 = max_curvature * DISC_VERT[u32(DISC_FACE[u32(jj) * 3u + 2u]) * 2u];

            let vy0 = max_curvature * DISC_VERT[u32(DISC_FACE[u32(jj) * 3u]) * 2u + 1u];
            let vy1 = max_curvature * DISC_VERT[u32(DISC_FACE[u32(jj) * 3u + 1u]) * 2u + 1u];
            let vy2 = max_curvature * DISC_VERT[u32(DISC_FACE[u32(jj) * 3u + 2u]) * 2u + 1u];

            ptt_k1_probe[k1_idx] = vx0 + r1 * (vx1 - vx0) + r2 * (vx2 - vx0);
            ptt_k2_probe[k2_idx] = vy0 + r1 * (vy1 - vy0) + r2 * (vy2 - vy0);
            rng = get_probing_frame_init_fn(frame_base, rng, pf_base);
        }
        subgroupBarrier();

        let this_support = calculate_data_support_ptt(
            first_val, pos,
            dimx, dimy, dimz, dimt,
            probe_step_size, absolpmf_thresh,
            prop_base, direc_base, pos_base,
            k1_idx, k2_idx, pf_base,
            tidy, tidx);
        subgroupBarrier();

        if (this_support < f32(PROBE_QUALITY) * absolpmf_thresh) {
            continue;
        }

        // IS_INIT: store the original direction
        if (tidx == 0u) {
            ptt_dirs[dirs_base]      = dir.x;
            ptt_dirs[dirs_base + 1u] = dir.y;
            ptt_dirs[dirs_base + 2u] = dir.z;
        }

        if (tidx < 9u) {
            ptt_frame_sh[frame_base + tidx] = ptt_probing_frame[pf_base + tidx];
        }
        subgroupBarrier();
        return GetDirPttResult(1, rng);
    }
    return GetDirPttResult(0, rng);
}

// ── get_direction_ptt_noinit ───────────────────────────────────────
// Non-init variant: frame_sh is already populated. Propagates 1/STEP_FRAC
// of a step and outputs direction.
fn get_direction_ptt_noinit_fn(
    st: PhiloxState,
    max_angle: f32,
    step_size: f32,
    dir: vec3<f32>,
    frame_base: u32,
    dimx: i32, dimy: i32, dimz: i32, dimt: i32,
    pos: vec3<f32>,
    dirs_base: u32,
    face_cdf_base: u32,
    vert_pdf_base: u32,
    pf_base: u32,
    k1_idx: u32,
    k2_idx: u32,
    prop_base: u32,
    direc_base: u32,
    pos_base: u32,
    tidy: u32, tidx: u32
) -> GetDirPttResult {
    var rng = st;

    let probe_step_size = (step_size / f32(PROBE_FRAC)) / f32(PROBE_QUALITY - 1);
    let max_curvature = 2.0 * sin(max_angle / 2.0) / (step_size / f32(PROBE_FRAC));
    let absolpmf_thresh = PMF_THRESHOLD_P * sg_max_reduce_dataf(dimt, REAL_MIN, tidx);

    subgroupBarrier();

    // Non-init: frame_sh is already populated

    let first_val = interp4_ptt_frame(
        pos, frame_base,
        dimx, dimy, dimz, dimt,
        tidy, tidx);
    subgroupBarrier();

    // Calculate vert_pdf
    var support_found = false;
    for (var ii = 0; ii < i32(DISC_VERT_CNT); ii++) {
        if (tidx == 0u) {
            ptt_k1_probe[k1_idx] = DISC_VERT[u32(ii) * 2u] * max_curvature;
            ptt_k2_probe[k2_idx] = DISC_VERT[u32(ii) * 2u + 1u] * max_curvature;
            get_probing_frame_noinit_fn(frame_base, pf_base);
        }
        subgroupBarrier();

        let this_support = calculate_data_support_ptt(
            first_val, pos,
            dimx, dimy, dimz, dimt,
            probe_step_size, absolpmf_thresh,
            prop_base, direc_base, pos_base,
            k1_idx, k2_idx, pf_base,
            tidy, tidx);

        if (this_support < f32(PROBE_QUALITY) * absolpmf_thresh) {
            if (tidx == 0u) {
                ptt_vert_pdf[vert_pdf_base + u32(ii)] = 0.0;
            }
        } else {
            if (tidx == 0u) {
                ptt_vert_pdf[vert_pdf_base + u32(ii)] = this_support;
            }
            support_found = true;
        }
    }
    if (!support_found) {
        return GetDirPttResult(0, rng);
    }

    // Initialise face_cdf
    for (var ii = i32(tidx); ii < i32(DISC_FACE_CNT); ii += i32(THR_X_SL)) {
        ptt_face_cdf[face_cdf_base + u32(ii)] = 0.0;
    }
    subgroupBarrier();

    // Move vert PDF to face PDF
    for (var ii = i32(tidx); ii < i32(DISC_FACE_CNT); ii += i32(THR_X_SL)) {
        var all_verts_valid = true;
        for (var jj = 0; jj < 3; jj++) {
            let vert_val = ptt_vert_pdf[vert_pdf_base + u32(DISC_FACE[u32(ii) * 3u + u32(jj)])];
            if (vert_val == 0.0) {
                all_verts_valid = false;  // Non-init: reject faces with unsupported vertices
            }
            ptt_face_cdf[face_cdf_base + u32(ii)] += vert_val;
        }
        if (!all_verts_valid) {
            ptt_face_cdf[face_cdf_base + u32(ii)] = 0.0;
        }
    }
    subgroupBarrier();

    // Prefix sum and check for zero total
    prefix_sum_ptt_face_cdf(face_cdf_base, i32(DISC_FACE_CNT), tidx);
    let last_cdf = ptt_face_cdf[face_cdf_base + DISC_FACE_CNT - 1u];

    if (last_cdf == 0.0) {
        return GetDirPttResult(0, rng);
    }

    // Rejection sampling
    for (var ii = 0; ii < TRIES_PER_REJECTION_SAMPLING; ii++) {
        if (tidx == 0u) {
            let ur1 = philox_uniform(rng); rng = ur1.state;
            let ur2 = philox_uniform(rng); rng = ur2.state;
            var r1 = ur1.value;
            var r2 = ur2.value;
            if (r1 + r2 > 1.0) {
                r1 = 1.0 - r1;
                r2 = 1.0 - r2;
            }

            let ur3 = philox_uniform(rng); rng = ur3.state;
            let tmp_sample = ur3.value * last_cdf;
            var jj: i32 = 0;
            for (; jj < i32(DISC_FACE_CNT); jj++) {
                if (ptt_face_cdf[face_cdf_base + u32(jj)] >= tmp_sample) {
                    break;
                }
            }

            let vx0 = max_curvature * DISC_VERT[u32(DISC_FACE[u32(jj) * 3u]) * 2u];
            let vx1 = max_curvature * DISC_VERT[u32(DISC_FACE[u32(jj) * 3u + 1u]) * 2u];
            let vx2 = max_curvature * DISC_VERT[u32(DISC_FACE[u32(jj) * 3u + 2u]) * 2u];

            let vy0 = max_curvature * DISC_VERT[u32(DISC_FACE[u32(jj) * 3u]) * 2u + 1u];
            let vy1 = max_curvature * DISC_VERT[u32(DISC_FACE[u32(jj) * 3u + 1u]) * 2u + 1u];
            let vy2 = max_curvature * DISC_VERT[u32(DISC_FACE[u32(jj) * 3u + 2u]) * 2u + 1u];

            ptt_k1_probe[k1_idx] = vx0 + r1 * (vx1 - vx0) + r2 * (vx2 - vx0);
            ptt_k2_probe[k2_idx] = vy0 + r1 * (vy1 - vy0) + r2 * (vy2 - vy0);
            get_probing_frame_noinit_fn(frame_base, pf_base);
        }
        subgroupBarrier();

        let this_support = calculate_data_support_ptt(
            first_val, pos,
            dimx, dimy, dimz, dimt,
            probe_step_size, absolpmf_thresh,
            prop_base, direc_base, pos_base,
            k1_idx, k2_idx, pf_base,
            tidy, tidx);
        subgroupBarrier();

        if (this_support < f32(PROBE_QUALITY) * absolpmf_thresh) {
            continue;
        }

        // Non-init: propagate 1/STEP_FRAC of a step and output direction
        if (tidx == 0u) {
            prepare_propagator_ptt(
                ptt_k1_probe[k1_idx], ptt_k2_probe[k2_idx],
                step_size / f32(STEP_FRAC), prop_base);
            get_probing_frame_noinit_fn(frame_base, pf_base);
            propagate_frame_ptt(prop_base, pf_base, direc_base);

            // Normalise direction
            norm3_direc(direc_base, 0);

            ptt_dirs[dirs_base]      = ptt_direc[direc_base];
            ptt_dirs[dirs_base + 1u] = ptt_direc[direc_base + 1u];
            ptt_dirs[dirs_base + 2u] = ptt_direc[direc_base + 2u];
        }

        if (tidx < 9u) {
            ptt_frame_sh[frame_base + tidx] = ptt_probing_frame[pf_base + tidx];
        }
        subgroupBarrier();
        return GetDirPttResult(1, rng);
    }
    return GetDirPttResult(0, rng);
}

// ── init_frame_ptt ─────────────────────────────────────────────────
// Initialise the parallel transport frame for a new streamline.
// Tries the negative direction first, then the positive, and flips if needed.
fn init_frame_ptt_fn(
    st: PhiloxState,
    max_angle: f32,
    step_size: f32,
    first_step: vec3<f32>,
    dimx: i32, dimy: i32, dimz: i32, dimt: i32,
    seed: vec3<f32>,
    frame_base: u32,
    dirs_base: u32,
    face_cdf_base: u32,
    vert_pdf_base: u32,
    pf_base: u32,
    k1_idx: u32,
    k2_idx: u32,
    prop_base: u32,
    direc_base: u32,
    pos_base: u32,
    tidy: u32, tidx: u32
) -> PttInitResult {
    var rng = st;

    // Try with negated direction first
    let neg_dir = vec3<f32>(-first_step.x, -first_step.y, -first_step.z);
    let r1 = get_direction_ptt_init_fn(
        rng, max_angle, step_size, neg_dir, frame_base,
        dimx, dimy, dimz, dimt, seed, dirs_base,
        face_cdf_base, vert_pdf_base, pf_base,
        k1_idx, k2_idx, prop_base, direc_base, pos_base,
        tidy, tidx);
    rng = r1.state;
    subgroupBarrier();

    if (r1.ndir == 0) {
        // Try the positive direction
        let r2 = get_direction_ptt_init_fn(
            rng, max_angle, step_size, first_step, frame_base,
            dimx, dimy, dimz, dimt, seed, dirs_base,
            face_cdf_base, vert_pdf_base, pf_base,
            k1_idx, k2_idx, prop_base, direc_base, pos_base,
            tidy, tidx);
        rng = r2.state;
        subgroupBarrier();

        if (r2.ndir == 0) {
            return PttInitResult(false, rng);
        } else {
            // Flip the frame
            if (tidx == 0u) {
                for (var ii = 0u; ii < 9u; ii++) {
                    ptt_frame_sh[frame_base + ii] = -ptt_frame_sh[frame_base + ii];
                }
            }
            subgroupBarrier();
        }
    }

    // Save flipped frame for the second (forward) run
    if (tidx == 0u) {
        for (var ii = 0u; ii < 9u; ii++) {
            ptt_frame_sh[frame_base + 9u + ii] = -ptt_frame_sh[frame_base + ii];
        }
    }
    subgroupBarrier();
    return PttInitResult(true, rng);
}

// ── tracker_ptt — step along streamline with parallel transport ────
// Takes fractional steps (STEP_FRAC sub-steps per full step), only
// stores every STEP_FRAC'th point.
fn tracker_ptt_fn(
    st: PhiloxState,
    max_angle: f32, tc_threshold: f32, step_size: f32,
    seed: vec3<f32>, first_step: vec3<f32>,
    dimx: i32, dimy: i32, dimz: i32, dimt: i32,
    frame_base: u32,    // into ptt_frame_sh
    sline_base: u32,    // flat f32 base into sline buffer
    dirs_base: u32,     // into ptt_dirs
    face_cdf_base: u32,
    vert_pdf_base: u32,
    pf_base: u32,
    k1_idx: u32,
    k2_idx: u32,
    prop_base: u32,
    direc_base: u32,
    pos_base: u32,
    tidy: u32, tidx: u32,
    use_stepsB: bool
) -> TrackerPttResult {
    var rng = st;
    var tissue_class: i32 = TRACKPOINT;
    var point = seed;
    var direction = first_step;

    if (tidx == 0u) {
        let off = sline_base;
        sline[off] = point.x;
        sline[off + 1u] = point.y;
        sline[off + 2u] = point.z;
    }
    subgroupBarrier();

    var i: i32 = 1;
    for (; i < MAX_SLINE_LEN * STEP_FRAC; i++) {
        let gdr = get_direction_ptt_noinit_fn(
            rng, max_angle, step_size, direction, frame_base,
            dimx, dimy, dimz, dimt, point, dirs_base,
            face_cdf_base, vert_pdf_base, pf_base,
            k1_idx, k2_idx, prop_base, direc_base, pos_base,
            tidy, tidx);
        rng = gdr.state;
        subgroupBarrier();

        direction = vec3<f32>(
            ptt_dirs[dirs_base],
            ptt_dirs[dirs_base + 1u],
            ptt_dirs[dirs_base + 2u]);
        subgroupBarrier();

        if (gdr.ndir == 0) {
            break;
        }

        // voxel_size is (1,1,1) so division by voxel_size is identity
        let frac_step = step_size / f32(STEP_FRAC);
        point.x += direction.x * frac_step;
        point.y += direction.y * frac_step;
        point.z += direction.z * frac_step;

        if (tidx == 0u && (i % STEP_FRAC) == 0) {
            let step_idx = u32(i / STEP_FRAC);
            let off = sline_base + step_idx * 3u;
            sline[off] = point.x;
            sline[off + 1u] = point.y;
            sline[off + 2u] = point.z;
        }
        subgroupBarrier();

        if ((i % STEP_FRAC) == 0) {
            tissue_class = check_point_fn(tc_threshold, point, dimx, dimy, dimz, tidx, tidy);

            if (tissue_class == ENDPOINT ||
                tissue_class == INVALIDPOINT ||
                tissue_class == OUTSIDEIMAGE) {
                break;
            }
        }
    }

    let nsteps = i / STEP_FRAC;
    if (use_stepsB) {
        ptt_stepsB[tidy] = nsteps;
    } else {
        ptt_stepsF[tidy] = nsteps;
    }

    // If stopped mid-fraction, store the final point
    if ((i % STEP_FRAC) != 0 && i < STEP_FRAC * (MAX_SLINE_LEN - 1)) {
        let final_step = nsteps + 1;
        if (use_stepsB) {
            ptt_stepsB[tidy] = final_step;
        } else {
            ptt_stepsF[tidy] = final_step;
        }
        if (tidx == 0u) {
            let off = sline_base + u32(final_step) * 3u;
            sline[off] = point.x;
            sline[off + 1u] = point.y;
            sline[off + 2u] = point.z;
        }
    }

    return TrackerPttResult(tissue_class, rng);
}

// ── genStreamlinesMergePtt_k ───────────────────────────────────────
// PTT generation kernel. Uses the same buffer layout as the Prob kernel
// (ProbTrackingParams, 2 bind groups, 11 buffers) so the Python dispatch
// code is shared. PTT reuses Prob's getNumStreamlinesProb_k for initial
// direction finding.
@compute @workgroup_size(32, 2, 1)
fn genStreamlinesMergePtt_k(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) gid: vec3<u32>
) {
    let tidx = tid.x;
    let tidy = tid.y;
    let slid = gid.x * BLOCK_Y + tidy;

    if (i32(slid) >= params.nseed) { return; }

    let global_id = gid.x * BLOCK_Y * THR_X_SL + THR_X_SL * tidy + tidx;
    var st = philox_init(
        u32(params.rng_seed_lo), u32(params.rng_seed_hi), global_id + 1u, 0u);

    // Pre-compute tidy-based offsets into PTT workgroup arrays
    let frame_base = tidy * 18u;
    let dirs_base = tidy * 3u;
    let face_cdf_base = tidy * DISC_FACE_CNT;
    let vert_pdf_base = tidy * DISC_VERT_CNT;
    let pf_base = tidy * 9u;
    let k1_idx = tidy;
    let k2_idx = tidy;
    let prop_base = tidy * 9u;
    let direc_base = tidy * 3u;
    let pos_base = tidy * 3u;

    // ── per-seed loop ──────────────────────────────────────────────
    let seed = load_seeds_f3(slid);

    let ndir = slineOutOff[slid + 1u] - slineOutOff[slid];
    subgroupBarrier();

    var sline_off = slineOutOff[slid];

    for (var i = 0; i < ndir; i++) {
        let dir_idx = slid * u32(params.samplm_nr) + u32(i);
        let first_step = load_shDir0_f3(dir_idx);

        let sline_base = u32(sline_off) * u32(MAX_SLINE_LEN) * 2u * 3u;

        if (tidx == 0u) {
            slineSeed[sline_off] = i32(slid);
        }

        // PTT frame initialization
        let init_r = init_frame_ptt_fn(
            st, params.max_angle, params.step_size, first_step,
            params.dimx, params.dimy, params.dimz, params.dimt,
            seed, frame_base, dirs_base,
            face_cdf_base, vert_pdf_base, pf_base,
            k1_idx, k2_idx, prop_base, direc_base, pos_base,
            tidy, tidx);
        st = init_r.state;

        if (!init_r.success) {
            // Init failed — store single-point streamline
            if (tidx == 0u) {
                slineLen[sline_off] = 1;
                sline[sline_base] = seed.x;
                sline[sline_base + 1u] = seed.y;
                sline[sline_base + 2u] = seed.z;
            }
            subgroupBarrier();
            sline_off += 1;
            continue;
        }

        // Backward tracking (using frame[0:9])
        let neg_step = vec3<f32>(-first_step.x, -first_step.y, -first_step.z);
        let trB = tracker_ptt_fn(
            st, params.max_angle, params.tc_threshold, params.step_size,
            seed, neg_step,
            params.dimx, params.dimy, params.dimz, params.dimt,
            frame_base,      // backward frame = first 9 elements
            sline_base, dirs_base,
            face_cdf_base, vert_pdf_base, pf_base,
            k1_idx, k2_idx, prop_base, direc_base, pos_base,
            tidy, tidx, true);
        st = trB.state;

        let stepsB = ptt_stepsB[tidy];

        // Reverse backward streamline
        for (var j = i32(tidx); j < stepsB / 2; j += i32(THR_X_SL)) {
            let a_off = sline_base + u32(j) * 3u;
            let b_off = sline_base + u32(stepsB - 1 - j) * 3u;
            let pa = vec3<f32>(sline[a_off], sline[a_off + 1u], sline[a_off + 2u]);
            let pb = vec3<f32>(sline[b_off], sline[b_off + 1u], sline[b_off + 2u]);
            sline[a_off] = pb.x; sline[a_off + 1u] = pb.y; sline[a_off + 2u] = pb.z;
            sline[b_off] = pa.x; sline[b_off + 1u] = pa.y; sline[b_off + 2u] = pa.z;
        }

        // Forward tracking (using frame[9:18])
        let fwd_sline_base = sline_base + u32(stepsB - 1) * 3u;
        let trF = tracker_ptt_fn(
            st, params.max_angle, params.tc_threshold, params.step_size,
            seed, first_step,
            params.dimx, params.dimy, params.dimz, params.dimt,
            frame_base + 9u,  // forward frame = last 9 elements
            fwd_sline_base, dirs_base,
            face_cdf_base, vert_pdf_base, pf_base,
            k1_idx, k2_idx, prop_base, direc_base, pos_base,
            tidy, tidx, false);
        st = trF.state;

        if (tidx == 0u) {
            slineLen[sline_off] = stepsB - 1 + ptt_stepsF[tidy];
        }

        sline_off += 1;
    }
}
