// boot.wgsl — Bootstrap streamline generation kernels.
// Mirrors cuslines/metal_shaders/boot.metal.
//
// Compiled as a SEPARATE shader module from generate_streamlines.wgsl.
// Concatenated AFTER: globals.wgsl, types.wgsl, philox_rng.wgsl,
// utils.wgsl, warp_sort.wgsl, tracking_helpers.wgsl.
//
// Key WGSL adaptations vs Metal:
//   - PhiloxState is pass-by-value; every function returns modified state
//   - No ptr<storage> function params; use module-scope buffer access
//     with read_matrix() dispatch for H/R/delta_b/delta_q/sampling_matrix
//   - Workgroup arrays at module scope with compile-time constant sizes
//   - subgroupShuffleXor / subgroupShuffleDown / subgroupBallot / subgroupBarrier
//   - float3 stored as 3 contiguous f32 in flat storage buffers

// ── parameter struct ────────────────────────────────────────────────
struct BootTrackingParams {
    max_angle: f32,
    tc_threshold: f32,
    step_size: f32,
    relative_peak_thresh: f32,
    min_separation_angle: f32,
    min_signal: f32,
    rng_seed_lo: i32,
    rng_seed_hi: i32,
    rng_offset: i32,
    nseed: i32,
    dimx: i32,
    dimy: i32,
    dimz: i32,
    dimt: i32,
    samplm_nr: i32,
    num_edges: i32,
    delta_nr: i32,
    model_type: i32,
}

// ── buffer bindings ─────────────────────────────────────────────────
// Group 0: common read-only data
@group(0) @binding(0) var<storage, read> params: BootTrackingParams;
@group(0) @binding(1) var<storage, read> seeds: array<f32>;
@group(0) @binding(2) var<storage, read> dataf: array<f32>;
@group(0) @binding(3) var<storage, read> metric_map: array<f32>;
@group(0) @binding(4) var<storage, read> sphere_vertices: array<f32>;
@group(0) @binding(5) var<storage, read> sphere_edges: array<i32>;

// Group 1: model matrices and mask
@group(1) @binding(0) var<storage, read> H: array<f32>;
@group(1) @binding(1) var<storage, read> R: array<f32>;
@group(1) @binding(2) var<storage, read> delta_b: array<f32>;
@group(1) @binding(3) var<storage, read> delta_q: array<f32>;
@group(1) @binding(4) var<storage, read> sampling_matrix: array<f32>;
@group(1) @binding(5) var<storage, read> b0s_mask: array<i32>;

// Group 2: per-batch / output buffers
@group(2) @binding(0) var<storage, read_write> slineOutOff: array<i32>;
@group(2) @binding(1) var<storage, read_write> shDir0: array<f32>;
@group(2) @binding(2) var<storage, read_write> slineSeed: array<i32>;
@group(2) @binding(3) var<storage, read_write> slineLen: array<i32>;
@group(2) @binding(4) var<storage, read_write> sline: array<f32>;

// ── buffer-specific vec3 load/store helpers ─────────────────────────
// Same names as generate_streamlines.wgsl so tracking_helpers.wgsl
// functions (trilinear_interp_dataf, check_point_fn, peak_directions_fn,
// load_sphere_verts_f3) can call them.

fn load_seeds_f3(idx: u32) -> vec3<f32> {
    let base = idx * 3u;
    return vec3<f32>(seeds[base], seeds[base + 1u], seeds[base + 2u]);
}

fn load_sphere_verts_f3(idx: u32) -> vec3<f32> {
    let base = idx * 3u;
    return vec3<f32>(sphere_vertices[base], sphere_vertices[base + 1u], sphere_vertices[base + 2u]);
}

fn load_shDir0_f3(idx: u32) -> vec3<f32> {
    let base = idx * 3u;
    return vec3<f32>(shDir0[base], shDir0[base + 1u], shDir0[base + 2u]);
}

fn store_sline_f3(idx: u32, v: vec3<f32>) {
    let base = idx * 3u;
    sline[base] = v.x;
    sline[base + 1u] = v.y;
    sline[base + 2u] = v.z;
}

fn store_shDir0_f3(idx: u32, v: vec3<f32>) {
    let base = idx * 3u;
    shDir0[base] = v.x;
    shDir0[base + 1u] = v.y;
    shDir0[base + 2u] = v.z;
}

fn load_sline_f3(idx: u32) -> vec3<f32> {
    let base = idx * 3u;
    return vec3<f32>(sline[base], sline[base + 1u], sline[base + 2u]);
}

// ── workgroup memory ────────────────────────────────────────────────
// Boot pool: BLOCK_Y * sh_per_row where sh_per_row = 2*MAX_N32DIMT + 2*MAX_N32DIMT = 2048
// Total = 2 * 2048 = 4096
var<workgroup> wg_sh_mem: array<f32, 4096>;
// For peak_directions atomics
var<workgroup> wg_sh_ind: array<atomic<i32>, 1024>;
// BLOCK_Y * MAX_SLINES_PER_SEED * 3 = 2 * 10 * 3 = 60
var<workgroup> wg_dirs_sh: array<f32, 60>;
// For check_point (one per tidy row)
var<workgroup> wg_interp_out: array<f32, 2>;
// Scratch for closest_peak / new direction: BLOCK_Y * 3
var<workgroup> wg_new_dir: array<f32, 6>;
// Step counts per tidy row
var<workgroup> wg_stepsB: array<i32, 2>;
var<workgroup> wg_stepsF: array<i32, 2>;

// ── matrix access dispatch ──────────────────────────────────────────
// Replaces device pointer parameters; boot functions specify which
// matrix to read by ID.
const MAT_H: i32 = 0;
const MAT_R: i32 = 1;
const MAT_DELTA_B: i32 = 2;
const MAT_DELTA_Q: i32 = 3;
const MAT_SAMPLING: i32 = 4;

fn read_matrix(mat_id: i32, idx: i32) -> f32 {
    switch mat_id {
        case 0: { return H[idx]; }
        case 1: { return R[idx]; }
        case 2: { return delta_b[idx]; }
        case 3: { return delta_q[idx]; }
        case 4: { return sampling_matrix[idx]; }
        default: { return 0.0; }
    }
}

// ═══════════════════════════════════════════════════════════════════
// BOOT-SPECIFIC HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

// ── avgMask — subgroup-parallel masked average ──────────────────────
// Averages entries in wg_sh_mem[data_offset..] where b0s_mask[i] != 0.
fn avgMask(mskLen: i32, data_offset: u32, tidx: u32) -> f32 {
    var myCnt: i32 = 0;
    var mySum: f32 = 0.0;

    for (var i = i32(tidx); i < mskLen; i += i32(THR_X_SL)) {
        if (b0s_mask[i] != 0) {
            myCnt += 1;
            mySum += wg_sh_mem[data_offset + u32(i)];
        }
    }

    // Reduce across subgroup
    mySum += subgroupShuffleXor(mySum, 16u);
    mySum += subgroupShuffleXor(mySum, 8u);
    mySum += subgroupShuffleXor(mySum, 4u);
    mySum += subgroupShuffleXor(mySum, 2u);
    mySum += subgroupShuffleXor(mySum, 1u);

    var cnt_f = f32(myCnt);
    cnt_f += subgroupShuffleXor(cnt_f, 16u);
    cnt_f += subgroupShuffleXor(cnt_f, 8u);
    cnt_f += subgroupShuffleXor(cnt_f, 4u);
    cnt_f += subgroupShuffleXor(cnt_f, 2u);
    cnt_f += subgroupShuffleXor(cnt_f, 1u);

    return mySum / cnt_f;
}

// ── maskGet — compact non-masked entries ────────────────────────────
// Copies entries from wg_sh_mem[plain_offset..] where b0s_mask==0
// into wg_sh_mem[masked_offset..] in compacted order.
// Returns the number of compacted entries (hr_side).
fn maskGet(n: i32, plain_offset: u32, masked_offset: u32, tidx: u32) -> i32 {
    let laneMask = (1u << tidx) - 1u;

    var woff: i32 = 0;
    for (var j = 0; j < n; j += i32(THR_X_SL)) {
        var act: i32 = 0;
        if (j + i32(tidx) < n) {
            act = select(0, 1, b0s_mask[j + i32(tidx)] == 0);
        }

        let ballot = subgroupBallot(act != 0);
        let msk = ballot.x;

        let toff = i32(countOneBits(msk & laneMask));
        if (act != 0) {
            wg_sh_mem[masked_offset + u32(woff + toff)] =
                wg_sh_mem[plain_offset + u32(j) + tidx];
        }
        woff += i32(countOneBits(msk));
    }
    return woff;
}

// ── maskPut — scatter masked entries back ───────────────────────────
// Inverse of maskGet: scatters wg_sh_mem[masked_offset..] back into
// wg_sh_mem[plain_offset..] at positions where b0s_mask==0.
fn maskPut(n: i32, masked_offset: u32, plain_offset: u32, tidx: u32) {
    let laneMask = (1u << tidx) - 1u;

    var woff: i32 = 0;
    for (var j = 0; j < n; j += i32(THR_X_SL)) {
        var act: i32 = 0;
        if (j + i32(tidx) < n) {
            act = select(0, 1, b0s_mask[j + i32(tidx)] == 0);
        }

        let ballot = subgroupBallot(act != 0);
        let msk = ballot.x;

        let toff = i32(countOneBits(msk & laneMask));
        if (act != 0) {
            wg_sh_mem[plain_offset + u32(j) + tidx] =
                wg_sh_mem[masked_offset + u32(woff + toff)];
        }
        woff += i32(countOneBits(msk));
    }
}

// ── closest_peak_d — find closest peak to current direction ─────────
// Reads peaks from wg_dirs_sh[dirs_offset..] (as flat f32 triplets).
// Writes result to wg_new_dir[peak_offset..peak_offset+3].
// Returns 1 if a peak within max_angle was found, 0 otherwise.
fn closest_peak_d(
    max_angle: f32, direction: vec3<f32>,
    npeaks: i32, dirs_offset: u32, peak_offset: u32,
    tidx: u32
) -> i32 {
    let cos_similarity = cos(max_angle);

    var cpeak_dot: f32 = 0.0;
    var cpeak_idx: i32 = -1;

    for (var j = 0; j < npeaks; j += i32(THR_X_SL)) {
        if (j + i32(tidx) < npeaks) {
            let base = dirs_offset + u32(j + i32(tidx)) * 3u;
            let px = wg_dirs_sh[base];
            let py = wg_dirs_sh[base + 1u];
            let pz = wg_dirs_sh[base + 2u];

            let dot_val = direction.x * px + direction.y * py + direction.z * pz;

            if (abs(dot_val) > abs(cpeak_dot)) {
                cpeak_dot = dot_val;
                cpeak_idx = j + i32(tidx);
            }
        }
    }

    // Reduce across subgroup to find best peak
    for (var j = i32(THR_X_SL) / 2; j > 0; j /= 2) {
        let other_dot = subgroupShuffleXor(cpeak_dot, u32(j));
        let other_idx = subgroupShuffleXor(cpeak_idx, u32(j));
        if (abs(other_dot) > abs(cpeak_dot)) {
            cpeak_dot = other_dot;
            cpeak_idx = other_idx;
        }
    }

    if (cpeak_idx >= 0) {
        let base = dirs_offset + u32(cpeak_idx) * 3u;
        if (cpeak_dot >= cos_similarity) {
            wg_new_dir[peak_offset] = wg_dirs_sh[base];
            wg_new_dir[peak_offset + 1u] = wg_dirs_sh[base + 1u];
            wg_new_dir[peak_offset + 2u] = wg_dirs_sh[base + 2u];
            return 1;
        }
        if (cpeak_dot <= -cos_similarity) {
            wg_new_dir[peak_offset] = -wg_dirs_sh[base];
            wg_new_dir[peak_offset + 1u] = -wg_dirs_sh[base + 1u];
            wg_new_dir[peak_offset + 2u] = -wg_dirs_sh[base + 2u];
            return 1;
        }
    }
    return 0;
}

// ── ndotp_d — matrix-vector dot product ─────────────────────────────
// dstV[i] = sum_j( srcV[j] * matrix[i*M+j] )  for i in [0..N)
// Source vector in wg_sh_mem[srcV_off..], destination in wg_sh_mem[dstV_off..].
// Matrix accessed via read_matrix(mat_id, ...).
fn ndotp_d(N: i32, M: i32, srcV_off: u32, mat_id: i32, dstV_off: u32, tidx: u32) {
    for (var i = 0; i < N; i++) {
        var tmp: f32 = 0.0;

        for (var j = 0; j < M; j += i32(THR_X_SL)) {
            if (j + i32(tidx) < M) {
                tmp += wg_sh_mem[srcV_off + u32(j) + tidx] *
                       read_matrix(mat_id, i * M + j + i32(tidx));
            }
        }
        // Reduce across subgroup using shuffle down
        tmp += subgroupShuffleDown(tmp, 16u);
        tmp += subgroupShuffleDown(tmp, 8u);
        tmp += subgroupShuffleDown(tmp, 4u);
        tmp += subgroupShuffleDown(tmp, 2u);
        tmp += subgroupShuffleDown(tmp, 1u);

        if (tidx == 0u) {
            wg_sh_mem[dstV_off + u32(i)] = tmp;
        }
    }
}

// ── ndotp_log_opdt_d — OPDT log-weighted dot product ────────────────
// dstV[i] = sum_j( -log(v) * (1.5 + log(v)) * v * matrix[i*M+j] )
fn ndotp_log_opdt_d(N: i32, M: i32, srcV_off: u32, mat_id: i32, dstV_off: u32, tidx: u32) {
    let ONEP5: f32 = 1.5;

    for (var i = 0; i < N; i++) {
        var tmp: f32 = 0.0;

        for (var j = 0; j < M; j += i32(THR_X_SL)) {
            if (j + i32(tidx) < M) {
                let v = wg_sh_mem[srcV_off + u32(j) + tidx];
                let lv = log(v);
                tmp += -lv * (ONEP5 + lv) * v *
                       read_matrix(mat_id, i * M + j + i32(tidx));
            }
        }
        tmp += subgroupShuffleDown(tmp, 16u);
        tmp += subgroupShuffleDown(tmp, 8u);
        tmp += subgroupShuffleDown(tmp, 4u);
        tmp += subgroupShuffleDown(tmp, 2u);
        tmp += subgroupShuffleDown(tmp, 1u);

        if (tidx == 0u) {
            wg_sh_mem[dstV_off + u32(i)] = tmp;
        }
    }
}

// ── ndotp_log_csa_d — CSA log-log-weighted dot product ──────────────
// dstV[i] = sum_j( log(-log(clamp(v))) * matrix[i*M+j] )
fn ndotp_log_csa_d(N: i32, M: i32, srcV_off: u32, mat_id: i32, dstV_off: u32, tidx: u32) {
    let csa_min: f32 = 0.001;
    let csa_max: f32 = 0.999;

    for (var i = 0; i < N; i++) {
        var tmp: f32 = 0.0;

        for (var j = 0; j < M; j += i32(THR_X_SL)) {
            if (j + i32(tidx) < M) {
                let v = clamp(wg_sh_mem[srcV_off + u32(j) + tidx], csa_min, csa_max);
                tmp += log(-log(v)) *
                       read_matrix(mat_id, i * M + j + i32(tidx));
            }
        }
        tmp += subgroupShuffleDown(tmp, 16u);
        tmp += subgroupShuffleDown(tmp, 8u);
        tmp += subgroupShuffleDown(tmp, 4u);
        tmp += subgroupShuffleDown(tmp, 2u);
        tmp += subgroupShuffleDown(tmp, 1u);

        if (tidx == 0u) {
            wg_sh_mem[dstV_off + u32(i)] = tmp;
        }
    }
}

// ── fit_opdt — OPDT model fitting ───────────────────────────────────
// r_sh <- delta_q . msk_data (log-opdt weighted)
// h_sh <- delta_b . msk_data (plain)
// r_sh -= h_sh
fn fit_opdt(delta_nr: i32, hr_side: i32,
            msk_data_off: u32, h_off: u32, r_off: u32, tidx: u32) {
    ndotp_log_opdt_d(delta_nr, hr_side, msk_data_off, MAT_DELTA_Q, r_off, tidx);
    ndotp_d(delta_nr, hr_side, msk_data_off, MAT_DELTA_B, h_off, tidx);
    subgroupBarrier();
    for (var j = i32(tidx); j < delta_nr; j += i32(THR_X_SL)) {
        wg_sh_mem[r_off + u32(j)] -= wg_sh_mem[h_off + u32(j)];
    }
    subgroupBarrier();
}

// ── fit_csa — CSA model fitting ─────────────────────────────────────
// r_sh <- delta_q . msk_data (log-log weighted)
// r_sh[0] = n0_const
fn fit_csa(delta_nr: i32, hr_side: i32,
           msk_data_off: u32, r_off: u32, tidx: u32) {
    let n0_const: f32 = 0.28209479177387814;
    ndotp_log_csa_d(delta_nr, hr_side, msk_data_off, MAT_DELTA_Q, r_off, tidx);
    subgroupBarrier();
    if (tidx == 0u) {
        wg_sh_mem[r_off] = n0_const;
    }
    subgroupBarrier();
}

// ── fit_model_coef — dispatch to OPDT or CSA ────────────────────────
fn fit_model_coef(model_type: i32, delta_nr: i32, hr_side: i32,
                  msk_data_off: u32, h_off: u32, r_off: u32, tidx: u32) {
    switch model_type {
        case 0: /* MODEL_OPDT */ {
            fit_opdt(delta_nr, hr_side, msk_data_off, h_off, r_off, tidx);
        }
        case 1: /* MODEL_CSA */ {
            fit_csa(delta_nr, hr_side, msk_data_off, r_off, tidx);
        }
        default: {}
    }
}

// ═══════════════════════════════════════════════════════════════════
// BOOTSTRAP DIRECTION GETTER
// ═══════════════════════════════════════════════════════════════════

struct GetDirBootResult {
    ndir: i32,
    state: PhiloxState,
}

fn get_direction_boot(
    st: PhiloxState,
    nattempts: i32,
    model_type: i32,
    max_angle: f32,
    min_signal: f32,
    relative_peak_thres: f32,
    min_separation_angle: f32,
    dir: vec3<f32>,
    dimx: i32, dimy: i32, dimz: i32, dimt: i32,
    point: vec3<f32>,
    delta_nr: i32,
    samplm_nr: i32,
    num_edges: i32,
    dirs_offset: u32,   // into wg_dirs_sh (tidy * MAX_SLINES_PER_SEED * 3)
    sh_offset: u32,     // into wg_sh_mem (tidy * sh_per_row = vox_data_off)
    scratch_offset: u32, // into wg_new_dir (tidy * 3)
    ind_offset: u32,    // into wg_sh_ind (tidy * MAX_N32DIMT)
    tidx: u32, tidy: u32
) -> GetDirBootResult {
    var rng = st;

    let n32dimt = u32(((dimt + 31) / 32) * 32);

    // Partition shared memory within the per-tidy row
    let vox_data_off = sh_offset;
    let msk_data_off = vox_data_off + n32dimt;
    let r_off = msk_data_off + n32dimt;
    let h_off = r_off + max(n32dimt, u32(samplm_nr));

    // Compute hr_side (number of non-b0 volumes)
    var hr_side: i32 = 0;
    for (var j = i32(tidx); j < dimt; j += i32(THR_X_SL)) {
        if (b0s_mask[j] == 0) {
            hr_side += 1;
        }
    }
    hr_side += subgroupShuffleXor(hr_side, 16u);
    hr_side += subgroupShuffleXor(hr_side, 8u);
    hr_side += subgroupShuffleXor(hr_side, 4u);
    hr_side += subgroupShuffleXor(hr_side, 2u);
    hr_side += subgroupShuffleXor(hr_side, 1u);

    for (var attempt = 0; attempt < nattempts; attempt++) {

        // Trilinear interpolation of dataf at point -> wg_sh_mem[vox_data_off..]
        let rv = trilinear_interp_dataf(dimx, dimy, dimz, dimt, point, vox_data_off, tidx);

        // maskGet: compact non-b0 entries from vox_data -> msk_data
        maskGet(dimt, vox_data_off, msk_data_off, tidx);

        subgroupBarrier();

        if (rv == 0) {
            // Multiply masked data by R and H matrices
            ndotp_d(hr_side, hr_side, msk_data_off, MAT_R, r_off, tidx);
            ndotp_d(hr_side, hr_side, msk_data_off, MAT_H, h_off, tidx);

            subgroupBarrier();

            // Bootstrap: add permuted residuals
            for (var j = 0; j < hr_side; j += i32(THR_X_SL)) {
                if (j + i32(tidx) < hr_side) {
                    let pr = philox_uint(rng);
                    rng = pr.state;
                    let srcPermInd = i32(pr.value % u32(hr_side));
                    wg_sh_mem[h_off + u32(j) + tidx] += wg_sh_mem[r_off + u32(srcPermInd)];
                }
            }
            subgroupBarrier();

            // Scatter back: vox_data[dwi_mask] = masked_data
            maskPut(dimt, h_off, vox_data_off, tidx);
            subgroupBarrier();

            // Clamp to min_signal
            for (var j = i32(tidx); j < dimt; j += i32(THR_X_SL)) {
                wg_sh_mem[vox_data_off + u32(j)] = max(min_signal, wg_sh_mem[vox_data_off + u32(j)]);
            }
            subgroupBarrier();

            // Normalize by b0 average
            let denom = avgMask(dimt, vox_data_off, tidx);

            for (var j = i32(tidx); j < dimt; j += i32(THR_X_SL)) {
                wg_sh_mem[vox_data_off + u32(j)] /= denom;
            }
            subgroupBarrier();

            // Re-compact after normalization
            maskGet(dimt, vox_data_off, msk_data_off, tidx);
            subgroupBarrier();

            // Fit model coefficients
            fit_model_coef(model_type, delta_nr, hr_side,
                           msk_data_off, h_off, r_off, tidx);

            // Compute PMF: sampling_matrix * coef -> h_sh
            // r_off holds the coefficients after fitting
            ndotp_d(samplm_nr, delta_nr, r_off, MAT_SAMPLING, h_off, tidx);

            // h_off now holds PMF
        } else {
            // Outside image: zero PMF
            for (var j = i32(tidx); j < samplm_nr; j += i32(THR_X_SL)) {
                wg_sh_mem[h_off + u32(j)] = 0.0;
            }
        }
        subgroupBarrier();

        // Absolute PMF threshold
        let abs_pmf_thr = PMF_THRESHOLD_P *
            sg_max_reduce_wg(samplm_nr, h_off, REAL_MIN, tidx);
        subgroupBarrier();

        // Zero entries below threshold
        for (var j = i32(tidx); j < samplm_nr; j += i32(THR_X_SL)) {
            if (wg_sh_mem[h_off + u32(j)] < abs_pmf_thr) {
                wg_sh_mem[h_off + u32(j)] = 0.0;
            }
        }
        subgroupBarrier();

        // Find peak directions
        let ndir = peak_directions_fn(
            h_off, dirs_offset, ind_offset,
            num_edges, samplm_nr,
            relative_peak_thres, min_separation_angle,
            tidx);

        if (nattempts == 1) {
            // init=True: return number of initial directions
            return GetDirBootResult(ndir, rng);
        } else {
            // init=False: find closest peak to current direction
            if (ndir > 0) {
                let foundPeak = closest_peak_d(
                    max_angle, dir, ndir, dirs_offset, scratch_offset, tidx);
                subgroupBarrier();
                if (foundPeak != 0) {
                    // Copy result from scratch to dirs[0]
                    if (tidx == 0u) {
                        wg_dirs_sh[dirs_offset] = wg_new_dir[scratch_offset];
                        wg_dirs_sh[dirs_offset + 1u] = wg_new_dir[scratch_offset + 1u];
                        wg_dirs_sh[dirs_offset + 2u] = wg_new_dir[scratch_offset + 2u];
                    }
                    return GetDirBootResult(1, rng);
                }
            }
        }
    }
    return GetDirBootResult(0, rng);
}

// ═══════════════════════════════════════════════════════════════════
// TRACKER — step along streamline in one direction
// ═══════════════════════════════════════════════════════════════════

struct TrackerBootResult {
    tissue_class: i32,
    state: PhiloxState,
}

fn tracker_boot(
    st: PhiloxState,
    model_type: i32,
    max_angle: f32,
    tc_threshold: f32,
    step_size: f32,
    relative_peak_thres: f32,
    min_separation_angle: f32,
    min_signal: f32,
    seed: vec3<f32>,
    first_step: vec3<f32>,
    dimx: i32, dimy: i32, dimz: i32, dimt: i32,
    samplm_nr: i32,
    num_edges: i32,
    delta_nr: i32,
    nsteps_idx: u32,    // index into wg_stepsB/wg_stepsF
    sline_base: u32,    // base flat f32 index into sline buffer
    dirs_offset: u32,   // into wg_dirs_sh
    sh_offset: u32,     // into wg_sh_mem
    scratch_offset: u32, // into wg_new_dir
    ind_offset: u32,    // into wg_sh_ind
    tidx: u32, tidy: u32,
    use_stepsB: bool
) -> TrackerBootResult {
    var rng = st;
    var tissue_class: i32 = TRACKPOINT;

    var point = seed;
    var direction = first_step;

    // Store initial point
    if (tidx == 0u) {
        store_sline_f3(sline_base, point);
    }
    subgroupBarrier();

    var i: i32 = 1;
    for (; i < MAX_SLINE_LEN; i++) {
        let gdr = get_direction_boot(
            rng,
            5,  // NATTEMPTS
            model_type,
            max_angle,
            min_signal,
            relative_peak_thres,
            min_separation_angle,
            direction,
            dimx, dimy, dimz, dimt,
            point,
            delta_nr, samplm_nr, num_edges,
            dirs_offset, sh_offset, scratch_offset, ind_offset,
            tidx, tidy);
        rng = gdr.state;
        subgroupBarrier();

        // Read direction from scratch (closest_peak wrote it there)
        direction = vec3<f32>(
            wg_new_dir[scratch_offset],
            wg_new_dir[scratch_offset + 1u],
            wg_new_dir[scratch_offset + 2u]);
        subgroupBarrier();

        if (gdr.ndir == 0) {
            break;
        }

        // Advance point (voxel_size is 1.0 for boot)
        point.x += direction.x * step_size;
        point.y += direction.y * step_size;
        point.z += direction.z * step_size;

        if (tidx == 0u) {
            store_sline_f3(sline_base + u32(i), point);
        }
        subgroupBarrier();

        tissue_class = check_point_fn(
            tc_threshold, point, dimx, dimy, dimz, tidx, tidy);

        if (tissue_class == ENDPOINT ||
            tissue_class == INVALIDPOINT ||
            tissue_class == OUTSIDEIMAGE) {
            break;
        }
    }

    if (use_stepsB) {
        wg_stepsB[nsteps_idx] = i;
    } else {
        wg_stepsF[nsteps_idx] = i;
    }
    return TrackerBootResult(tissue_class, rng);
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL ENTRY POINTS
// ═══════════════════════════════════════════════════════════════════

// ── getNumStreamlinesBoot_k — count streamlines per seed ────────────
@compute @workgroup_size(32, 2, 1)
fn getNumStreamlinesBoot_k(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) gid: vec3<u32>
) {
    let tidx = tid.x;
    let tidy = tid.y;
    let slid = gid.x * BLOCK_Y + tidy;

    if (i32(slid) >= params.nseed) { return; }

    let global_id = gid.x * BLOCK_Y * THR_X_SL + THR_X_SL * tidy + tidx;
    var st = philox_init(
        u32(params.rng_seed_lo), u32(params.rng_seed_hi), global_id, 0u);

    let n32dimt = u32(((params.dimt + 31) / 32) * 32);
    let sh_per_row = 2u * n32dimt + 2u * max(n32dimt, u32(params.samplm_nr));

    let sh_offset = tidy * sh_per_row;
    let dirs_offset = tidy * MAX_SLINES_PER_SEED * 3u;
    let scratch_offset = tidy * 3u;
    let ind_offset = tidy * max(n32dimt, u32(params.samplm_nr));

    let seed = load_seeds_f3(slid);

    var ndir: i32 = 0;
    switch params.model_type {
        case 0, 1: /* MODEL_OPDT, MODEL_CSA */ {
            let gdr = get_direction_boot(
                st,
                1,  // NATTEMPTS=1 (init=True)
                params.model_type,
                params.max_angle,
                params.min_signal,
                params.relative_peak_thresh,
                params.min_separation_angle,
                vec3<f32>(0.0, 0.0, 0.0),
                params.dimx, params.dimy, params.dimz, params.dimt,
                seed,
                params.delta_nr,
                params.samplm_nr,
                params.num_edges,
                dirs_offset, sh_offset, scratch_offset, ind_offset,
                tidx, tidy);
            ndir = gdr.ndir;
        }
        default: {
            ndir = 0;
        }
    }

    // Copy found directions to global output buffer
    for (var j = i32(tidx); j < ndir; j += i32(THR_X_SL)) {
        let src_base = dirs_offset + u32(j) * 3u;
        let dst_idx = (slid * u32(params.samplm_nr) + u32(j)) * 3u;
        shDir0[dst_idx] = wg_dirs_sh[src_base];
        shDir0[dst_idx + 1u] = wg_dirs_sh[src_base + 1u];
        shDir0[dst_idx + 2u] = wg_dirs_sh[src_base + 2u];
    }

    if (tidx == 0u) {
        slineOutOff[slid] = ndir;
    }
}

// ── genStreamlinesMergeBoot_k — main bootstrap streamline kernel ────
@compute @workgroup_size(32, 2, 1)
fn genStreamlinesMergeBoot_k(
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

    let seed = load_seeds_f3(slid);

    let ndir = slineOutOff[slid + 1u] - slineOutOff[slid];

    subgroupBarrier();

    var sline_off = slineOutOff[slid];

    let n32dimt = u32(((params.dimt + 31) / 32) * 32);
    let sh_per_row = 2u * n32dimt + 2u * max(n32dimt, u32(params.samplm_nr));

    let sh_offset = tidy * sh_per_row;
    let dirs_offset = tidy * MAX_SLINES_PER_SEED * 3u;
    let scratch_offset = tidy * 3u;
    let ind_offset = tidy * max(n32dimt, u32(params.samplm_nr));

    for (var i = 0; i < ndir; i++) {
        let dir_idx = slid * u32(params.samplm_nr) + u32(i);
        let first_step = load_shDir0_f3(dir_idx);

        // Flat f32 base for this streamline's sline storage
        let sline_base_f3 = u32(sline_off) * u32(MAX_SLINE_LEN) * 2u;

        if (tidx == 0u) {
            slineSeed[sline_off] = i32(slid);
        }

        // ── Track backward ──
        let trB = tracker_boot(
            st,
            params.model_type,
            params.max_angle,
            params.tc_threshold,
            params.step_size,
            params.relative_peak_thresh,
            params.min_separation_angle,
            params.min_signal,
            seed,
            vec3<f32>(-first_step.x, -first_step.y, -first_step.z),
            params.dimx, params.dimy, params.dimz, params.dimt,
            params.samplm_nr, params.num_edges, params.delta_nr,
            tidy, sline_base_f3,
            dirs_offset, sh_offset, scratch_offset, ind_offset,
            tidx, tidy, true);
        st = trB.state;

        let stepsB = wg_stepsB[tidy];

        // ── Reverse backward streamline ──
        for (var j = i32(tidx); j < stepsB / 2; j += i32(THR_X_SL)) {
            let a_idx = sline_base_f3 + u32(j);
            let b_idx = sline_base_f3 + u32(stepsB - 1 - j);
            let pa = load_sline_f3(a_idx);
            let pb = load_sline_f3(b_idx);
            store_sline_f3(a_idx, pb);
            store_sline_f3(b_idx, pa);
        }

        // ── Track forward ──
        let fwd_base = sline_base_f3 + u32(stepsB - 1);
        let trF = tracker_boot(
            st,
            params.model_type,
            params.max_angle,
            params.tc_threshold,
            params.step_size,
            params.relative_peak_thresh,
            params.min_separation_angle,
            params.min_signal,
            seed,
            first_step,
            params.dimx, params.dimy, params.dimz, params.dimt,
            params.samplm_nr, params.num_edges, params.delta_nr,
            tidy, fwd_base,
            dirs_offset, sh_offset, scratch_offset, ind_offset,
            tidx, tidy, false);
        st = trF.state;

        if (tidx == 0u) {
            slineLen[sline_off] = stepsB - 1 + wg_stepsF[tidy];
        }

        sline_off += 1;
    }
}
