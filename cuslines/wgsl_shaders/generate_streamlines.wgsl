// generate_streamlines.wgsl — Probabilistic streamline generation kernels.
// Mirrors cuslines/metal_shaders/generate_streamlines_metal.metal.
//
// Contains buffer binding declarations, workgroup memory, the probabilistic
// direction getter function, and kernel entry points.

// ── parameter struct ────────────────────────────────────────────────
struct ProbTrackingParams {
    max_angle: f32,
    tc_threshold: f32,
    step_size: f32,
    relative_peak_thresh: f32,
    min_separation_angle: f32,
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
    model_type: i32,
}

// ── buffer bindings ─────────────────────────────────────────────────
// Group 0: common data (used by both getNum and gen)
@group(0) @binding(0) var<storage, read> params: ProbTrackingParams;
@group(0) @binding(1) var<storage, read> seeds: array<f32>;
@group(0) @binding(2) var<storage, read> dataf: array<f32>;
@group(0) @binding(3) var<storage, read> metric_map: array<f32>;
@group(0) @binding(4) var<storage, read> sphere_vertices: array<f32>;
@group(0) @binding(5) var<storage, read> sphere_edges: array<i32>;

// Group 1: per-batch / output buffers
@group(1) @binding(0) var<storage, read_write> slineOutOff: array<i32>;
@group(1) @binding(1) var<storage, read_write> shDir0: array<f32>;
@group(1) @binding(2) var<storage, read_write> slineSeed: array<i32>;
@group(1) @binding(3) var<storage, read_write> slineLen: array<i32>;
@group(1) @binding(4) var<storage, read_write> sline: array<f32>;

// ── buffer-specific vec3 load/store helpers ─────────────────────────
// WGSL does not allow ptr<storage> as function parameters, so we define
// buffer-specific helpers that access module-scope variables directly.

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

// ── workgroup memory ────────────────────────────────────────────────
// BLOCK_Y * MAX_N32DIMT = 2 * 512 = 1024 elements
var<workgroup> wg_sh_mem: array<f32, 1024>;
var<workgroup> wg_sh_ind: array<atomic<i32>, 1024>;
// BLOCK_Y * MAX_SLINES_PER_SEED * 3 = 2 * 10 * 3 = 60
var<workgroup> wg_dirs_sh: array<f32, 60>;
// BLOCK_Y small arrays
var<workgroup> wg_interp_out: array<f32, 2>;
var<workgroup> wg_new_dir: array<f32, 6>;   // BLOCK_Y * 3
var<workgroup> wg_stepsB: array<i32, 2>;
var<workgroup> wg_stepsF: array<i32, 2>;

// ── probabilistic direction getter ──────────────────────────────────

struct GetDirProbResult {
    ndir: i32,
    state: PhiloxState,
}

fn get_direction_prob(
    st: PhiloxState,
    max_angle: f32, relative_peak_thres: f32,
    min_separation_angle: f32, dir: vec3<f32>,
    dimx: i32, dimy: i32, dimz: i32, dimt: i32,
    point: vec3<f32>, num_edges: i32,
    dirs_offset: u32, sh_offset: u32, ind_offset: u32,
    is_start: bool, tidx: u32, tidy: u32
) -> GetDirProbResult {
    var rng = st;

    // Trilinear interpolation of PMF at point → wg_sh_mem[sh_offset..]
    subgroupBarrier();
    let rv = trilinear_interp_dataf(dimx, dimy, dimz, dimt, point, sh_offset, tidx);
    subgroupBarrier();
    if (rv != 0) {
        return GetDirProbResult(0, rng);
    }

    // Absolute PMF threshold
    let absol_thresh = PMF_THRESHOLD_P * sg_max_reduce_wg(dimt, sh_offset, REAL_MIN, tidx);
    subgroupBarrier();

    // Zero out entries below threshold
    for (var i = i32(tidx); i < dimt; i += i32(THR_X_SL)) {
        if (wg_sh_mem[sh_offset + u32(i)] < absol_thresh) {
            wg_sh_mem[sh_offset + u32(i)] = 0.0;
        }
    }
    subgroupBarrier();

    if (is_start) {
        let ndir = peak_directions_fn(
            sh_offset, dirs_offset, ind_offset,
            num_edges, dimt,
            relative_peak_thres, min_separation_angle, tidx);
        return GetDirProbResult(ndir, rng);
    } else {
        // Filter by angle similarity
        let cos_similarity = cos(max_angle);

        for (var i = i32(tidx); i < dimt; i += i32(THR_X_SL)) {
            let sv = load_sphere_verts_f3(u32(i));
            let dot_val = dir.x * sv.x + dir.y * sv.y + dir.z * sv.z;
            if (abs(dot_val) < cos_similarity) {
                wg_sh_mem[sh_offset + u32(i)] = 0.0;
            }
        }
        subgroupBarrier();

        // Prefix sum for CDF
        prefix_sum_sh(sh_offset, dimt, tidx);

        let last_cdf = wg_sh_mem[sh_offset + u32(dimt - 1)];
        if (last_cdf == 0.0) {
            return GetDirProbResult(0, rng);
        }

        // Sample from CDF (lane 0 draws random, broadcast to all)
        var selected_cdf: f32 = 0.0;
        if (tidx == 0u) {
            let ur = philox_uniform(rng);
            rng = ur.state;
            selected_cdf = ur.value * last_cdf;
        }
        selected_cdf = subgroupBroadcastFirst(selected_cdf);

        // Also broadcast updated RNG state from lane 0
        // (only lane 0 consumed a random number)
        // Note: PhiloxState can't be shuffled directly; lane 0 holds the
        // authoritative state. Other lanes' rng variable is stale but
        // they don't use it for Prob tracking.

        // Binary search + ballot for insertion point
        var low: i32 = 0;
        var high: i32 = dimt - 1;
        while ((high - low) >= i32(THR_X_SL)) {
            let mid = (low + high) / 2;
            if (wg_sh_mem[sh_offset + u32(mid)] < selected_cdf) {
                low = mid;
            } else {
                high = mid;
            }
        }

        var ballot_pred = false;
        if (low + i32(tidx) <= high) {
            ballot_pred = selected_cdf < wg_sh_mem[sh_offset + u32(low) + tidx];
        }
        let ballot = subgroupBallot(ballot_pred);
        let msk = ballot.x;
        var ind_prob: i32;
        if (msk != 0u) {
            ind_prob = low + i32(countTrailingZeros(msk));
        } else {
            ind_prob = dimt - 1;
        }

        // Select direction, flip if needed
        if (tidx == 0u) {
            let sv = load_sphere_verts_f3(u32(ind_prob));
            let dot_val = dir.x * sv.x + dir.y * sv.y + dir.z * sv.z;
            if (dot_val > 0.0) {
                wg_dirs_sh[dirs_offset] = sv.x;
                wg_dirs_sh[dirs_offset + 1u] = sv.y;
                wg_dirs_sh[dirs_offset + 2u] = sv.z;
            } else {
                wg_dirs_sh[dirs_offset] = -sv.x;
                wg_dirs_sh[dirs_offset + 1u] = -sv.y;
                wg_dirs_sh[dirs_offset + 2u] = -sv.z;
            }
        }

        return GetDirProbResult(1, rng);
    }
}

// ── tracker — step along streamline ─────────────────────────────────
struct TrackerResult {
    tissue_class: i32,
    state: PhiloxState,
}

fn tracker_prob_fn(
    st: PhiloxState,
    max_angle: f32, tc_threshold: f32, step_size: f32,
    relative_peak_thres: f32, min_separation_angle: f32,
    seed: vec3<f32>, first_step: vec3<f32>,
    dimx: i32, dimy: i32, dimz: i32, dimt: i32,
    num_edges: i32,
    nsteps_idx: u32,  // index into wg_stepsB or wg_stepsF
    sline_base: u32,  // base index in sline buffer (flat f32 triplets)
    new_dir_offset: u32,  // into wg_new_dir (tidy * 3)
    sh_offset: u32, ind_offset: u32,
    tidx: u32, tidy: u32, use_stepsB: bool
) -> TrackerResult {
    var rng = st;
    var tissue_class: i32 = TRACKPOINT;
    var point = seed;
    var direction = first_step;

    if (tidx == 0u) {
        sline[sline_base] = point.x;
        sline[sline_base + 1u] = point.y;
        sline[sline_base + 2u] = point.z;
    }
    subgroupBarrier();

    var i: i32 = 1;
    for (; i < MAX_SLINE_LEN; i++) {
        let gdr = get_direction_prob(
            rng, max_angle, relative_peak_thres, min_separation_angle,
            direction, dimx, dimy, dimz, dimt, point, num_edges,
            new_dir_offset, sh_offset, ind_offset,
            false, tidx, tidy);
        rng = gdr.state;
        subgroupBarrier();

        direction = vec3<f32>(
            wg_dirs_sh[new_dir_offset],
            wg_dirs_sh[new_dir_offset + 1u],
            wg_dirs_sh[new_dir_offset + 2u]);
        subgroupBarrier();

        if (gdr.ndir == 0) { break; }

        point.x += direction.x * step_size;
        point.y += direction.y * step_size;
        point.z += direction.z * step_size;

        if (tidx == 0u) {
            let off = sline_base + u32(i) * 3u;
            sline[off] = point.x;
            sline[off + 1u] = point.y;
            sline[off + 2u] = point.z;
        }
        subgroupBarrier();

        tissue_class = check_point_fn(tc_threshold, point, dimx, dimy, dimz, tidx, tidy);

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
    return TrackerResult(tissue_class, rng);
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL ENTRY POINTS
// ═══════════════════════════════════════════════════════════════════

// ── getNumStreamlinesProb_k ─────────────────────────────────────────
@compute @workgroup_size(32, 2, 1)
fn getNumStreamlinesProb_k(
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

    let sh_offset = tidy * n32dimt;
    let ind_offset = tidy * n32dimt;
    let dirs_offset = tidy * MAX_SLINES_PER_SEED * 3u;

    let seed = load_seeds_f3(slid);

    let gdr = get_direction_prob(
        st, params.max_angle, params.relative_peak_thresh,
        params.min_separation_angle, vec3<f32>(0.0, 0.0, 0.0),
        params.dimx, params.dimy, params.dimz, params.dimt,
        seed, params.num_edges,
        dirs_offset, sh_offset, ind_offset,
        true, tidx, tidy);

    // Copy found directions to global memory
    if (tidx == 0u) {
        let my_shDir_base = slid * u32(params.samplm_nr);
        for (var d = 0; d < gdr.ndir; d++) {
            let src = dirs_offset + u32(d) * 3u;
            let dst = (my_shDir_base + u32(d)) * 3u;
            shDir0[dst] = wg_dirs_sh[src];
            shDir0[dst + 1u] = wg_dirs_sh[src + 1u];
            shDir0[dst + 2u] = wg_dirs_sh[src + 2u];
        }
        slineOutOff[slid] = gdr.ndir;
    }
}

// ── genStreamlinesMergeProb_k ───────────────────────────────────────
@compute @workgroup_size(32, 2, 1)
fn genStreamlinesMergeProb_k(
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

    let n32dimt = u32(((params.dimt + 31) / 32) * 32);
    let sh_offset = tidy * n32dimt;
    let ind_offset = tidy * n32dimt;
    let new_dir_offset = tidy * 3u;

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

        // Backward tracking (negated first step)
        let neg_step = vec3<f32>(-first_step.x, -first_step.y, -first_step.z);
        let trB = tracker_prob_fn(
            st, params.max_angle, params.tc_threshold, params.step_size,
            params.relative_peak_thresh, params.min_separation_angle,
            seed, neg_step,
            params.dimx, params.dimy, params.dimz, params.dimt,
            params.num_edges, tidy, sline_base, new_dir_offset,
            sh_offset, ind_offset, tidx, tidy, true);
        st = trB.state;

        let stepsB = wg_stepsB[tidy];

        // Reverse backward streamline
        for (var j = i32(tidx); j < stepsB / 2; j += i32(THR_X_SL)) {
            let a_off = sline_base + u32(j) * 3u;
            let b_off = sline_base + u32(stepsB - 1 - j) * 3u;
            let pa = vec3<f32>(sline[a_off], sline[a_off + 1u], sline[a_off + 2u]);
            let pb = vec3<f32>(sline[b_off], sline[b_off + 1u], sline[b_off + 2u]);
            sline[a_off] = pb.x; sline[a_off + 1u] = pb.y; sline[a_off + 2u] = pb.z;
            sline[b_off] = pa.x; sline[b_off + 1u] = pa.y; sline[b_off + 2u] = pa.z;
        }

        // Forward tracking
        let fwd_base = sline_base + u32(stepsB - 1) * 3u;
        let trF = tracker_prob_fn(
            st, params.max_angle, params.tc_threshold, params.step_size,
            params.relative_peak_thresh, params.min_separation_angle,
            seed, first_step,
            params.dimx, params.dimy, params.dimz, params.dimt,
            params.num_edges, tidy, fwd_base, new_dir_offset,
            sh_offset, ind_offset, tidx, tidy, false);
        st = trF.state;

        if (tidx == 0u) {
            slineLen[sline_off] = stepsB - 1 + wg_stepsF[tidy];
        }

        sline_off += 1;
    }
}
