// tracking_helpers.wgsl — Trilinear interpolation, tissue checking, peak direction finding.
// Mirrors cuslines/metal_shaders/tracking_helpers.metal.
//
// Functions access module-scope storage buffers (dataf, metric_map, sphere_vertices,
// sphere_edges) and workgroup arrays (wg_sh_mem, wg_sh_ind, wg_dirs_sh) directly.
// Offset parameters select subarrays within workgroup memory.

// ── trilinear interpolation (inner loop for one channel) ────────────
fn interpolation_helper_dataf(
    wgh: array<array<f32, 2>, 3>,
    coo: array<array<i32, 2>, 3>,
    dimy: i32, dimz: i32, dimt: i32, t: i32
) -> f32 {
    var tmp: f32 = 0.0;
    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            for (var k = 0; k < 2; k++) {
                let idx = coo[0][i] * dimy * dimz * dimt +
                          coo[1][j] * dimz * dimt +
                          coo[2][k] * dimt + t;
                tmp += wgh[0][i] * wgh[1][j] * wgh[2][k] * dataf[idx];
            }
        }
    }
    return tmp;
}

fn interpolation_helper_metric(
    wgh: array<array<f32, 2>, 3>,
    coo: array<array<i32, 2>, 3>,
    dimy: i32, dimz: i32
) -> f32 {
    var tmp: f32 = 0.0;
    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            for (var k = 0; k < 2; k++) {
                let idx = coo[0][i] * dimy * dimz + coo[1][j] * dimz + coo[2][k];
                tmp += wgh[0][i] * wgh[1][j] * wgh[2][k] * metric_map[idx];
            }
        }
    }
    return tmp;
}

// Compute trilinear weights and coordinates from a point.
// Returns -1 if outside image, 0 otherwise.
struct TrilinearSetup {
    status: i32,
    wgh: array<array<f32, 2>, 3>,
    coo: array<array<i32, 2>, 3>,
}

fn trilinear_setup(
    dimx: i32, dimy: i32, dimz: i32, point: vec3<f32>
) -> TrilinearSetup {
    let HALF: f32 = 0.5;
    var r: TrilinearSetup;

    if (point.x < -HALF || point.x + HALF >= f32(dimx) ||
        point.y < -HALF || point.y + HALF >= f32(dimy) ||
        point.z < -HALF || point.z + HALF >= f32(dimz)) {
        r.status = -1;
        return r;
    }

    let fl = floor(point);

    r.wgh[0][1] = point.x - fl.x;
    r.wgh[0][0] = 1.0 - r.wgh[0][1];
    r.coo[0][0] = max(0, i32(fl.x));
    r.coo[0][1] = min(dimx - 1, r.coo[0][0] + 1);

    r.wgh[1][1] = point.y - fl.y;
    r.wgh[1][0] = 1.0 - r.wgh[1][1];
    r.coo[1][0] = max(0, i32(fl.y));
    r.coo[1][1] = min(dimy - 1, r.coo[1][0] + 1);

    r.wgh[2][1] = point.z - fl.z;
    r.wgh[2][0] = 1.0 - r.wgh[2][1];
    r.coo[2][0] = max(0, i32(fl.z));
    r.coo[2][1] = min(dimz - 1, r.coo[2][0] + 1);

    r.status = 0;
    return r;
}

// ── trilinear interp: multi-channel from dataf → wg_sh_mem ─────────
fn trilinear_interp_dataf(
    dimx: i32, dimy: i32, dimz: i32, dimt: i32,
    point: vec3<f32>, wg_offset: u32, tidx: u32
) -> i32 {
    let setup = trilinear_setup(dimx, dimy, dimz, point);
    if (setup.status != 0) { return -1; }

    for (var t = i32(tidx); t < dimt; t += i32(THR_X_SL)) {
        wg_sh_mem[wg_offset + u32(t)] =
            interpolation_helper_dataf(setup.wgh, setup.coo, dimy, dimz, dimt, t);
    }
    return 0;
}

// ── trilinear interp: single channel from metric_map → wg_interp_out
fn trilinear_interp_metric(
    dimx: i32, dimy: i32, dimz: i32,
    point: vec3<f32>, interp_idx: u32
) -> i32 {
    let setup = trilinear_setup(dimx, dimy, dimz, point);
    if (setup.status != 0) { return -1; }

    wg_interp_out[interp_idx] =
        interpolation_helper_metric(setup.wgh, setup.coo, dimy, dimz);
    return 0;
}

// ── tissue check at a point ──────────────────────────────────────────
fn check_point_fn(
    tc_threshold: f32, point: vec3<f32>,
    dimx: i32, dimy: i32, dimz: i32,
    tidx: u32, tidy: u32
) -> i32 {
    let rv = trilinear_interp_metric(dimx, dimy, dimz, point, tidy);
    subgroupBarrier();

    if (rv != 0) {
        return OUTSIDEIMAGE;
    }
    if (wg_interp_out[tidy] > tc_threshold) {
        return TRACKPOINT;
    }
    return ENDPOINT;
}

// ── peak direction finding ──────────────────────────────────────────
// Finds local maxima on the ODF sphere, filters by relative threshold
// and minimum separation angle.
// ODF data is in wg_sh_mem[odf_offset .. odf_offset + samplm_nr].
// Results stored in wg_dirs_sh[dirs_offset .. dirs_offset + n * 3].
fn peak_directions_fn(
    odf_offset: u32, dirs_offset: u32, ind_offset: u32,
    num_edges: i32, samplm_nr: i32,
    relative_peak_thres: f32, min_separation_angle: f32,
    tidx: u32
) -> i32 {
    // Initialize index array (atomic store 0)
    for (var j = i32(tidx); j < samplm_nr; j += i32(THR_X_SL)) {
        atomicStore(&wg_sh_ind[ind_offset + u32(j)], 0);
    }

    let odf_min_raw = sg_min_reduce_wg(samplm_nr, odf_offset, REAL_MAX, tidx);
    let odf_min = max(0.0, odf_min_raw);

    subgroupBarrier();

    // Local maxima detection using sphere edges (atomic ops)
    for (var j = 0; j < num_edges; j += i32(THR_X_SL)) {
        if (j + i32(tidx) < num_edges) {
            let edge_idx = u32(j) + tidx;
            let u_ind = sphere_edges[edge_idx * 2u];
            let v_ind = sphere_edges[edge_idx * 2u + 1u];

            let u_val = wg_sh_mem[odf_offset + u32(u_ind)];
            let v_val = wg_sh_mem[odf_offset + u32(v_ind)];

            if (u_val < v_val) {
                atomicStore(&wg_sh_ind[ind_offset + u32(u_ind)], -1);
                atomicOr(&wg_sh_ind[ind_offset + u32(v_ind)], 1);
            } else if (v_val < u_val) {
                atomicStore(&wg_sh_ind[ind_offset + u32(v_ind)], -1);
                atomicOr(&wg_sh_ind[ind_offset + u32(u_ind)], 1);
            }
        }
    }
    subgroupBarrier();

    let comp_thres = relative_peak_thres *
        sg_max_mask_transl(samplm_nr, ind_offset, odf_offset, -odf_min, REAL_MIN, tidx);

    // Compact indices of local maxima above threshold using ballot
    var n: i32 = 0;
    let lmask = (1u << tidx) - 1u;  // lanes below me

    for (var j = 0; j < samplm_nr; j += i32(THR_X_SL)) {
        var v: i32 = -1;
        if (j + i32(tidx) < samplm_nr) {
            v = atomicLoad(&wg_sh_ind[ind_offset + u32(j) + tidx]);
        }
        let keep = (v > 0) &&
            ((wg_sh_mem[odf_offset + u32(j) + tidx] - odf_min) >= comp_thres);

        let ballot = subgroupBallot(keep);
        let msk = ballot.x;  // 32-bit mask for subgroup of 32

        if (keep) {
            let myoff = i32(countOneBits(msk & lmask));
            atomicStore(&wg_sh_ind[ind_offset + u32(n + myoff)], j + i32(tidx));
        }
        n += i32(countOneBits(msk));
    }
    subgroupBarrier();

    // Sort local maxima by ODF value (descending)
    if (n > 0 && n < i32(THR_X_SL)) {
        var k: f32 = REAL_MIN;
        var val: i32 = 0;
        if (i32(tidx) < n) {
            val = atomicLoad(&wg_sh_ind[ind_offset + tidx]);
            k = wg_sh_mem[odf_offset + u32(val)];
        }
        let sorted = warp_sort_kv_dec(k, val, tidx);
        subgroupBarrier();

        if (i32(tidx) < n) {
            atomicStore(&wg_sh_ind[ind_offset + tidx], sorted.val);
        }
    }
    subgroupBarrier();

    // Remove similar vertices (single-threaded on lane 0)
    if (n != 0) {
        if (tidx == 0u) {
            let cos_similarity = cos(min_separation_angle);

            let idx0 = atomicLoad(&wg_sh_ind[ind_offset]);
            let sv0 = load_sphere_verts_f3(u32(idx0));
            wg_dirs_sh[dirs_offset] = sv0.x;
            wg_dirs_sh[dirs_offset + 1u] = sv0.y;
            wg_dirs_sh[dirs_offset + 2u] = sv0.z;

            var k: i32 = 1;
            for (var i = 1; i < n; i++) {
                let idx_i = atomicLoad(&wg_sh_ind[ind_offset + u32(i)]);
                let abc = load_sphere_verts_f3(u32(idx_i));

                var j = 0;
                for (; j < k; j++) {
                    let d_base = dirs_offset + u32(j) * 3u;
                    let dx = wg_dirs_sh[d_base];
                    let dy = wg_dirs_sh[d_base + 1u];
                    let dz = wg_dirs_sh[d_base + 2u];
                    let cs = abs(abc.x * dx + abc.y * dy + abc.z * dz);
                    if (cs > cos_similarity) {
                        break;
                    }
                }
                if (j == k) {
                    let d_base = dirs_offset + u32(k) * 3u;
                    wg_dirs_sh[d_base] = abc.x;
                    wg_dirs_sh[d_base + 1u] = abc.y;
                    wg_dirs_sh[d_base + 2u] = abc.z;
                    k++;
                }
            }
            n = k;
        }
        n = i32(subgroupBroadcastFirst(u32(n)));
        subgroupBarrier();
    }

    return n;
}
