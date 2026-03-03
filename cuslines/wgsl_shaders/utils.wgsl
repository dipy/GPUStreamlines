// utils.wgsl — Reduction and prefix-sum primitives.
// Mirrors cuslines/metal_shaders/utils.metal.
//
// Uses subgroup operations (subgroupShuffleXor, subgroupShuffleUp, etc.)
// via the `enable subgroups;` directive prepended by the Python compiler.

// ── max reduction across subgroup ──────────────────────────────────
fn sg_max_reduce_wg(n: i32, wg_offset: u32, min_val: f32, tidx: u32) -> f32 {
    var m = min_val;
    for (var i = i32(tidx); i < n; i += i32(THR_X_SL)) {
        m = max(m, wg_sh_mem[wg_offset + u32(i)]);
    }
    m = max(m, subgroupShuffleXor(m, 16u));
    m = max(m, subgroupShuffleXor(m, 8u));
    m = max(m, subgroupShuffleXor(m, 4u));
    m = max(m, subgroupShuffleXor(m, 2u));
    m = max(m, subgroupShuffleXor(m, 1u));
    return m;
}

// ── min reduction across subgroup ──────────────────────────────────
fn sg_min_reduce_wg(n: i32, wg_offset: u32, max_val: f32, tidx: u32) -> f32 {
    var m = max_val;
    for (var i = i32(tidx); i < n; i += i32(THR_X_SL)) {
        m = min(m, wg_sh_mem[wg_offset + u32(i)]);
    }
    m = min(m, subgroupShuffleXor(m, 16u));
    m = min(m, subgroupShuffleXor(m, 8u));
    m = min(m, subgroupShuffleXor(m, 4u));
    m = min(m, subgroupShuffleXor(m, 2u));
    m = min(m, subgroupShuffleXor(m, 1u));
    return m;
}

// ── max with mask+translate reduction ──────────────────────────────
// Only considers entries where mask > 0, adds offset to value.
fn sg_max_mask_transl(n: i32, wg_ind_offset: u32, wg_val_offset: u32,
                      offset_val: f32, min_val: f32, tidx: u32) -> f32 {
    var m = min_val;
    for (var i = i32(tidx); i < n; i += i32(THR_X_SL)) {
        let sel = atomicLoad(&wg_sh_ind[wg_ind_offset + u32(i)]);
        if (sel > 0) {
            m = max(m, wg_sh_mem[wg_val_offset + u32(i)] + offset_val);
        }
    }
    m = max(m, subgroupShuffleXor(m, 16u));
    m = max(m, subgroupShuffleXor(m, 8u));
    m = max(m, subgroupShuffleXor(m, 4u));
    m = max(m, subgroupShuffleXor(m, 2u));
    m = max(m, subgroupShuffleXor(m, 1u));
    return m;
}

// ── inclusive prefix sum in workgroup memory ─────────────────────────
// Operates on wg_sh_mem[offset .. offset+len].
fn prefix_sum_sh(wg_offset: u32, len: i32, tidx: u32) {
    for (var j = 0; j < len; j += i32(THR_X_SL)) {
        if (tidx == 0u && j != 0) {
            wg_sh_mem[wg_offset + u32(j)] += wg_sh_mem[wg_offset + u32(j - 1)];
        }
        subgroupBarrier();

        var t_pmf: f32 = 0.0;
        if (j + i32(tidx) < len) {
            t_pmf = wg_sh_mem[wg_offset + u32(j) + tidx];
        }
        for (var i = 1u; i < THR_X_SL; i *= 2u) {
            let tmp = subgroupShuffleUp(t_pmf, i);
            if (tidx >= i && j + i32(tidx) < len) {
                t_pmf += tmp;
            }
        }
        if (j + i32(tidx) < len) {
            wg_sh_mem[wg_offset + u32(j) + tidx] = t_pmf;
        }
        subgroupBarrier();
    }
}
