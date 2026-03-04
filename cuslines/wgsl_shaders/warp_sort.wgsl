// warp_sort.wgsl — Bitonic merge sort within a subgroup (32 lanes).
// Mirrors cuslines/metal_shaders/warp_sort.metal.
//
// WGSL has no templates, so we implement the 32-lane version directly
// (the only size used by peak_directions).

const WSORT_DIR_DEC: i32 = 0;

// Batcher's bitonic merge sort comparator networks for 32 elements.
// 15 stages, each with 32 swap indices.
const swap32_0:  array<i32, 32> = array<i32, 32>(16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15);
const swap32_1:  array<i32, 32> = array<i32, 32>( 8, 9,10,11,12,13,14,15, 0, 1, 2, 3, 4, 5, 6, 7,24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23);
const swap32_2:  array<i32, 32> = array<i32, 32>( 4, 5, 6, 7, 0, 1, 2, 3,16,17,18,19,20,21,22,23, 8, 9,10,11,12,13,14,15,28,29,30,31,24,25,26,27);
const swap32_3:  array<i32, 32> = array<i32, 32>( 2, 3, 0, 1, 4, 5, 6, 7,12,13,14,15, 8, 9,10,11,20,21,22,23,16,17,18,19,24,25,26,27,30,31,28,29);
const swap32_4:  array<i32, 32> = array<i32, 32>( 1, 0, 2, 3,16,17,18,19, 8, 9,10,11,24,25,26,27, 4, 5, 6, 7,20,21,22,23,12,13,14,15,28,29,31,30);
const swap32_5:  array<i32, 32> = array<i32, 32>( 0, 1, 2, 3, 8, 9,10,11, 4, 5, 6, 7,16,17,18,19,12,13,14,15,24,25,26,27,20,21,22,23,28,29,30,31);
const swap32_6:  array<i32, 32> = array<i32, 32>( 0, 1, 2, 3, 6, 7, 4, 5,10,11, 8, 9,14,15,12,13,18,19,16,17,22,23,20,21,26,27,24,25,28,29,30,31);
const swap32_7:  array<i32, 32> = array<i32, 32>( 0, 1,16,17, 4, 5,20,21, 8, 9,24,25,12,13,28,29, 2, 3,18,19, 6, 7,22,23,10,11,26,27,14,15,30,31);
const swap32_8:  array<i32, 32> = array<i32, 32>( 0, 1, 8, 9, 4, 5,12,13, 2, 3,16,17, 6, 7,20,21,10,11,24,25,14,15,28,29,18,19,26,27,22,23,30,31);
const swap32_9:  array<i32, 32> = array<i32, 32>( 0, 1, 4, 5, 2, 3, 8, 9, 6, 7,12,13,10,11,16,17,14,15,20,21,18,19,24,25,22,23,28,29,26,27,30,31);
const swap32_10: array<i32, 32> = array<i32, 32>( 0, 1, 3, 2, 5, 4, 7, 6, 9, 8,11,10,13,12,15,14,17,16,19,18,21,20,23,22,25,24,27,26,29,28,30,31);
const swap32_11: array<i32, 32> = array<i32, 32>( 0,16, 2,18, 4,20, 6,22, 8,24,10,26,12,28,14,30, 1,17, 3,19, 5,21, 7,23, 9,25,11,27,13,29,15,31);
const swap32_12: array<i32, 32> = array<i32, 32>( 0, 8, 2,10, 4,12, 6,14, 1,16, 3,18, 5,20, 7,22, 9,24,11,26,13,28,15,30,17,25,19,27,21,29,23,31);
const swap32_13: array<i32, 32> = array<i32, 32>( 0, 4, 2, 6, 1, 8, 3,10, 5,12, 7,14, 9,16,11,18,13,20,15,22,17,24,19,26,21,28,23,30,25,29,27,31);
const swap32_14: array<i32, 32> = array<i32, 32>( 0, 2, 1, 4, 3, 6, 5, 8, 7,10, 9,12,11,14,13,16,15,18,17,20,19,22,21,24,23,26,25,28,27,30,29,31);

// Helper to look up swap partner for a given stage and lane
fn swap32_lookup(stage: i32, lane: u32) -> i32 {
    switch (stage) {
        case 0:  { return swap32_0[lane]; }
        case 1:  { return swap32_1[lane]; }
        case 2:  { return swap32_2[lane]; }
        case 3:  { return swap32_3[lane]; }
        case 4:  { return swap32_4[lane]; }
        case 5:  { return swap32_5[lane]; }
        case 6:  { return swap32_6[lane]; }
        case 7:  { return swap32_7[lane]; }
        case 8:  { return swap32_8[lane]; }
        case 9:  { return swap32_9[lane]; }
        case 10: { return swap32_10[lane]; }
        case 11: { return swap32_11[lane]; }
        case 12: { return swap32_12[lane]; }
        case 13: { return swap32_13[lane]; }
        default: { return swap32_14[lane]; }
    }
}

// Key-value sort (descending) within subgroup of 32 lanes.
// Returns sorted (key, value) pair.
struct SortKV {
    key: f32,
    val: i32,
}

fn warp_sort_kv_dec(k_in: f32, val_in: i32, gid: u32) -> SortKV {
    var k = k_in;
    var val = val_in;

    for (var i = 0; i < 15; i++) {
        let srclane = swap32_lookup(i, gid);

        let a = subgroupShuffle(k, u32(srclane));
        let b = subgroupShuffle(val, u32(srclane));

        // WSORT_DIR_DEC = 0: descending
        if (i32(gid) < srclane) {
            // direction == DEC == 0 → (gid < srclane) == 0 is false → MAX branch
            if (a > k) { k = a; val = b; }
        } else {
            // (gid < srclane) == 0 → MIN branch
            if (a < k) { k = a; val = b; }
        }
    }
    return SortKV(k, val);
}
