// philox_rng.wgsl — Philox4x32-10 counter-based RNG for WGSL.
//
// Implements the same algorithm as curandStatePhilox4_32_10_t (CUDA) and
// the MSL port in philox_rng.h, so that given the same seed and sequence,
// all backends produce identical random streams.
//
// Key WGSL adaptation: no mutable references to local structs across function
// boundaries. Every function that modifies PhiloxState receives it by value
// and returns the modified copy.
//
// Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3"
//            (SC '11).  DOI 10.1145/2063384.2063405

// Philox constants
const PHILOX_M4x32_0: u32 = 0xD2511F53u;
const PHILOX_M4x32_1: u32 = 0xCD9E8D57u;
const PHILOX_W32_0: u32   = 0x9E3779B9u;
const PHILOX_W32_1: u32   = 0xBB67AE85u;

const PI_F: f32 = 3.14159265358979323846;

struct PhiloxState {
    counter: vec4<u32>,  // 128-bit counter
    key: vec2<u32>,      // 64-bit key
    output: vec4<u32>,   // cached output of last round
    idx: u32,            // 0..3 index into output
    cached_normal: f32,  // Box-Muller second output cache
    has_cached: u32,     // 1 if cached_normal is valid, 0 otherwise
}

// ── 32-bit high multiplication (upper 32 bits of a*b) ───────────────
// WGSL has no u64, so we split into 16-bit halves and recombine.
fn mulhi32(a: u32, b: u32) -> u32 {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let lo_lo = a_lo * b_lo;
    let lo_hi = a_lo * b_hi;
    let hi_lo = a_hi * b_lo;
    let hi_hi = a_hi * b_hi;

    // Accumulate the middle terms, tracking carry into the upper 32 bits
    let mid_sum = (lo_lo >> 16u) + (lo_hi & 0xFFFFu) + (hi_lo & 0xFFFFu);
    let result = hi_hi + (lo_hi >> 16u) + (hi_lo >> 16u) + (mid_sum >> 16u);
    return result;
}

// ── single Philox round ─────────────────────────────────────────────
fn philox4x32_single_round(ctr: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    let lo0 = ctr.x * PHILOX_M4x32_0;
    let hi0 = mulhi32(ctr.x, PHILOX_M4x32_0);
    let lo1 = ctr.z * PHILOX_M4x32_1;
    let hi1 = mulhi32(ctr.z, PHILOX_M4x32_1);

    return vec4<u32>(
        hi1 ^ ctr.y ^ key.x,
        lo1,
        hi0 ^ ctr.w ^ key.y,
        lo0
    );
}

// ── 10-round Philox4x32 ────────────────────────────────────────────
fn philox4x32_10(ctr_in: vec4<u32>, key_in: vec2<u32>) -> vec4<u32> {
    var ctr = ctr_in;
    var key = key_in;
    let bump = vec2<u32>(PHILOX_W32_0, PHILOX_W32_1);

    ctr = philox4x32_single_round(ctr, key); key += bump;
    ctr = philox4x32_single_round(ctr, key); key += bump;
    ctr = philox4x32_single_round(ctr, key); key += bump;
    ctr = philox4x32_single_round(ctr, key); key += bump;
    ctr = philox4x32_single_round(ctr, key); key += bump;
    ctr = philox4x32_single_round(ctr, key); key += bump;
    ctr = philox4x32_single_round(ctr, key); key += bump;
    ctr = philox4x32_single_round(ctr, key); key += bump;
    ctr = philox4x32_single_round(ctr, key); key += bump;
    ctr = philox4x32_single_round(ctr, key);
    return ctr;
}

// ── curand-compatible initialisation ────────────────────────────────
// Matches curand_init(seed, subsequence, offset, &state)
fn philox_init(seed_lo: u32, seed_hi: u32, subsequence: u32, offset: u32) -> PhiloxState {
    var s: PhiloxState;
    s.key = vec2<u32>(seed_lo, seed_hi);
    s.counter = vec4<u32>(0u, 0u, 0u, 0u);

    // Advance by subsequence (each subsequence = 2^67 values)
    s.counter.y += subsequence;
    // High bits of subsequence would go into counter.z, but subsequence
    // fits in 32 bits in practice, so no shift needed.

    // Advance by offset (each offset = 4 outputs since Philox produces
    // 4 u32 per call)
    let advance = offset / 4u;
    let remainder = offset % 4u;
    s.counter.x += advance;

    // Generate first batch
    s.output = philox4x32_10(s.counter, s.key);
    s.idx = remainder;
    s.has_cached = 0u;
    s.cached_normal = 0.0;
    return s;
}

// ── advance counter ─────────────────────────────────────────────────
fn philox_next(s: PhiloxState) -> PhiloxState {
    var r = s;
    r.counter.x += 1u;
    if (r.counter.x == 0u) {  // overflow
        r.counter.y += 1u;
        if (r.counter.y == 0u) {
            r.counter.z += 1u;
            if (r.counter.z == 0u) {
                r.counter.w += 1u;
            }
        }
    }
    r.output = philox4x32_10(r.counter, r.key);
    r.idx = 0u;
    return r;
}

// ── result types for pass-by-value pattern ──────────────────────────

struct PhiloxUniformResult {
    state: PhiloxState,
    value: f32,
}

struct PhiloxUintResult {
    state: PhiloxState,
    value: u32,
}

struct PhiloxNormalResult {
    state: PhiloxState,
    value: f32,
}

// ── generate raw u32 ────────────────────────────────────────────────
fn philox_uint(s: PhiloxState) -> PhiloxUintResult {
    var r = s;
    if (r.idx >= 4u) {
        r = philox_next(r);
    }
    var bits: u32;
    switch (r.idx) {
        case 0u: { bits = r.output.x; }
        case 1u: { bits = r.output.y; }
        case 2u: { bits = r.output.z; }
        default: { bits = r.output.w; }
    }
    r.idx += 1u;
    return PhiloxUintResult(r, bits);
}

// ── generate uniform float in (0, 1] ───────────────────────────────
// Matches curand_uniform(&state)
fn philox_uniform(s: PhiloxState) -> PhiloxUniformResult {
    let ur = philox_uint(s);
    let value = f32(ur.value) * 2.3283064365386963e-10 + 2.3283064365386963e-10;
    return PhiloxUniformResult(ur.state, value);
}

// ── generate standard normal via Box-Muller ─────────────────────────
// Matches curand_normal(&state) — caches second output for efficiency.
fn philox_normal(s: PhiloxState) -> PhiloxNormalResult {
    var r = s;
    if (r.has_cached == 1u) {
        r.has_cached = 0u;
        return PhiloxNormalResult(r, r.cached_normal);
    }
    let ur1 = philox_uniform(r);
    r = ur1.state;
    let ur2 = philox_uniform(r);
    r = ur2.state;
    let u1 = max(ur1.value, 1.0e-38);
    let u2 = ur2.value;
    let rad = sqrt(-2.0 * log(u1));
    let theta = 2.0 * PI_F * u2;
    r.cached_normal = rad * sin(theta);
    r.has_cached = 1u;
    return PhiloxNormalResult(r, rad * cos(theta));
}
