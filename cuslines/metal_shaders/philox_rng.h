/* Philox4x32-10 counter-based RNG for Metal Shading Language.
 *
 * This implements the same algorithm as curandStatePhilox4_32_10_t so that,
 * given the same seed and sequence, the Metal and CUDA paths produce
 * identical random streams.
 *
 * Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3"
 *            (SC '11).  DOI 10.1145/2063384.2063405
 */

#ifndef __PHILOX_RNG_H__
#define __PHILOX_RNG_H__

#include <metal_stdlib>
using namespace metal;

// Philox constants
constant uint PHILOX_M4x32_0 = 0xD2511F53u;
constant uint PHILOX_M4x32_1 = 0xCD9E8D57u;
constant uint PHILOX_W32_0   = 0x9E3779B9u;
constant uint PHILOX_W32_1   = 0xBB67AE85u;

struct PhiloxState {
    uint4 counter;  // 128-bit counter (ctr)
    uint2 key;      // 64-bit key
    uint4 output;   // cached output of last round
    uint  idx;      // 0..3 index into output
    float cached_normal;   // Box-Muller second output cache
    bool  has_cached;      // true if cached_normal is valid
};

// ── single Philox round ──────────────────────────────────────────────

inline uint mulhi32(uint a, uint b) {
    return uint((ulong(a) * ulong(b)) >> 32);
}

inline uint4 philox4x32_single_round(uint4 ctr, uint2 key) {
    uint lo0 = ctr.x * PHILOX_M4x32_0;
    uint hi0 = mulhi32(ctr.x, PHILOX_M4x32_0);
    uint lo1 = ctr.z * PHILOX_M4x32_1;
    uint hi1 = mulhi32(ctr.z, PHILOX_M4x32_1);

    return uint4(hi1 ^ ctr.y ^ key.x,
                 lo1,
                 hi0 ^ ctr.w ^ key.y,
                 lo0);
}

// ── 10-round Philox4x32 ─────────────────────────────────────────────

inline uint4 philox4x32_10(uint4 ctr, uint2 key) {
    ctr = philox4x32_single_round(ctr, key); key += uint2(PHILOX_W32_0, PHILOX_W32_1);
    ctr = philox4x32_single_round(ctr, key); key += uint2(PHILOX_W32_0, PHILOX_W32_1);
    ctr = philox4x32_single_round(ctr, key); key += uint2(PHILOX_W32_0, PHILOX_W32_1);
    ctr = philox4x32_single_round(ctr, key); key += uint2(PHILOX_W32_0, PHILOX_W32_1);
    ctr = philox4x32_single_round(ctr, key); key += uint2(PHILOX_W32_0, PHILOX_W32_1);
    ctr = philox4x32_single_round(ctr, key); key += uint2(PHILOX_W32_0, PHILOX_W32_1);
    ctr = philox4x32_single_round(ctr, key); key += uint2(PHILOX_W32_0, PHILOX_W32_1);
    ctr = philox4x32_single_round(ctr, key); key += uint2(PHILOX_W32_0, PHILOX_W32_1);
    ctr = philox4x32_single_round(ctr, key); key += uint2(PHILOX_W32_0, PHILOX_W32_1);
    ctr = philox4x32_single_round(ctr, key);
    return ctr;
}

// ── curand-compatible initialisation ─────────────────────────────────
// Matches curand_init(seed, subsequence, offset, &state)

inline PhiloxState philox_init(uint seed_lo, uint seed_hi, uint subsequence, uint offset) {
    PhiloxState s;
    // curand packs the 64-bit seed into the two key words
    s.key = uint2(seed_lo, seed_hi);
    // subsequence goes into counter.y/z, offset into counter.x
    s.counter = uint4(0, 0, 0, 0);

    // Advance by subsequence (each subsequence = 2^67 values)
    // In practice subsequence fits in 32 bits; mirror curand layout.
    ulong subseq = ulong(subsequence);
    s.counter.y += uint(subseq);
    s.counter.z += uint(subseq >> 32);

    // Advance by offset (each offset = 4 outputs since Philox produces 4 uint per call)
    uint advance = offset / 4;
    uint remainder = offset % 4;
    s.counter.x += advance;

    // Generate first batch
    s.output = philox4x32_10(s.counter, s.key);
    s.idx = remainder;
    s.has_cached = false;
    s.cached_normal = 0.0f;
    return s;
}

// ── advance counter ──────────────────────────────────────────────────

inline void philox_next(thread PhiloxState& s) {
    s.counter.x += 1;
    if (s.counter.x == 0) {  // overflow
        s.counter.y += 1;
        if (s.counter.y == 0) {
            s.counter.z += 1;
            if (s.counter.z == 0) {
                s.counter.w += 1;
            }
        }
    }
    s.output = philox4x32_10(s.counter, s.key);
    s.idx = 0;
}

// ── generate uniform float in (0, 1] ────────────────────────────────
// Matches curand_uniform(&state)

inline float philox_uniform(thread PhiloxState& s) {
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
    // curand maps uint to (0, 1] then we mirror to [0, 1)
    // curand_uniform: result = uint * (1/2^32) but never 0
    // We use the same approach
    return float(bits) * 2.3283064365386963e-10f + 2.3283064365386963e-10f;
}

// ── generate standard normal via Box-Muller ──────────────────────────
// Matches curand_normal(&state) — caches second output for efficiency.

inline float philox_normal(thread PhiloxState& s) {
    if (s.has_cached) {
        s.has_cached = false;
        return s.cached_normal;
    }
    float u1 = philox_uniform(s);
    float u2 = philox_uniform(s);
    // Ensure u1 is not exactly 0 for the log
    u1 = max(u1, 1.0e-38f);
    float r = sqrt(-2.0f * log(u1));
    float theta = 2.0f * M_PI_F * u2;
    s.cached_normal = r * sin(theta);
    s.has_cached = true;
    return r * cos(theta);
}

#endif
