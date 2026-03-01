/* Metal type helpers — handles the packed_float3 / float3 alignment difference.
 *
 * In CUDA, float3 is 12 bytes in arrays (no padding).
 * In Metal, float3 is 16 bytes.  packed_float3 is 12 bytes.
 *
 * Strategy:
 *   - Device buffers use packed_float3 (12 bytes) → matches CUDA layout and
 *     Python numpy dtype, so all buffer size calculations remain unchanged.
 *   - Computation uses float3 (16 bytes) in registers/threadgroup memory.
 *   - load/store helpers convert between the two.
 */

#ifndef __TYPES_H__
#define __TYPES_H__

#include <metal_stdlib>
using namespace metal;

// ── buffer ↔ register conversions ────────────────────────────────────

inline float3 load_f3(const device packed_float3* p, uint idx) {
    return float3(p[idx]);
}

inline float3 load_f3(const device packed_float3& p) {
    return float3(p);
}

inline void store_f3(device packed_float3* p, uint idx, float3 v) {
    p[idx] = packed_float3(v);
}

inline void store_f3(device packed_float3& p, float3 v) {
    p = packed_float3(v);
}

// threadgroup load/store — threadgroup memory can use float3 directly
// but we sometimes index packed arrays in threadgroup memory too
inline float3 load_f3(const threadgroup packed_float3* p, uint idx) {
    return float3(p[idx]);
}

inline void store_f3(threadgroup packed_float3* p, uint idx, float3 v) {
    p[idx] = packed_float3(v);
}

// ── CUDA MAKE_REAL3 replacement ──────────────────────────────────────
#define MAKE_REAL3(x, y, z) float3((x), (y), (z))

#endif
