// types.wgsl — Load/store helpers for packed float3 data in flat f32 buffers.
//
// In CUDA, float3 arrays use 12 bytes per element (no padding).
// In Metal, device buffers use packed_float3 (12 bytes) with load/store helpers.
// In WGSL, storage buffer arrays of f32 store data as contiguous floats.
// We store vec3<f32> as 3 consecutive f32 values and convert on load/store.
//
// WGSL does not allow ptr<storage, ...> as function parameters, so we cannot
// write generic load/store helpers. Instead, buffer-specific helpers are
// defined alongside the buffer declarations in each kernel file, e.g.:
//
//   fn load_seeds_f3(idx: u32) -> vec3<f32> {
//       let base = idx * 3u;
//       return vec3<f32>(seeds[base], seeds[base + 1u], seeds[base + 2u]);
//   }
//
// For workgroup memory, inline the 3-element access pattern directly:
//   let v = vec3<f32>(wg_arr[base], wg_arr[base+1], wg_arr[base+2]);
//   wg_arr[base] = v.x; wg_arr[base+1] = v.y; wg_arr[base+2] = v.z;
