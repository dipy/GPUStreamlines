# WebGPU Backend for GPUStreamlines

The WebGPU backend runs GPU-accelerated tractography on any GPU (NVIDIA, AMD, Intel, Apple) via [wgpu-py](https://github.com/pygfx/wgpu-py), Python bindings for wgpu-native. It mirrors the Metal and CUDA backends' functionality with the same API surface, and is auto-detected at import time when no vendor-specific backend is available.

## Installation

```bash
pip install "cuslines[webgpu]"     # from PyPI
pip install ".[webgpu]"            # from source
```

Requires a GPU with subgroup operation support. Dependency: `wgpu>=0.18` (pure Python, installs pre-built wgpu-native binaries for all platforms).

## Usage

```bash
# GPU (auto-detects: Metal -> CUDA -> WebGPU)
python run_gpu_streamlines.py --output-prefix out --nseeds 10000 --ngpus 1

# Explicit WebGPU device
python run_gpu_streamlines.py --device webgpu --output-prefix out --nseeds 10000

# CPU reference (DIPY)
python run_gpu_streamlines.py --device cpu --output-prefix out_cpu --nseeds 10000
```

All CLI arguments (`--max-angle`, `--step-size`, `--fa-threshold`, `--model`, `--dg`, etc.) work identically to the CUDA and Metal backends.

## Benchmarks

Measured on Apple M4 Pro (20-core GPU), Stanford HARDI dataset (81x106x76, 160 directions), OPDT model with bootstrap direction getter, 100,000 seeds:

| | WebGPU | Metal GPU | CPU (DIPY) |
|---|---|---|---|
| **Streamline generation time** | 19.1 s | 9.4 s | 894 s |
| **Speedup vs CPU** | **~47x** | ~95x | 1x |
| **Streamlines generated** | 132,201 | 132,201 | 135,984 |
| **Mean fiber length** | 54.5 pts | 54.5 pts | 45.6 pts |
| **Median fiber length** | 43.0 pts | 43.0 pts | 34.0 pts |
| **Commissural fibers** | 19,412 | 19,412 | 17,381 |

WebGPU and Metal produce bit-identical streamline results (same RNG, same float32 codepath). The ~2x speed difference vs Metal on Apple Silicon is due to explicit `read_buffer()` readbacks — Metal's unified memory gives zero-copy buffer access, while WebGPU requires ~3 GPU-to-CPU readbacks per seed batch. On non-Apple hardware (NVIDIA/AMD via Vulkan, Intel via D3D12), WebGPU is the only cross-platform option and the readback overhead is comparable to CUDA's `cudaMemcpy`.

Mean fiber length is ~19% longer on the GPU than CPU due to float32 vs float64 precision differences in ODF peak selection at fiber crossings.

The CPU benchmark uses DIPY's `LocalTracking`, which is single-threaded Python. Multi-threaded BLAS/numpy libraries (OpenMP, MKL) do not measurably affect tracking time since each streamline step involves small Python-level operations rather than large matrix computations. Verified: restricting to 1 BLAS thread (`OMP_NUM_THREADS=1`) produces identical CPU timing (~89s at 10k seeds vs ~90s with default threads).

### Reproducing benchmarks

A self-contained benchmark script auto-detects available backends and prints a comparison table:

```bash
# Default: 10k seeds, all available backends + CPU
python -m cuslines.webgpu.benchmark

# 100k seeds, skip slow CPU run
python -m cuslines.webgpu.benchmark --nseeds 100000 --skip-cpu

# Specific backends only
python -m cuslines.webgpu.benchmark --nseeds 10000 --backends webgpu metal
```

The script downloads the Stanford HARDI dataset on first run, then reports timing, streamline count, mean/median fiber length, and commissural fiber count for each backend.

## Architecture

### Cross-platform GPU access

WebGPU is a hardware abstraction layer that maps to the native GPU API on each platform:
- **macOS**: Metal (via wgpu-native)
- **Linux**: Vulkan
- **Windows**: D3D12 or Vulkan

This means the same WGSL shader code runs on NVIDIA, AMD, Intel, and Apple GPUs without modification.

### Explicit buffer readbacks

Unlike Metal on Apple Silicon (unified memory, zero-copy), WebGPU requires `device.queue.read_buffer()` to read GPU results back to CPU. Three readbacks per seed batch:
1. After pass 1: `slinesOffs` for CPU prefix sum
2. After pass 2: `sline` (streamline coordinates)
3. After pass 2: `slineLen` and `slineSeed`

This matches the CUDA backend's `cudaMemcpy` pattern.

### Shader compilation

WGSL source files in `cuslines/wgsl_shaders/` are concatenated in dependency order and compiled at runtime via `device.create_shader_module()`. Boot compiles as a standalone module (separate buffer bindings); Prob/PTT share a module with `generate_streamlines.wgsl`.

### Buffer binding groups

WebGPU's default guarantees only 8 storage buffers per shader stage. The Boot direction getter needs 17 buffers, so the device requests `maxStorageBuffersPerShaderStage: 17` and splits buffers across 3 bind groups:

- **Group 0**: params, seeds, dataf, metric_map, sphere_vertices, sphere_edges
- **Group 1**: H, R, delta_b, delta_q, sampling_matrix, b0s_mask
- **Group 2**: slineOutOff, shDir0, slineSeed, slineLen, sline

Prob/PTT need only 11 buffers across 2 bind groups.

### File layout

```
cuslines/webgpu/
  wg_tractography.py          WebGPUTracker context manager
  wg_propagate_seeds.py       Chunked seed processing (explicit readbacks)
  wg_direction_getters.py     Boot/Prob/PTT direction getters
  wgutils.py                  Constants, buffer helpers, ModelType enum

cuslines/wgsl_shaders/
  globals.wgsl                Shared constants (const declarations)
  types.wgsl                  f32x3 load/store documentation
  philox_rng.wgsl             Philox4x32-10 RNG (replaces curand)
  boot.wgsl                   Bootstrap direction getter kernel (standalone)
  ptt.wgsl                    PTT direction getter kernel
  disc.wgsl                   Lookup tables for PTT
  generate_streamlines.wgsl   Prob/PTT buffer bindings + Prob kernels
  tracking_helpers.wgsl       Trilinear interpolation, peak finding
  utils.wgsl                  Subgroup reductions, prefix sum
  warp_sort.wgsl              Bitonic sort
```

### Key implementation details

- **Subgroup operations required**: All kernels use `subgroupShuffle`, `subgroupBallot`, `subgroupBarrier` for SIMD-parallel reductions. The `"subgroup"` device feature must be available; device creation fails with a clear error if not. Naga (wgpu-native's shader compiler) does not support the `enable subgroups;` WGSL directive — subgroup builtins work via the device feature alone.
- **No `ptr<storage>` function parameters**: WGSL only allows `function`, `private`, and `workgroup` address space pointers as function parameters. Buffer access uses buffer-specific helper functions at module scope.
- **PhiloxState pass-by-value**: WGSL has no mutable references to local structs. Every function that modifies PhiloxState returns a result struct bundling the RNG state with its output.
- **Static workgroup memory**: WGSL requires compile-time-constant `var<workgroup>` array sizes. Boot uses `array<f32, 4096>` (16KB); PTT arrays are prefixed with `ptt_` to avoid name conflicts.
- **RNG**: Philox4x32-10 counter-based RNG in WGSL, matching the CUDA and Metal implementations for reproducible streams.
- **SIMD mapping**: CUDA/Metal warp primitives map to WGSL subgroup operations (`simd_shuffle` -> `subgroupShuffle`, `simd_ballot` -> `subgroupBallot`). Apple GPU subgroup size is 32, matching CUDA's warp size.
- **No double precision**: WGSL `f64` is not widely supported. Only the float32 path is ported.
- **SH basis convention**: Same as Metal — the sampling matrix, H/R matrices, and OPDT/CSA model matrices must all use `real_sh_descoteaux` with `legacy=True`.
