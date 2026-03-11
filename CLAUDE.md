# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPUStreamlines (`cuslines`) is a GPU-accelerated tractography package for diffusion MRI. It supports **three GPU backends**: NVIDIA CUDA, Apple Metal (Apple Silicon), and WebGPU (cross-platform via wgpu-py). Backend is auto-detected at import time in `cuslines/__init__.py` (priority: Metal → CUDA → WebGPU). Kernels are compiled at runtime (NVRTC for CUDA, `MTLDevice.newLibraryWithSource` for Metal, `device.create_shader_module` for WebGPU/WGSL).

## Build & Run

```bash
# Install (pick your backend)
pip install ".[cu13]"    # CUDA 13
pip install ".[cu12]"    # CUDA 12
pip install ".[metal]"   # Apple Metal (Apple Silicon)
pip install ".[webgpu]"  # WebGPU (cross-platform: NVIDIA, AMD, Intel, Apple)

# From PyPI
pip install "cuslines[cu13]"
pip install "cuslines[metal]"
pip install "cuslines[webgpu]"

# GPU run (downloads HARDI dataset if no data passed)
python run_gpu_streamlines.py --output-prefix small --nseeds 1000 --ngpus 1

# Force a specific backend
python run_gpu_streamlines.py --device=webgpu --output-prefix small --nseeds 1000

# CPU reference run (for comparison/debugging)
python run_gpu_streamlines.py --device=cpu --output-prefix small --nseeds 1000

# Docker
docker build -t gpustreamlines .
```

There is no dedicated test or lint suite. Validate by comparing CPU vs GPU outputs on the same seeds.

## Architecture

**Two-layer design**: Python orchestration + GPU kernels compiled at runtime. Three parallel backend implementations share the same API surface.

```
run_gpu_streamlines.py          # CLI entry: DIPY model fitting → CPU or GPU tracking
cuslines/
  __init__.py                   # Auto-detects Metal → CUDA → WebGPU backend at import
  boot_utils.py                 # Shared bootstrap matrix preparation (OPDT/CSA) for all backends
  cuda_python/                  # CUDA backend
    cu_tractography.py          # GPUTracker: context manager, multi-GPU allocation
    cu_propagate_seeds.py       # SeedBatchPropagator: chunked seed processing
    cu_direction_getters.py     # Direction getter ABC + Boot/Prob/PTT implementations
    cutils.py                   # REAL_DTYPE, REAL3_DTYPE, checkCudaErrors(), ModelType enum
    _globals.py                 # AUTO-GENERATED from globals.h (never edit manually)
  cuda_c/                       # CUDA kernel source
    globals.h                   # Source-of-truth for constants (REAL_SIZE, thread config)
    generate_streamlines_cuda.cu, boot.cu, ptt.cu, tracking_helpers.cu, utils.cu
    cudamacro.h, cuwsort.cuh, ptt.cuh, disc.h
  metal/                        # Metal backend (mirrors cuda_python/)
    mt_tractography.py, mt_propagate_seeds.py, mt_direction_getters.py, mutils.py
  metal_shaders/                # MSL kernel source (mirrors cuda_c/)
    globals.h, types.h, philox_rng.h
    generate_streamlines_metal.metal, boot.metal, ptt.metal
    tracking_helpers.metal, utils.metal, warp_sort.metal
  webgpu/                       # WebGPU backend (mirrors metal/)
    wg_tractography.py, wg_propagate_seeds.py, wg_direction_getters.py, wgutils.py
    benchmark.py                # Cross-backend benchmark: python -m cuslines.webgpu.benchmark
  wgsl_shaders/                 # WGSL kernel source (mirrors metal_shaders/)
    globals.wgsl, types.wgsl, philox_rng.wgsl
    utils.wgsl, warp_sort.wgsl, tracking_helpers.wgsl
    generate_streamlines.wgsl   # Prob/PTT buffer bindings + Prob getNum/gen kernels
    boot.wgsl                   # Boot direction getter kernels (standalone module)
    disc.wgsl, ptt.wgsl         # PTT support
```

**Data flow**: DIPY preprocessing → seed generation → GPUTracker context → SeedBatchPropagator chunks seeds across GPUs → kernel launch → stream results to TRK/TRX output.

**Direction getters** (subclasses of `GPUDirectionGetter`):
- `BootDirectionGetter` — bootstrap sampling from SH coefficients (OPDT/CSA models)
- `ProbDirectionGetter` — probabilistic selection from ODF/PMF (CSD model)
- `PttDirectionGetter` — Probabilistic Tracking with Turning (CSD model)

Each has `from_dipy_*()` class methods for initialization from DIPY models.

## Critical Conventions

- **`_globals.py` is auto-generated** from `cuslines/cuda_c/globals.h` during `setup.py` build via `defines_to_python()`. Never edit it manually; change `globals.h` and rebuild.
- **GPU arrays must be C-contiguous** — always use `np.ascontiguousarray()` and project scalar types (`REAL_DTYPE`, `REAL_SIZE` from `cutils.py` or `mutils.py`).
- **All CUDA API calls must be wrapped** with `checkCudaErrors()`.
- **Angle units**: CLI accepts degrees, internals convert to radians before the GPU layer.
- **Multi-GPU**: CUDA uses explicit `cudaSetDevice()` calls; Metal and WebGPU are single-GPU only.
- **CPU/GPU parity**: `run_gpu_streamlines.py` maintains parallel CPU and GPU code paths — keep both in sync when changing arguments or model-selection logic.
- **Logger**: use `logging.getLogger("GPUStreamlines")`.
- **Kernel compilation**: CUDA uses `cuda.core.Program` with NVIDIA headers. Metal uses `MTLDevice.newLibraryWithSource_options_error_()` with MSL source concatenated from `metal_shaders/`. WebGPU uses `device.create_shader_module()` with WGSL source concatenated from `wgsl_shaders/`.

## Metal Backend Notes

- **Unified memory**: Metal buffers use `storageModeShared` — numpy arrays are directly GPU-accessible (zero memcpy per batch, vs ~6 in CUDA).
- **float3 alignment**: All buffers use `packed_float3` (12 bytes) with `load_f3()`/`store_f3()` helpers. Metal `float3` is 16 bytes in registers.
- **Page alignment**: Use `aligned_array()` from `mutils.py` for arrays passed to `newBufferWithBytesNoCopy`.
- **No double precision**: Only `REAL_SIZE=4` (float32) is ported.
- **Warp primitives**: `__shfl_sync` → `simd_shuffle`, `__ballot_sync` → `simd_ballot`. SIMD width = 32.
- **SH basis**: Always use `real_sh_descoteaux(legacy=True)` for all matrices. See `boot_utils.py`.

## WebGPU Backend Notes

- **Cross-platform**: wgpu-py maps to Metal (macOS), Vulkan (Linux/Windows), D3D12 (Windows). Install: `pip install "cuslines[webgpu]"`.
- **Explicit readbacks**: `device.queue.read_buffer()` for GPU→CPU (~3 per seed batch, matching CUDA's cudaMemcpy pattern).
- **WGSL shaders**: Concatenated in dependency order by `compile_program()`. Boot compiles standalone; Prob/PTT share `generate_streamlines.wgsl`.
- **Buffer binding**: Boot needs 17 buffers across 3 bind groups. Prob/PTT use 2 bind groups. `layout="auto"` only includes reachable bindings.
- **Subgroups required**: Device feature `"subgroup"` (singular, not `"subgroups"`). Naga does NOT support `enable subgroups;` directive.
- **WGSL constraints**: No `ptr<storage>` parameters (use module-scope accessors). `var<workgroup>` sizes must be compile-time constants. PhiloxState is pass-by-value (return result structs).
- **Boot standalone module**: `_kernel_files()` returns `[]` to avoid `params` struct redefinition.
- **Benchmark**: `python -m cuslines.webgpu.benchmark --nseeds 10000` — auto-detects all backends.

## Key Dependencies

- `dipy` — diffusion models, CPU direction getters, seeding, stopping criteria
- `nibabel` — NIfTI/TRK file I/O (`StatefulTractogram`)
- `trx-python` — TRX format support (memory-mapped, for large outputs)
- `cuda-python` / `cuda-core` / `cuda-cccl` — CUDA Python bindings, kernel compilation, C++ headers
- `pyobjc-framework-Metal` / `pyobjc-framework-MetalPerformanceShaders` — Metal Python bindings (macOS only)
- `wgpu` — WebGPU Python bindings (wgpu-native, cross-platform)
- `numpy` — array operations throughout
