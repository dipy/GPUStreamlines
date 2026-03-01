# Metal Backend for GPUStreamlines

The Metal backend runs GPU-accelerated tractography on Apple Silicon (M1/M2/M3/M4) using Apple's Metal Shading Language. It mirrors the CUDA backend's functionality with the same API surface, and is auto-detected at import time on macOS.

## Installation

```bash
pip install "cuslines[metal]"     # from PyPI
pip install ".[metal]"            # from source
```

Requires macOS 13+ and Apple Silicon. Dependencies: `pyobjc-framework-Metal`, `pyobjc-framework-MetalPerformanceShaders`.

## Usage

```bash
# GPU (auto-detects Metal on macOS)
python run_gpu_streamlines.py --output-prefix out --nseeds 10000 --ngpus 1

# Explicit Metal device
python run_gpu_streamlines.py --device metal --output-prefix out --nseeds 10000

# CPU reference (DIPY)
python run_gpu_streamlines.py --device cpu --output-prefix out_cpu --nseeds 10000
```

All CLI arguments (`--max-angle`, `--step-size`, `--fa-threshold`, `--model`, `--dg`, etc.) work identically to the CUDA backend.

## Benchmarks

Measured on Apple M4 Pro (20-core GPU), Stanford HARDI dataset (81x106x76, 160 directions), OPDT model with bootstrap direction getter, 10,000 seeds:

| | Metal GPU | CPU (DIPY) |
|---|---|---|
| **Streamline generation time** | 0.89 s | 91.6 s |
| **Speedup** | **~100x** | 1x |
| **Streamlines generated** | 13,205 | 13,647 |
| **Mean fiber length** | 53.8 pts | 45.4 pts |
| **Median fiber length** | 42.0 pts | 33.0 pts |
| **Commissural fibers** | 1,656 | 1,522 |

The GPU produces comparable streamline counts and commissural fiber density. Mean fiber length is ~18% longer on the GPU due to float32 vs float64 precision differences in ODF peak selection at fiber crossings.

## Architecture

### Unified memory advantage

Apple Silicon shares CPU and GPU memory. Metal buffers use `storageModeShared`, so numpy arrays backing `MTLBuffer` objects are directly GPU-accessible. The CUDA backend requires ~6 `cudaMemcpy` calls per seed batch to transfer data between host and device; **the Metal backend requires zero**. For workloads with large read-only input data (the 4D ODF array is often hundreds of MB), this eliminates a significant source of latency.

### Kernel compilation

MSL source files in `cuslines/metal_shaders/` are concatenated and compiled at runtime via `MTLDevice.newLibraryWithSource`. This mirrors the CUDA path (NVRTC), with compile-time constants passed as preprocessor defines.

### File layout

```
cuslines/metal/
  mt_tractography.py          MetalGPUTracker context manager
  mt_propagate_seeds.py       Chunked seed processing (no memcpy)
  mt_direction_getters.py     Boot/Prob/PTT direction getters
  mutils.py                   Types, aligned allocation, error checking

cuslines/metal_shaders/
  globals.h                   Shared constants (float32 only)
  types.h                     packed_float3 <-> float3 helpers
  philox_rng.h                Philox4x32-10 RNG (replaces curand)
  boot.metal                  Bootstrap direction getter kernel
  ptt.metal                   PTT direction getter kernel
  generate_streamlines_metal.metal   Main streamline generation kernel
  tracking_helpers.metal       Trilinear interpolation, peak finding
  utils.metal                 SIMD reductions, prefix sum
  warp_sort.metal             Bitonic sort
  disc.h                      Lookup tables for PTT
```

### Key implementation details

- **float3 alignment**: CUDA `float3` is 12 bytes in arrays; Metal `float3` is 16 bytes. All device buffers use `packed_float3` (12 bytes) with `load_f3()`/`store_f3()` helpers for register conversion.
- **Page alignment**: Metal shared buffers require 16KB-aligned memory. `aligned_array()` in `mutils.py` handles this.
- **RNG**: Philox4x32-10 counter-based RNG in MSL, matching curand's algorithm for reproducible streams.
- **SIMD mapping**: CUDA warp primitives map directly to Metal SIMD group operations (`__shfl_sync` -> `simd_shuffle`, `__ballot_sync` -> `simd_ballot`). Apple GPU SIMD width is 32, matching CUDA's warp size.
- **No double precision**: Metal GPUs do not support float64. Only the float32 path is ported.
- **SH basis convention**: The sampling matrix, H/R matrices, and OPDT/CSA model matrices must all use the same spherical harmonics basis (`real_sh_descoteaux` with `legacy=True`). A basis mismatch causes sign flips in odd-m SH columns that corrupt ODF reconstruction.

## Optional: Soft Angular Weighting

The bootstrap direction getter in `boot.metal` includes an optional soft angular weighting feature that is **disabled by default** and compiled out at the preprocessor level (zero runtime cost when disabled).

### Motivation

At fiber crossings (e.g., the corona radiata, where commissural and projection fibers intersect), the ODF typically shows multiple peaks. The standard algorithm selects the peak closest to the current trajectory direction. However, when two peaks have similar magnitudes, float32 precision noise can cause the wrong peak to be selected, sending the fiber on an incorrect trajectory.

In biological white matter, a fiber that has been traveling in a consistent direction is more likely to continue in that direction than to make a sharp turn. This prior is not captured by the standard closest-peak algorithm, which treats all peaks above threshold equally during the peak-finding step.

### Implementation

When enabled, the weighting multiplies each ODF sample by an angular similarity factor before the PMF threshold is applied:

```
PMF[j] *= (1 - w) + w * |cos(angle between current direction and sphere vertex j)|
```

This has two effects:
1. Peaks aligned with the current trajectory retain full weight
2. Perpendicular peaks are suppressed by a factor of `(1 - w)`

Because the weighting is applied before the 5% absolute threshold and 25% relative peak threshold, it can prevent aligned peaks from being incorrectly zeroed out when a strong perpendicular peak dominates.

### Configuration

Set the `angular_weight` attribute on the direction getter before tracking:

```python
from cuslines import BootDirectionGetter
dg = BootDirectionGetter.from_dipy_opdt(gtab, sphere)
dg.angular_weight = 0.5  # 0.0 = disabled (default), 0.5 = moderate
```

### Effect on tracking (10,000 seeds, HARDI dataset)

| | weight = 0.0 (default) | weight = 0.5 | CPU (DIPY) |
|---|---|---|---|
| **Streamlines** | 13,205 | 13,307 | 13,647 |
| **Mean fiber length** | 53.8 pts | 64.8 pts | 45.4 pts |
| **Commissural fibers** | 1,656 | 1,915 | 1,522 |

With the corrected SH basis, the default (no weighting) already produces good parity with CPU. The weighting increases mean fiber length and commissural fiber count beyond what the CPU produces. Whether this deviation is desirable depends on the application: for strict CPU/GPU reproducibility, leave it disabled; for applications where longer fibers through crossing regions are preferred, a value of 0.3-0.5 may be appropriate.
