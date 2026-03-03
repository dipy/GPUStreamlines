# Project Guidelines

## Code Style
- Follow existing style in Python modules under `cuslines/`: typed signatures where already present, NumPy arrays normalized with `np.ascontiguousarray(...)`, and explicit dtype casting via constants from `cutils.py`, `mutils.py`, or `wgutils.py`.
- Keep public exports centralized in `cuslines/__init__.py` and each backend's `__init__.py`.
- Reuse logger pattern `logging.getLogger("GPUStreamlines")`.
- Do not edit generated constants in `cuslines/cuda_python/_globals.py` directly.

## Architecture
- **Three GPU backends**: CUDA, Metal (Apple Silicon), WebGPU (cross-platform via wgpu-py). Auto-detected at import time in `cuslines/__init__.py` (priority: Metal → CUDA → WebGPU).
- Entry script `run_gpu_streamlines.py` builds DIPY models, seeds, stopping criteria, and dispatches CPU (`LocalTracking`) or GPU (`GPUTracker`) pipelines.
- Each backend has a parallel structure: `*_tractography.py` (context manager), `*_propagate_seeds.py` (seed chunking), `*_direction_getters.py` (kernel dispatch).
- `boot_utils.py` is shared across all backends for bootstrap matrix preparation (OPDT/CSA).
- Kernel sources: `cuslines/cuda_c/` (CUDA), `cuslines/metal_shaders/` (MSL), `cuslines/wgsl_shaders/` (WGSL).
- Packaging step in `setup.py` generates Python constants from `cuslines/cuda_c/globals.h` via `defines_to_python(...)`.

## Build and Test
- Install (CUDA): `pip install ".[cu13]"` or `pip install ".[cu12]"`
- Install (Metal): `pip install ".[metal]"`
- Install (WebGPU): `pip install ".[webgpu]"`
- Minimal local run: `python run_gpu_streamlines.py --output-prefix small --nseeds 1000 --ngpus 1`
- Force a backend: `python run_gpu_streamlines.py --device=webgpu --output-prefix small --nseeds 1000`
- CPU reference: `python run_gpu_streamlines.py --device=cpu --output-prefix small --nseeds 1000`
- Benchmark: `python -m cuslines.webgpu.benchmark --nseeds 10000`
- Container: `docker build -t gpustreamlines .`
- No dedicated test/lint suite; validate by comparing CPU vs GPU outputs on the same seeds.

## Project Conventions
- GPU-facing arrays must be contiguous and use project scalar types (`REAL_DTYPE`, `REAL_SIZE`).
- Multi-GPU: CUDA supports multiple GPUs; Metal and WebGPU are single-GPU only.
- Angle units: CLI accepts degrees, internals convert to radians before GPU tracking.
- Preserve CPU/GPU parity pathways in `run_gpu_streamlines.py`.
- Treat `globals.h` as source-of-truth for constants; regenerate `_globals.py` through package build.
- SH basis: always use `real_sh_descoteaux(legacy=True)` for all matrices. See `boot_utils.py`.

## Integration Points
- DIPY provides model fitting, direction getters (CPU path), seeding, and stopping criteria.
- CUDA: `cuda-python` packages (`cuda.bindings`, `cuda.core`). Metal: `pyobjc-framework-Metal`. WebGPU: `wgpu`.
- I/O: nibabel (`StatefulTractogram`/TRK) and `trx-python` (`TrxFile`).

## Security
- No auth/secrets flow; primary risk surface is native GPU execution and memory management.
- CUDA: wrap all API calls with `checkCudaErrors(...)`.
- Avoid unbounded allocations: respect `--chunk-size`, `--nseeds`, and resizing behavior.
- Kernels are loaded only from packaged source directories, not arbitrary paths.
