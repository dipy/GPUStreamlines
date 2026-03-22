import platform as _platform


def _detect_backend():
    """Auto-detect the best available GPU backend."""
    system = _platform.system()
    if system == "Darwin":
        try:
            import Metal

            if Metal.MTLCreateSystemDefaultDevice() is not None:
                return "metal"
        except ImportError:
            pass
    try:
        from cuda.bindings import runtime

        count = runtime.cudaGetDeviceCount()
        if count[1] > 0:
            return "cuda"
    except (ImportError, Exception):
        pass
    try:
        import wgpu

        adapter = wgpu.gpu.request_adapter_sync()
        if adapter is not None:
            return "webgpu"
    except (ImportError, Exception):
        pass
    return None


BACKEND = _detect_backend()

if BACKEND == "metal":
    from cuslines.metal import (
        MetalBootDirectionGetter as BootDirectionGetter,
    )
    from cuslines.metal import (
        MetalGPUTracker as GPUTracker,
    )
    from cuslines.metal import (
        MetalProbDirectionGetter as ProbDirectionGetter,
    )
    from cuslines.metal import (
        MetalPttDirectionGetter as PttDirectionGetter,
    )
elif BACKEND == "cuda":
    from cuslines.cuda_python import (
        BootDirectionGetter,
        GPUTracker,
        ProbDirectionGetter,
        PttDirectionGetter,
    )
elif BACKEND == "webgpu":
    from cuslines.webgpu import (
        WebGPUBootDirectionGetter as BootDirectionGetter,
    )
    from cuslines.webgpu import (
        WebGPUProbDirectionGetter as ProbDirectionGetter,
    )
    from cuslines.webgpu import (
        WebGPUPttDirectionGetter as PttDirectionGetter,
    )
    from cuslines.webgpu import (
        WebGPUTracker as GPUTracker,
    )
else:
    raise ImportError(
        "No GPU backend available. Install either:\n"
        "  - CUDA: pip install 'cuslines[cu13]' (NVIDIA GPU)\n"
        "  - Metal: pip install 'cuslines[metal]' (Apple Silicon)\n"
        "  - WebGPU: pip install 'cuslines[webgpu]' (cross-platform)"
    )

__all__ = [
    "GPUTracker",
    "ProbDirectionGetter",
    "PttDirectionGetter",
    "BootDirectionGetter",
    "BACKEND",
]
