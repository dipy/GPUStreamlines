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
    return None


BACKEND = _detect_backend()

if BACKEND == "metal":
    from cuslines.metal import (
        MetalGPUTracker as GPUTracker,
        MetalProbDirectionGetter as ProbDirectionGetter,
        MetalPttDirectionGetter as PttDirectionGetter,
        MetalBootDirectionGetter as BootDirectionGetter,
    )
elif BACKEND == "cuda":
    from cuslines.cuda_python import (
        GPUTracker,
        ProbDirectionGetter,
        PttDirectionGetter,
        BootDirectionGetter,
    )
else:
    raise ImportError(
        "No GPU backend available. Install either:\n"
        "  - CUDA: pip install 'cuslines[cu13]' (NVIDIA GPU)\n"
        "  - Metal: pip install 'cuslines[metal]' (Apple Silicon)"
    )

__all__ = [
    "GPUTracker",
    "ProbDirectionGetter",
    "PttDirectionGetter",
    "BootDirectionGetter",
    "BACKEND",
]
