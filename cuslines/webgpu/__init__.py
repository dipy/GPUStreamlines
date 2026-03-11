"""WebGPU backend for GPU-accelerated tractography.

Uses wgpu-py (Python WebGPU bindings backed by wgpu-native) for
cross-platform GPU compute on NVIDIA, AMD, Intel, and Apple GPUs.
"""

from cuslines.webgpu.wg_tractography import WebGPUTracker
from cuslines.webgpu.wg_direction_getters import (
    WebGPUProbDirectionGetter,
    WebGPUPttDirectionGetter,
    WebGPUBootDirectionGetter,
)

__all__ = [
    "WebGPUTracker",
    "WebGPUProbDirectionGetter",
    "WebGPUPttDirectionGetter",
    "WebGPUBootDirectionGetter",
]
