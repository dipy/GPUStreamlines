"""WebGPU backend utilities — type definitions, constants, buffer helpers.

Mirrors cuslines/metal/mutils.py and cuslines/cuda_python/cutils.py.
WebGPU (WGSL) only supports f32 (no f64), so REAL_SIZE is always 4.
"""

import numpy as np
import importlib.util
from enum import IntEnum
from pathlib import Path

# Import _globals.py directly (bypasses cuslines.cuda_python.__init__
# which would trigger CUDA imports).
_globals_path = Path(__file__).resolve().parent.parent / "cuda_python" / "_globals.py"
_spec = importlib.util.spec_from_file_location("_globals", str(_globals_path))
_globals_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_globals_mod)

MAX_SLINE_LEN = _globals_mod.MAX_SLINE_LEN
EXCESS_ALLOC_FACT = _globals_mod.EXCESS_ALLOC_FACT
MAX_SLINES_PER_SEED = _globals_mod.MAX_SLINES_PER_SEED
THR_X_BL = _globals_mod.THR_X_BL
THR_X_SL = _globals_mod.THR_X_SL
PMF_THRESHOLD_P = _globals_mod.PMF_THRESHOLD_P
NORM_EPS = _globals_mod.NORM_EPS


class ModelType(IntEnum):
    OPDT = 0
    CSA = 1
    PROB = 2
    PTT = 3


# WebGPU/WGSL only supports float32
REAL_SIZE = 4
REAL_DTYPE = np.float32

# Packed float3: 3 consecutive f32 values (12 bytes), matching CUDA/Metal layout.
REAL3_SIZE = 3 * REAL_SIZE
REAL3_DTYPE = np.dtype(
    [("x", np.float32), ("y", np.float32), ("z", np.float32)], align=False
)

BLOCK_Y = THR_X_BL // THR_X_SL


def div_up(a, b):
    return (a + b - 1) // b


def create_buffer_from_data(device, data, label=None):
    """Create a GPU storage buffer initialized with numpy array data.

    Parameters
    ----------
    device : wgpu.GPUDevice
    data : numpy.ndarray
        Must be C-contiguous.
    label : str, optional

    Returns
    -------
    wgpu.GPUBuffer
    """
    buf = device.create_buffer_with_data(
        data=np.ascontiguousarray(data).tobytes(),
        usage="STORAGE | COPY_SRC",
        label=label or "",
    )
    return buf


def create_empty_buffer(device, size_bytes, label=None):
    """Create an empty GPU storage buffer (for GPU-written outputs).

    Parameters
    ----------
    device : wgpu.GPUDevice
    size_bytes : int
    label : str, optional

    Returns
    -------
    wgpu.GPUBuffer
    """
    buf = device.create_buffer(
        size=size_bytes,
        usage="STORAGE | COPY_SRC | COPY_DST",
        label=label or "",
    )
    return buf


def read_buffer(device, buf, dtype=None):
    """Read a GPU buffer back to CPU as a numpy array.

    Unlike Metal's unified memory, WebGPU requires an explicit readback.

    Parameters
    ----------
    device : wgpu.GPUDevice
    buf : wgpu.GPUBuffer
    dtype : numpy dtype, optional
        If None, returns raw bytes.

    Returns
    -------
    numpy.ndarray or bytes
    """
    raw = device.queue.read_buffer(buf)
    if dtype is not None:
        # read_buffer returns an independent bytes copy; no need for .copy()
        return np.frombuffer(raw, dtype=dtype)
    return raw


def write_buffer(device, buf, data):
    """Write numpy data to a GPU buffer.

    Parameters
    ----------
    device : wgpu.GPUDevice
    buf : wgpu.GPUBuffer
    data : numpy.ndarray
    """
    device.queue.write_buffer(buf, 0, np.ascontiguousarray(data).tobytes())
