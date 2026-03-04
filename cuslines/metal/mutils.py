"""Metal backend utilities — type definitions, error checking, aligned allocation.

Mirrors cuslines/cuda_python/cutils.py for the Metal backend.
Metal only supports float32, so no REAL_SIZE branching is needed.
"""

import numpy as np
import ctypes
import ctypes.util
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

# Re-export globals
__all__ = [
    "ModelType",
    "REAL_SIZE",
    "REAL_DTYPE",
    "REAL3_SIZE",
    "REAL3_DTYPE",
    "BLOCK_Y",
    "MAX_SLINE_LEN",
    "EXCESS_ALLOC_FACT",
    "MAX_SLINES_PER_SEED",
    "THR_X_BL",
    "THR_X_SL",
    "PMF_THRESHOLD_P",
    "NORM_EPS",
    "div_up",
    "checkMetalError",
    "aligned_array",
    "PAGE_SIZE",
]


class ModelType(IntEnum):
    OPDT = 0
    CSA = 1
    PROB = 2
    PTT = 3


# Metal only supports float32
REAL_SIZE = 4
REAL_DTYPE = np.float32

# packed_float3 in Metal is 12 bytes — same layout as CUDA float3 in arrays.
# align=False ensures numpy uses 12-byte stride, not 16.
REAL3_SIZE = 3 * REAL_SIZE
REAL3_DTYPE = np.dtype(
    [("x", np.float32), ("y", np.float32), ("z", np.float32)], align=False
)

BLOCK_Y = THR_X_BL // THR_X_SL

# Apple Silicon page size (16 KB). Buffers passed to
# newBufferWithBytesNoCopy must be page-aligned.
PAGE_SIZE = 16384


def div_up(a, b):
    return (a + b - 1) // b


def checkMetalError(error):
    """Raise if an NSError was returned from a Metal API call."""
    if error is not None:
        desc = error.localizedDescription()
        raise RuntimeError(f"Metal error: {desc}")


# ── page-aligned allocation ───────────────────────────────────────────

_libc_name = ctypes.util.find_library("c")
_libc = ctypes.CDLL(_libc_name, use_errno=True)
_libc.free.argtypes = [ctypes.c_void_p]
_libc.free.restype = None


def _posix_memalign(size, alignment=PAGE_SIZE):
    """Allocate *size* bytes aligned to *alignment* using posix_memalign."""
    ptr = ctypes.c_void_p()
    ret = _libc.posix_memalign(ctypes.byref(ptr), alignment, size)
    if ret != 0:
        raise MemoryError(
            f"posix_memalign failed (ret={ret}) for size={size}, align={alignment}"
        )
    return ptr


def aligned_array(shape, dtype=np.float32, alignment=PAGE_SIZE):
    """Return a C-contiguous numpy array whose underlying memory is page-aligned.

    Suitable for wrapping with Metal's ``newBufferWithBytesNoCopy``.
    The returned array owns a prevent-GC reference to the raw buffer.
    """
    dtype = np.dtype(dtype)
    count = int(np.prod(shape))
    nbytes = count * dtype.itemsize
    # Round up to page boundary so the buffer length is also page-aligned,
    # which Metal requires for newBufferWithBytesNoCopy.
    nbytes_aligned = ((nbytes + alignment - 1) // alignment) * alignment

    raw_ptr = _posix_memalign(nbytes_aligned, alignment)

    # Create a numpy array that shares the allocated memory.
    # We use ctypes to expose the raw pointer to numpy.
    ctypes_array = (ctypes.c_byte * nbytes_aligned).from_address(raw_ptr.value)
    arr = np.frombuffer(ctypes_array, dtype=dtype, count=count).reshape(shape)

    # Prevent the raw allocation from being freed while the array lives.
    # When the ref is dropped numpy will drop ctypes_array which does NOT
    # free the underlying posix_memalign memory (ctypes doesn't own it).
    # We attach a Release helper via the buffer owner chain instead.
    arr._aligned_raw_ptr = raw_ptr  # prevent GC
    arr._aligned_ctypes_buf = ctypes_array  # prevent GC

    # Register a weakref-free destructor using a ref-cycle-safe closure.
    import weakref

    def _free_cb(ptr_val=raw_ptr.value):
        _libc.free(ptr_val)

    # Invoke _free_cb when arr gets collected.
    weakref.ref(ctypes_array, lambda _: _free_cb())

    return arr
