"""Metal GPU tracker — mirrors cuslines/cuda_python/cu_tractography.py.

Key difference from the CUDA backend: Apple Silicon unified memory means
we wrap numpy arrays as Metal shared buffers with zero copies.
"""

import numpy as np
import logging
from math import radians

from cuslines.metal.mutils import (
    REAL_DTYPE,
)

from cuslines.metal.mt_direction_getters import MetalGPUDirectionGetter, MetalBootDirectionGetter
from cuslines.metal.mt_propagate_seeds import MetalSeedBatchPropagator
from cuslines.generic_tracker import GenericTracker

logger = logging.getLogger("GPUStreamlines")


def _make_shared_buffer(device, arr):
    """Copy a numpy array into a Metal shared buffer.

    Uses newBufferWithBytes (one copy at setup time). The buffer lives in
    unified memory and is GPU-accessible without further copies.
    """
    import Metal

    buf = device.newBufferWithBytes_length_options_(
        arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared
    )
    return buf


def _make_dynamic_buffer(device, nbytes):
    """Create an empty Metal shared buffer and return (buf, numpy_view).

    The numpy array is a writable view of the Metal buffer's contents,
    giving true zero-copy CPU/GPU sharing for dynamic per-batch data.
    """
    import Metal

    buf = device.newBufferWithLength_options_(
        nbytes, Metal.MTLResourceStorageModeShared
    )
    return buf


def _buffer_as_array(buf, dtype, shape):
    """Create a numpy array view of a Metal buffer's contents (zero-copy)."""
    nbytes = buf.length()
    memview = buf.contents().as_buffer(nbytes)
    count = int(np.prod(shape))
    return np.frombuffer(memview, dtype=dtype, count=count).reshape(shape)


class MetalGPUTracker(GenericTracker):
    def __init__(
        self,
        dg: MetalGPUDirectionGetter,
        dataf: np.ndarray,
        stop_map: np.ndarray,
        stop_threshold: float,
        sphere_vertices: np.ndarray,
        sphere_edges: np.ndarray,
        sphere_symm: bool = False,
        max_angle: float = radians(60),
        step_size: float = 0.5,
        min_pts=0,
        max_pts=np.inf,
        relative_peak_thresh: float = 0.25,
        min_separation_angle: float = radians(45),
        ngpus: int = 1,
        rng_seed: int = 0,
        rng_offset: int = 0,
        chunk_size: int = 25000,
    ):
        import Metal

        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal GPU device found")
        self.command_queue = self.device.newCommandQueue()

        # Ensure contiguous float32 arrays
        self.dataf = np.ascontiguousarray(dataf, dtype=REAL_DTYPE)
        self.metric_map = np.ascontiguousarray(stop_map, dtype=REAL_DTYPE)
        self.sphere_vertices = np.ascontiguousarray(sphere_vertices, dtype=REAL_DTYPE)
        self.sphere_edges = np.ascontiguousarray(sphere_edges, dtype=np.int32)

        self.dimx, self.dimy, self.dimz, self.dimt = dataf.shape
        self.nedges = int(sphere_edges.shape[0])
        if isinstance(dg, MetalBootDirectionGetter):
            self.samplm_nr = int(dg.sampling_matrix.shape[0])
        else:
            self.samplm_nr = self.dimt
        self.n32dimt = ((self.dimt + 31) // 32) * 32

        self.dg = dg
        self.sphere_symm = bool(sphere_symm)
        self.max_angle = np.float32(max_angle)
        self.tc_threshold = np.float32(stop_threshold)
        self.step_size = np.float32(step_size)
        self.relative_peak_thresh = np.float32(relative_peak_thresh)
        self.min_separation_angle = np.float32(min_separation_angle)

        # Metal: single GPU (ngpus ignored, always 1)
        self.ngpus = 1
        self.rng_seed = int(rng_seed)
        self.rng_offset = int(rng_offset)
        self.chunk_size = int(chunk_size)

        logger.info("Creating MetalGPUTracker on %s", self.device.name())

        # Shared buffers — created lazily in __enter__
        self.dataf_buf = None
        self.metric_map_buf = None
        self.sphere_vertices_buf = None
        self.sphere_edges_buf = None

        self.seed_propagator = MetalSeedBatchPropagator(
            gpu_tracker=self, minlen=min_pts, maxlen=max_pts
        )
        self._allocated = False

    def __enter__(self):
        self._allocate()
        return self

    def _allocate(self):
        if self._allocated:
            return

        # Validate buffer size against device limit
        dataf_bytes = self.dataf.nbytes
        max_buf = self.device.maxBufferLength()
        if dataf_bytes > max_buf:
            raise RuntimeError(
                f"Input data ({dataf_bytes / 1e9:.1f} GB) exceeds Metal device "
                f"buffer limit ({max_buf / 1e9:.1f} GB). "
                f"Try a smaller volume or fewer ODF directions."
            )

        # Unified memory: wrap numpy arrays as shared Metal buffers
        self.dataf_buf = _make_shared_buffer(self.device, self.dataf)
        self.metric_map_buf = _make_shared_buffer(self.device, self.metric_map)
        self.sphere_vertices_buf = _make_shared_buffer(self.device, self.sphere_vertices)
        self.sphere_edges_buf = _make_shared_buffer(self.device, self.sphere_edges)

        self.dg.setup_device(self.device, self.sphere_symm)
        self._allocated = True

    def __exit__(self, exc_type, exc, tb):
        logger.info("Destroying MetalGPUTracker...")
        # Metal buffers are reference-counted; dropping refs is sufficient.
        self.dataf_buf = None
        self.metric_map_buf = None
        self.sphere_vertices_buf = None
        self.sphere_edges_buf = None
        # Clean up direction getter buffers
        if hasattr(self.dg, 'H_buf'):
            for attr in ('H_buf', 'R_buf', 'delta_b_buf', 'delta_q_buf',
                          'b0s_mask_buf', 'sampling_matrix_buf'):
                setattr(self.dg, attr, None)
        self.dg.library = None
        self.dg.getnum_pipeline = None
        self.dg.gen_pipeline = None
        self._allocated = False
        return False
