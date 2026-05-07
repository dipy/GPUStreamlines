"""WebGPU GPU tracker — mirrors cuslines/metal/mt_tractography.py.

Key difference from Metal: no unified memory. GPU buffers require explicit
readbacks via device.queue.read_buffer() (similar to CUDA's cudaMemcpy).
"""

import numpy as np
import logging
from math import radians

from cuslines.webgpu.wgutils import (
    REAL_DTYPE,
    create_buffer_from_data,
)

from cuslines.webgpu.wg_direction_getters import WebGPUDirectionGetter, WebGPUBootDirectionGetter
from cuslines.webgpu.wg_propagate_seeds import WebGPUSeedBatchPropagator
from cuslines.generic_tracker import GenericTracker

logger = logging.getLogger("GPUStreamlines")


class WebGPUTracker(GenericTracker):
    def __init__(
        self,
        dg: WebGPUDirectionGetter,
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
        self.device = None  # created in __enter__

        # Ensure contiguous float32 arrays
        self.dataf = np.ascontiguousarray(dataf, dtype=REAL_DTYPE)
        self.metric_map = np.ascontiguousarray(stop_map, dtype=REAL_DTYPE)
        self.sphere_vertices = np.ascontiguousarray(sphere_vertices, dtype=REAL_DTYPE)
        self.sphere_edges = np.ascontiguousarray(sphere_edges, dtype=np.int32)

        self.dimx, self.dimy, self.dimz, self.dimt = dataf.shape
        self.nedges = int(sphere_edges.shape[0])
        if isinstance(dg, WebGPUBootDirectionGetter):
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

        # WebGPU: single GPU (ngpus ignored)
        self.ngpus = 1
        self.rng_seed = int(rng_seed)
        self.rng_offset = int(rng_offset)
        self.chunk_size = int(chunk_size)

        # GPU buffers — created in _allocate
        self.dataf_buf = None
        self.metric_map_buf = None
        self.sphere_vertices_buf = None
        self.sphere_edges_buf = None

        self.seed_propagator = WebGPUSeedBatchPropagator(
            gpu_tracker=self, minlen=min_pts, maxlen=max_pts
        )
        self._allocated = False

    def __enter__(self):
        self._allocate()
        return self

    def _setup_device(self):
        """Request a WebGPU adapter and device with required features."""
        import wgpu

        adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        )
        if adapter is None:
            raise RuntimeError("No WebGPU adapter found")

        # Subgroup operations are required by all kernels (shuffle, ballot, barrier)
        features = []
        if "subgroup" not in adapter.features:
            raise RuntimeError(
                "WebGPU adapter does not support subgroup operations. "
                "GPUStreamlines requires subgroups for SIMD-parallel reductions. "
                "Upgrade your GPU driver or use a different backend."
            )
        features.append("subgroup")
        if "subgroup-barrier" in adapter.features:
            features.append("subgroup-barrier")

        # Request adapter's maximum limits for buffer sizes and storage buffers.
        # Without this, the device gets WebGPU spec defaults (256 MB buffer,
        # 128 MB binding) which are too small for real-world diffusion MRI
        # datasets (e.g. HBN CSD with asymmetric ODFs can be ~5 GB).
        device = adapter.request_device_sync(
            required_features=features,
            required_limits={
                "max-storage-buffers-per-shader-stage": 17,
                "max-bind-groups": 4,
                "max-buffer-size": adapter.limits["max-buffer-size"],
                "max-storage-buffer-binding-size": adapter.limits[
                    "max-storage-buffer-binding-size"
                ],
            },
        )

        self.device = device
        self.has_subgroups = "subgroup" in features

        info = adapter.info
        max_buf_mb = device.limits["max-buffer-size"] / (1024 * 1024)
        logger.info(
            "WebGPU device: %s (backend: %s, subgroups: %s, max buffer: %.0f MB)",
            getattr(info, "device", "unknown"),
            getattr(info, "backend_type", "unknown"),
            self.has_subgroups,
            max_buf_mb,
        )

    def _allocate(self):
        if self._allocated:
            return

        self._setup_device()

        # Validate buffer sizes against device limits
        dataf_bytes = self.dataf.nbytes
        max_buf = self.device.limits["max-buffer-size"]
        max_binding = self.device.limits["max-storage-buffer-binding-size"]
        effective_max = min(max_buf, max_binding)
        if dataf_bytes > effective_max:
            raise RuntimeError(
                f"Input data ({dataf_bytes / 1e9:.1f} GB) exceeds WebGPU device "
                f"buffer limit ({effective_max / 1e9:.1f} GB). "
                f"Try a smaller volume, fewer ODF directions, or a GPU with more VRAM. "
                f"If using 'run_gpu_streamlines.py', consider setting "
                f"--sphere small"
            )

        # Upload static data arrays to GPU buffers
        try:
            self.dataf_buf = create_buffer_from_data(
                self.device, self.dataf.ravel(), label="dataf"
            )
            self.metric_map_buf = create_buffer_from_data(
                self.device, self.metric_map.ravel(), label="metric_map"
            )
            self.sphere_vertices_buf = create_buffer_from_data(
                self.device, self.sphere_vertices.ravel(), label="sphere_vertices"
            )
            self.sphere_edges_buf = create_buffer_from_data(
                self.device, self.sphere_edges.ravel(), label="sphere_edges"
            )

            self.dg.setup_device(self.device, self.has_subgroups, self.sphere_symm)
        except Exception:
            # Clean up any partially allocated buffers
            self.dataf_buf = None
            self.metric_map_buf = None
            self.sphere_vertices_buf = None
            self.sphere_edges_buf = None
            self.device = None
            raise
        self._allocated = True

    def __exit__(self, exc_type, exc, tb):
        logger.info("Destroying WebGPUTracker...")
        self.dataf_buf = None
        self.metric_map_buf = None
        self.sphere_vertices_buf = None
        self.sphere_edges_buf = None
        if hasattr(self.dg, "H_buf"):
            for attr in (
                "H_buf", "R_buf", "delta_b_buf", "delta_q_buf",
                "b0s_mask_buf", "sampling_matrix_buf",
            ):
                setattr(self.dg, attr, None)
        self.dg.shader_module = None
        self.dg.getnum_pipeline = None
        self.dg.gen_pipeline = None
        self.device = None
        self._allocated = False
        return False
