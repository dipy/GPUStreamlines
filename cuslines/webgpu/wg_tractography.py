"""WebGPU GPU tracker — mirrors cuslines/metal/mt_tractography.py.

Key difference from Metal: no unified memory. GPU buffers require explicit
readbacks via device.queue.read_buffer() (similar to CUDA's cudaMemcpy).
"""

import numpy as np
from tqdm import tqdm
import logging
from math import radians

from cuslines.webgpu.wgutils import (
    REAL_SIZE,
    REAL_DTYPE,
    create_buffer_from_data,
)

from cuslines.webgpu.wg_direction_getters import WebGPUDirectionGetter, WebGPUBootDirectionGetter
from cuslines.webgpu.wg_propagate_seeds import WebGPUSeedBatchPropagator

from trx.trx_file_memmap import TrxFile
from nibabel.streamlines.tractogram import Tractogram
from nibabel.streamlines.array_sequence import ArraySequence, MEGABYTE
from dipy.io.stateful_tractogram import Space, StatefulTractogram

logger = logging.getLogger("GPUStreamlines")


class WebGPUTracker:
    def __init__(
        self,
        dg: WebGPUDirectionGetter,
        dataf: np.ndarray,
        stop_map: np.ndarray,
        stop_theshold: float,
        sphere_vertices: np.ndarray,
        sphere_edges: np.ndarray,
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
        self.max_angle = np.float32(max_angle)
        self.tc_threshold = np.float32(stop_theshold)
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

        # Request sufficient storage buffer count for boot kernels (17)
        device = adapter.request_device_sync(
            required_features=features,
            required_limits={
                "max-storage-buffers-per-shader-stage": 17,
                "max-bind-groups": 4,
            },
        )

        self.device = device
        self.has_subgroups = "subgroup" in features

        info = adapter.info
        logger.info(
            "WebGPU device: %s (backend: %s, subgroups: %s)",
            getattr(info, "device", "unknown"),
            getattr(info, "backend_type", "unknown"),
            self.has_subgroups,
        )

    def _allocate(self):
        if self._allocated:
            return

        self._setup_device()

        # Upload static data arrays to GPU buffers
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

        self.dg.setup_device(self.device, self.has_subgroups)
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

    def _divide_chunks(self, seeds):
        global_chunk_sz = self.chunk_size  # single GPU
        nchunks = (seeds.shape[0] + global_chunk_sz - 1) // global_chunk_sz
        return global_chunk_sz, nchunks

    def generate_sft(self, seeds, ref_img):
        global_chunk_sz, nchunks = self._divide_chunks(seeds)
        buffer_size = 0
        generators = []

        with tqdm(total=seeds.shape[0]) as pbar:
            for idx in range(nchunks):
                chunk = seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz]
                self.seed_propagator.propagate(chunk)
                buffer_size += self.seed_propagator.get_buffer_size()
                generators.append(self.seed_propagator.as_generator())
                pbar.update(chunk.shape[0])

        array_sequence = ArraySequence(
            (item for gen in generators for item in gen), buffer_size
        )
        return StatefulTractogram(array_sequence, ref_img, Space.VOX)

    def generate_trx(self, seeds, ref_img):
        global_chunk_sz, nchunks = self._divide_chunks(seeds)

        sl_len_guess = 100
        sl_per_seed_guess = 2
        n_sls_guess = sl_per_seed_guess * seeds.shape[0]

        trx_reference = TrxFile(reference=ref_img)
        trx_reference.streamlines._data = trx_reference.streamlines._data.astype(
            np.float32
        )
        trx_reference.streamlines._offsets = trx_reference.streamlines._offsets.astype(
            np.uint64
        )

        trx_file = TrxFile(
            nb_streamlines=n_sls_guess,
            nb_vertices=n_sls_guess * sl_len_guess,
            init_as=trx_reference,
        )
        offsets_idx = 0
        sls_data_idx = 0

        with tqdm(total=seeds.shape[0]) as pbar:
            for idx in range(int(nchunks)):
                chunk = seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz]
                self.seed_propagator.propagate(chunk)
                tractogram = Tractogram(
                    self.seed_propagator.as_array_sequence(),
                    affine_to_rasmm=ref_img.affine,
                )
                tractogram.to_world()
                sls = tractogram.streamlines

                new_offsets_idx = offsets_idx + len(sls._offsets)
                new_sls_data_idx = sls_data_idx + len(sls._data)

                if (
                    new_offsets_idx > trx_file.header["NB_STREAMLINES"]
                    or new_sls_data_idx > trx_file.header["NB_VERTICES"]
                ):
                    logger.info("TRX resizing...")
                    trx_file.resize(
                        nb_streamlines=new_offsets_idx * 2,
                        nb_vertices=new_sls_data_idx * 2,
                    )

                trx_file.streamlines._data[sls_data_idx:new_sls_data_idx] = sls._data
                trx_file.streamlines._offsets[offsets_idx:new_offsets_idx] = (
                    sls_data_idx + sls._offsets
                )
                trx_file.streamlines._lengths[offsets_idx:new_offsets_idx] = sls._lengths

                offsets_idx = new_offsets_idx
                sls_data_idx = new_sls_data_idx
                pbar.update(chunk.shape[0])

        trx_file.resize()
        return trx_file
