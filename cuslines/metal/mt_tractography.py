"""Metal GPU tracker — mirrors cuslines/cuda_python/cu_tractography.py.

Key difference from the CUDA backend: Apple Silicon unified memory means
we wrap numpy arrays as Metal shared buffers with zero copies.
"""

import numpy as np
from tqdm import tqdm
import logging
from math import radians

from cuslines.metal.mutils import (
    REAL_SIZE,
    REAL_DTYPE,
    aligned_array,
    PAGE_SIZE,
    checkMetalError,
)

from cuslines.metal.mt_direction_getters import MetalGPUDirectionGetter, MetalBootDirectionGetter
from cuslines.metal.mt_propagate_seeds import MetalSeedBatchPropagator

from trx.trx_file_memmap import TrxFile
from nibabel.streamlines.tractogram import Tractogram
from nibabel.streamlines.array_sequence import ArraySequence, MEGABYTE
from dipy.io.stateful_tractogram import Space, StatefulTractogram

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


class MetalGPUTracker:
    def __init__(
        self,
        dg: MetalGPUDirectionGetter,
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
        self.max_angle = np.float32(max_angle)
        self.tc_threshold = np.float32(stop_theshold)
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

        # Unified memory: wrap numpy arrays as shared Metal buffers
        self.dataf_buf = _make_shared_buffer(self.device, self.dataf)
        self.metric_map_buf = _make_shared_buffer(self.device, self.metric_map)
        self.sphere_vertices_buf = _make_shared_buffer(self.device, self.sphere_vertices)
        self.sphere_edges_buf = _make_shared_buffer(self.device, self.sphere_edges)

        self.dg.setup_device(self.device)
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
        trx_reference.streamlines._data = trx_reference.streamlines._data.astype(np.float32)
        trx_reference.streamlines._offsets = trx_reference.streamlines._offsets.astype(np.uint64)

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
