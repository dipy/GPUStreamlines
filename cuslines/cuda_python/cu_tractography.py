from cuda.bindings import runtime
from cuda.bindings.runtime import cudaMemcpyKind
# TODO: consider cuda core over cuda bindings

import numpy as np
from tqdm import tqdm
import logging
from math import radians

from cuslines.cuda_python.cutils import (
    REAL_SIZE,
    REAL_DTYPE,
    checkCudaErrors,
)
from cuslines.cuda_python.cu_direction_getters import (
    GPUDirectionGetter,
    BootDirectionGetter,
)
from cuslines.cuda_python.cu_propagate_seeds import SeedBatchPropagator

from trx.trx_file_memmap import TrxFile

from nibabel.streamlines.tractogram import Tractogram
from nibabel.streamlines.array_sequence import ArraySequence, MEGABYTE

from dipy.io.stateful_tractogram import Space, StatefulTractogram

logger = logging.getLogger("GPUStreamlines")

# TODO performance:
# ACT
# SCIL streamline reduction onboard GPU
# Remove small/long streamlines on gpu


class GPUTracker:
    def __init__(
        self,
        dg: GPUDirectionGetter,
        dataf: np.ndarray,
        stop_map: np.ndarray,
        stop_theshold: float,
        sphere_vertices: np.ndarray,
        sphere_edges: np.ndarray,
        max_angle: float = radians(60),
        step_size: float = 0.5,
        relative_peak_thresh: float = 0.25,
        min_separation_angle: float = radians(45),
        ngpus: int = 1,
        rng_seed: int = 0,
        rng_offset: int = 0,
        chunk_size: int = 25000,
    ):
        """
        Initialize GPUTracker with necessary data.

        Parameters
        ----------
        dg : GPUDirectionGetter
            Direction getter to use for tracking from
            cuslines.cu_direction_getters
        dataf : np.ndarray
            4D numpy array with ODFs for prob/ptt, diffusion data if doing
            bootstrapping.
        stop_map : np.ndarray
            3D numpy array with stopping metric (e.g., GFA, FA)
        stop_theshold : float
            Threshold for stopping metric (e.g., 0.2)
        sphere_vertices : np.ndarray
            Vertices of the sphere used for direction sampling.
        sphere_edges : np.ndarray
            Edges of the sphere used for direction sampling.
        max_angle : float, optional
            Maximum angle (in radians) between steps
            default: radians(60)
        step_size : float, optional
            Step size for tracking
            default: 0.5
        relative_peak_thresh : float, optional
            Relative peak threshold for direction selection
            default: 0.25
        min_separation_angle : float, optional
            Minimum separation angle (in radians) between peaks
            default: radians(45)
        ngpus : int, optional
            Number of GPUs to use
            default: 1
        rng_seed : int, optional
            Seed for random number generator
            default: 0
        rng_offset : int, optional
            Offset for random number generator
            default: 0
        chunk_size : int, optional
            Number of seeds to process in each chunk per GPU
            default: 25000
        """
        self.dataf = np.ascontiguousarray(dataf, dtype=REAL_DTYPE)
        self.metric_map = np.ascontiguousarray(stop_map, dtype=REAL_DTYPE)
        self.sphere_vertices = np.ascontiguousarray(sphere_vertices, dtype=REAL_DTYPE)
        self.sphere_edges = np.ascontiguousarray(sphere_edges, dtype=np.int32)

        self.dimx, self.dimy, self.dimz, self.dimt = dataf.shape
        self.nedges = int(sphere_edges.shape[0])
        if isinstance(dg, BootDirectionGetter):
            self.samplm_nr = int(dg.sampling_matrix.shape[0])
        else:
            self.samplm_nr = self.dimt
        self.n32dimt = ((self.dimt + 31) // 32) * 32

        self.dg = dg
        self.max_angle = REAL_DTYPE(max_angle)
        self.tc_threshold = REAL_DTYPE(stop_theshold)
        self.step_size = REAL_DTYPE(step_size)
        self.relative_peak_thresh = REAL_DTYPE(relative_peak_thresh)
        self.min_separation_angle = REAL_DTYPE(min_separation_angle)

        self.ngpus = int(ngpus)
        self.rng_seed = int(rng_seed)
        self.rng_offset = int(rng_offset)
        self.chunk_size = int(chunk_size)

        avail = checkCudaErrors(runtime.cudaGetDeviceCount())
        if self.ngpus > avail:
            raise RuntimeError(
                f"Requested {self.ngpus} GPUs but only {avail} available"
            )

        logger.info("Creating GPUTracker with %d GPUs...", self.ngpus)

        self.dataf_d = []
        self.metric_map_d = []
        self.sphere_vertices_d = []
        self.sphere_edges_d = []

        self.streams = []
        self.managed_data = []

        self.seed_propagator = SeedBatchPropagator(gpu_tracker=self)
        self._allocated = False

    def __enter__(self):
        self._allocate()
        return self

    def _allocate(self):
        if self._allocated:
            return

        for ii in range(self.ngpus):
            checkCudaErrors(runtime.cudaSetDevice(ii))
            self.streams.append(
                checkCudaErrors(
                    runtime.cudaStreamCreateWithFlags(runtime.cudaStreamNonBlocking)
                )
            )

        for ii in range(self.ngpus):
            checkCudaErrors(runtime.cudaSetDevice(ii))

            # TODO: performance: dataf could be managed or texture memory instead?
            self.dataf_d.append(
                checkCudaErrors(runtime.cudaMalloc(REAL_SIZE * self.dataf.size))
            )
            self.metric_map_d.append(
                checkCudaErrors(runtime.cudaMalloc(REAL_SIZE * self.metric_map.size))
            )
            self.sphere_vertices_d.append(
                checkCudaErrors(
                    runtime.cudaMalloc(REAL_SIZE * self.sphere_vertices.size)
                )
            )
            self.sphere_edges_d.append(
                checkCudaErrors(
                    runtime.cudaMalloc(np.int32().nbytes * self.sphere_edges.size)
                )
            )

            checkCudaErrors(
                runtime.cudaMemcpy(
                    self.dataf_d[ii],
                    self.dataf.ctypes.data,
                    REAL_SIZE * self.dataf.size,
                    cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )
            checkCudaErrors(
                runtime.cudaMemcpy(
                    self.metric_map_d[ii],
                    self.metric_map.ctypes.data,
                    REAL_SIZE * self.metric_map.size,
                    cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )
            checkCudaErrors(
                runtime.cudaMemcpy(
                    self.sphere_vertices_d[ii],
                    self.sphere_vertices.ctypes.data,
                    REAL_SIZE * self.sphere_vertices.size,
                    cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )
            checkCudaErrors(
                runtime.cudaMemcpy(
                    self.sphere_edges_d[ii],
                    self.sphere_edges.ctypes.data,
                    np.int32().nbytes * self.sphere_edges.size,
                    cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )
            self.dg.allocate_on_gpu(ii)

        self._allocated = True

    def __exit__(self, exc_type, exc, tb):
        logger.info("Destroying GPUTracker and freeing GPU memory...")

        for n in range(self.ngpus):
            checkCudaErrors(runtime.cudaSetDevice(n))
            if self.dataf_d[n]:
                checkCudaErrors(runtime.cudaFree(self.dataf_d[n]))
            if self.metric_map_d[n]:
                checkCudaErrors(runtime.cudaFree(self.metric_map_d[n]))
            if self.sphere_vertices_d[n]:
                checkCudaErrors(runtime.cudaFree(self.sphere_vertices_d[n]))
            if self.sphere_edges_d[n]:
                checkCudaErrors(runtime.cudaFree(self.sphere_edges_d[n]))
            self.dg.deallocate_on_gpu(n)

            checkCudaErrors(runtime.cudaStreamDestroy(self.streams[n]))
        return False

    def _divide_chunks(self, seeds):
        global_chunk_sz = self.chunk_size * self.ngpus
        nchunks = (seeds.shape[0] + global_chunk_sz - 1) // global_chunk_sz
        return global_chunk_sz, nchunks

    def generate_sft(self, seeds, ref_img):
        global_chunk_sz, nchunks = self._divide_chunks(seeds)
        buffer_size = 0
        generators = []

        with tqdm(total=seeds.shape[0]) as pbar:
            for idx in range(nchunks):
                self.seed_propagator.propagate(
                    seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz]
                )
                buffer_size += self.seed_propagator.get_buffer_size()
                generators.append(self.seed_propagator.as_generator())
                pbar.update(
                    seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz].shape[0]
                )
        array_sequence = ArraySequence(
            (item for gen in generators for item in gen), buffer_size
        )
        return StatefulTractogram(array_sequence, ref_img, Space.VOX)

    def generate_trx(self, seeds, ref_img):
        global_chunk_sz, nchunks = self._divide_chunks(seeds)

        # Will resize by a factor of 2 if these are exceeded
        sl_len_guess = 100
        sl_per_seed_guess = 4
        n_sls_guess = sl_per_seed_guess * seeds.shape[0]

        # trx files use memory mapping
        trx_reference = TrxFile(
            reference=ref_img
        )
        trx_reference.streamlines._data = trx_file.streamlines._data.astype(np.float32)
        trx_reference.streamlines._offsets = trx_file.streamlines._offsets.astype(np.uint64)

        trx_file = TrxFile(
            nb_streamlines=n_sls_guess,
            nb_vertices=n_sls_guess * sl_len_guess,
            init_as=trx_reference
        )
        offsets_idx = 0
        sls_data_idx = 0

        with tqdm(total=seeds.shape[0]) as pbar:
            for idx in range(int(nchunks)):
                self.seed_propagator.propagate(
                    seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz]
                )
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

                # TRX uses memmaps here
                trx_file.streamlines._data[sls_data_idx:new_sls_data_idx] = sls._data
                trx_file.streamlines._offsets[offsets_idx:new_offsets_idx] = (
                    sls_data_idx + sls._offsets
                )
                trx_file.streamlines._lengths[offsets_idx:new_offsets_idx] = (
                    sls._lengths
                )

                offsets_idx = new_offsets_idx
                sls_data_idx = new_sls_data_idx
                pbar.update(
                    seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz].shape[0]
                )
        trx_file.resize()

        return trx_file
