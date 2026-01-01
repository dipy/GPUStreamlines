from cuda.bindings import driver, runtime
# TODO: consider cuda core over cuda bindings

import numpy as np
import logging

from cutils import (
    REAL_SIZE,
    REAL_DTYPE,
    checkCudaErrors,
)
from cu_direction_getters import (
    GPUDirectionGetter,
    BootDirectionGetter
)
from cu_propagate_seeds import SeedBatchPropagator


logger = logging.getLogger("GPUStreamlines")

# TODO: we need to organize this package into folders, then make it pip installable.
# but should merge in PTT FIRST
class GPUTracker:  # TODO: bring in pyAFQ prep stuff
    def __init__(
        self,
        dg: GPUDirectionGetter,
        max_angle: float,
        tc_threshold: float,
        step_size: float,
        relative_peak_thresh: float,
        min_separation_angle: float,
        dataf: np.ndarray, # TODO: reasonable defaults for floats, reorganize order, better names, documentation
        metric_map: np.ndarray,
        sphere_vertices: np.ndarray,
        sphere_edges: np.ndarray,
        ngpus: int = 1,
        rng_seed: int = 0,
        rng_offset: int = 0,
    ):
        for name, arr, dt in [
            ("dataf", dataf, REAL_DTYPE),
            ("metric_map", metric_map, REAL_DTYPE),
            ("sphere_vertices", sphere_vertices, REAL_DTYPE),
            ("sphere_edges", sphere_edges, np.int32),
        ]:
            if arr.dtype != dt:
                raise TypeError(f"{name} must have dtype {dt}, got {arr.dtype}")
            if not arr.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")

        self.dataf = dataf
        self.metric_map = metric_map
        self.sphere_vertices = sphere_vertices
        self.sphere_edges = sphere_edges

        self.dimx, self.dimy, self.dimz, self.dimt = dataf.shape
        self.nedges = int(sphere_edges.shape[0])
        if isinstance(dg, BootDirectionGetter):
            self.samplm_nr = int(dg.sampling_matrix.shape[0])
        else:
            self.samplm_nr = self.dimt

        self.dg = dg
        self.max_angle = REAL_DTYPE(max_angle)
        self.tc_threshold = REAL_DTYPE(tc_threshold)
        self.step_size = REAL_DTYPE(step_size)
        self.relative_peak_thresh = REAL_DTYPE(relative_peak_thresh)
        self.min_separation_angle = REAL_DTYPE(min_separation_angle)

        self.ngpus = int(ngpus)
        self.rng_seed = int(rng_seed)
        self.rng_offset = int(rng_offset)

        checkCudaErrors(driver.cuInit(0))
        avail = checkCudaErrors(runtime.cudaGetDeviceCount())
        if self.ngpus > avail:
            raise RuntimeError(f"Requested {self.ngpus} GPUs but only {avail} available")

        logger.info("Creating GPUTracker with %d GPUs...", self.ngpus)

        self.dataf_d = []
        self.metric_map_d = []
        self.sphere_vertices_d = []
        self.sphere_edges_d = []

        self.seed_propagator = SeedBatchPropagator(
            gpu_tracker=self)
        self._allocated = False

    def __enter__(self):
        self._allocate()
        return self

    def _allocate(self):
        if self._allocated:
            return

        for ii in range(self.ngpus):
            checkCudaErrors(runtime.cudaSetDevice(ii))
            self.dataf_d.append( # TODO: put this in texture memory?
                checkCudaErrors(runtime.cudaMallocManaged(  # TODO: look at cuda core managed memory
                    REAL_SIZE*self.dataf.size, 
                    runtime.cudaMemAttachGlobal)))
            checkCudaErrors(runtime.cudaMemAdvise(
                self.dataf_d[ii],
                REAL_SIZE*self.dataf.size,
                runtime.cudaMemAdviseSetPreferredLocation,
                ii))
            self.metric_map_d.append(
                checkCudaErrors(runtime.cudaMalloc(
                    REAL_SIZE*self.metric_map.size)))
            self.sphere_vertices_d.append(
                checkCudaErrors(runtime.cudaMalloc(
                    REAL_SIZE*self.sphere_vertices.size)))
            self.sphere_edges_d.append(
                checkCudaErrors(runtime.cudaMalloc(
                    np.int32().nbytes*self.sphere_edges.size)))
            
            checkCudaErrors(runtime.cudaMemcpy(
                self.dataf_d[ii],
                self.dataf.ctypes.data,
                REAL_SIZE*self.dataf.size,
                runtime.cudaMemcpyHostToDevice))
            checkCudaErrors(runtime.cudaMemcpy(
                self.metric_map_d[ii],
                self.metric_map.ctypes.data,
                REAL_SIZE*self.metric_map.size,
                runtime.cudaMemcpyHostToDevice))
            checkCudaErrors(runtime.cudaMemcpy(
                self.sphere_vertices_d[ii],
                self.sphere_vertices.ctypes.data,
                REAL_SIZE*self.sphere_vertices.size,
                runtime.cudaMemcpyHostToDevice))
            checkCudaErrors(runtime.cudaMemcpy(
                self.sphere_edges_d[ii],
                self.sphere_edges.ctypes.data,
                np.int32().nbytes*self.sphere_edges.size,
                runtime.cudaMemcpyHostToDevice))
            
            self.dg.allocate_on_gpu(ii)

        self.streams = []
        for ii in range(self.ngpus):
            checkCudaErrors(runtime.cudaSetDevice(ii))
            self.streams.append(
                checkCudaErrors(runtime.cudaStreamCreateWithFlags(
                    runtime.cudaStreamNonBlocking)))

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

            if self.seed_propagator.sline_lens[n]:
                checkCudaErrors(runtime.cudaFreeHost(
                    self.seed_propagator.sline_lens[n]))
            if self.seed_propagator.slines[n]:
                checkCudaErrors(runtime.cudaFreeHost(
                    self.seed_propagator.slines[n]))
                
            self.dg.deallocate_on_gpu(n)

            checkCudaErrors(runtime.cudaStreamDestroy(self.streams[n]))
        return False

    def generate_streamlines(self, seeds):
        self.seed_propagator.propagate(seeds)
        return self.seed_propagator.as_array_sequence()
