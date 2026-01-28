import numpy as np
import math
import gc
from cuda.bindings import runtime
from cuda.bindings.runtime import cudaMemcpyKind

from nibabel.streamlines.array_sequence import ArraySequence, MEGABYTE
import logging

from cuslines.cuda_python.cutils import (
    REAL_SIZE,
    REAL_DTYPE,
    REAL3_DTYPE,
    MAX_SLINE_LEN,
    EXCESS_ALLOC_FACT,
    THR_X_SL,
    THR_X_BL,
    DEV_PTR,
    div_up,
    checkCudaErrors,
)


logger = logging.getLogger("GPUStreamlines")


class SeedBatchPropagator:
    def __init__(self, gpu_tracker):
        self.gpu_tracker = gpu_tracker
        self.ngpus = gpu_tracker.ngpus

        self.nSlines_old = np.zeros(self.ngpus, dtype=np.int32)
        self.nSlines = np.zeros(self.ngpus, dtype=np.int32)
        self.slines = [None] * self.ngpus
        self.sline_lens = [None] * self.ngpus

        self.seeds_d = np.empty(self.ngpus, dtype=DEV_PTR)
        self.slineSeed_d = np.empty(self.ngpus, dtype=DEV_PTR)
        self.slinesOffs_d = np.empty(self.ngpus, dtype=DEV_PTR)
        self.shDirTemp0_d = np.empty(self.ngpus, dtype=DEV_PTR)
        self.slineLen_d = np.empty(self.ngpus, dtype=DEV_PTR)
        self.sline_d = np.empty(self.ngpus, dtype=DEV_PTR)

    def _switch_device(self, n):
        checkCudaErrors(runtime.cudaSetDevice(n))

        nseeds_gpu = min(
            self.nseeds_per_gpu, max(0, self.nseeds - n * self.nseeds_per_gpu)
        )
        block = (THR_X_SL, THR_X_BL // THR_X_SL, 1)
        grid = (div_up(nseeds_gpu, THR_X_BL // THR_X_SL), 1, 1)

        return nseeds_gpu, block, grid

    def _get_sl_buffer_size(self, n):
        return REAL_SIZE * 2 * 3 * MAX_SLINE_LEN * self.nSlines[n].astype(np.int64)

    def _allocate_seed_memory(self, seeds):
        # Move seeds to GPU
        for ii in range(self.ngpus):
            nseeds_gpu, _, _ = self._switch_device(ii)
            self.seeds_d[ii] = checkCudaErrors(
                runtime.cudaMalloc(REAL_SIZE * 3 * nseeds_gpu)
            )
            seeds_host = np.ascontiguousarray(
                seeds[ii * self.nseeds_per_gpu : ii * self.nseeds_per_gpu + nseeds_gpu],
                dtype=REAL_DTYPE,
            )
            checkCudaErrors(
                runtime.cudaMemcpy(
                    self.seeds_d[ii],
                    seeds_host.ctypes.data,
                    REAL_SIZE * 3 * nseeds_gpu,
                    cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )

        for ii in range(self.ngpus):
            nseeds_gpu, block, grid = self._switch_device(ii)
            # Streamline offsets
            self.slinesOffs_d[ii] = checkCudaErrors(
                runtime.cudaMalloc(np.int32().nbytes * (nseeds_gpu + 1))
            )
            # Initial directions from each seed
            self.shDirTemp0_d[ii] = checkCudaErrors(
                runtime.cudaMalloc(
                    REAL3_DTYPE.itemsize
                    * self.gpu_tracker.samplm_nr
                    * grid[0]
                    * block[1]
                )
            )

    def _cumsum_offsets(
        self,
    ):  # TODO: performance: do this on device? not crucial for performance now
        for ii in range(self.ngpus):
            nseeds_gpu, _, _ = self._switch_device(ii)
            if nseeds_gpu == 0:
                self.nSlines[ii] = 0
                continue

            slinesOffs_h = np.empty(nseeds_gpu + 1, dtype=np.int32)
            checkCudaErrors(
                runtime.cudaMemcpy(
                    slinesOffs_h.ctypes.data,
                    self.slinesOffs_d[ii],
                    slinesOffs_h.nbytes,
                    cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
            )

            __pval = slinesOffs_h[0]
            slinesOffs_h[0] = 0
            for jj in range(1, nseeds_gpu + 1):
                __cval = slinesOffs_h[jj]
                slinesOffs_h[jj] = slinesOffs_h[jj - 1] + __pval
                __pval = __cval
            self.nSlines[ii] = int(slinesOffs_h[nseeds_gpu])

            checkCudaErrors(
                runtime.cudaMemcpy(
                    self.slinesOffs_d[ii],
                    slinesOffs_h.ctypes.data,
                    slinesOffs_h.nbytes,
                    cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )

    def _allocate_tracking_memory(self):
        for ii in range(self.ngpus):
            self._switch_device(ii)

            self.slineSeed_d[ii] = checkCudaErrors(
                runtime.cudaMalloc(self.nSlines[ii] * np.int32().nbytes)
            )
            checkCudaErrors(
                runtime.cudaMemset(
                    self.slineSeed_d[ii], -1, self.nSlines[ii] * np.int32().nbytes
                )
            )

            if self.nSlines[ii] > EXCESS_ALLOC_FACT * self.nSlines_old[ii]:
                self.slines[ii] = None
                self.sline_lens[ii] = None
                gc.collect()

            buffer_size = self._get_sl_buffer_size(ii)
            logger.debug(f"Streamline buffer size: {buffer_size}")

            if self.slines[ii] is None:
                self.slines[ii] = np.empty(
                    (EXCESS_ALLOC_FACT * self.nSlines[ii], MAX_SLINE_LEN * 2, 3),
                    dtype=REAL_DTYPE,
                )
            if self.sline_lens[ii] is None:
                self.sline_lens[ii] = np.empty(
                    EXCESS_ALLOC_FACT * self.nSlines[ii], dtype=np.int32
                )

        for ii in range(self.ngpus):
            self._switch_device(ii)
            buffer_size = self._get_sl_buffer_size(ii)

            self.slineLen_d[ii] = checkCudaErrors(
                runtime.cudaMalloc(np.int32().nbytes * self.nSlines[ii])
            )
            self.sline_d[ii] = checkCudaErrors(runtime.cudaMalloc(buffer_size))

    def _cleanup(self):
        for ii in range(self.ngpus):
            self._switch_device(ii)
            checkCudaErrors(
                runtime.cudaMemcpyAsync(
                    self.slines[ii],
                    self.sline_d[ii],
                    self._get_sl_buffer_size(ii),
                    cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.gpu_tracker.streams[ii],
                )
            )
            checkCudaErrors(
                runtime.cudaMemcpyAsync(
                    self.sline_lens[ii],
                    self.slineLen_d[ii],
                    np.int32().nbytes * self.nSlines[ii],
                    cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.gpu_tracker.streams[ii],
                )
            )

        for ii in range(self.ngpus):
            self._switch_device(ii)
            checkCudaErrors(runtime.cudaStreamSynchronize(self.gpu_tracker.streams[ii]))
            checkCudaErrors(runtime.cudaFree(self.seeds_d[ii]))
            checkCudaErrors(runtime.cudaFree(self.slineSeed_d[ii]))
            checkCudaErrors(runtime.cudaFree(self.slinesOffs_d[ii]))
            checkCudaErrors(runtime.cudaFree(self.shDirTemp0_d[ii]))
            checkCudaErrors(runtime.cudaFree(self.slineLen_d[ii]))
            checkCudaErrors(runtime.cudaFree(self.sline_d[ii]))

        self.nSlines_old = self.nSlines
        self.gpu_tracker.rng_offset += self.nseeds

    # TODO: performance: better queuing/batching of seeds,
    # if more performance needed,
    # given exponential nature of streamlines
    # May be better to do in cuda code directly
    def propagate(self, seeds):
        self.nseeds = len(seeds)
        self.nseeds_per_gpu = (
            self.nseeds + self.gpu_tracker.ngpus - 1
        ) // self.gpu_tracker.ngpus

        self._allocate_seed_memory(seeds)

        for ii in range(self.ngpus):
            nseeds_gpu, block, grid = self._switch_device(ii)
            if nseeds_gpu == 0:
                continue
            self.gpu_tracker.dg.getNumStreamlines(ii, nseeds_gpu, block, grid, self)
        for ii in range(self.ngpus):
            checkCudaErrors(runtime.cudaStreamSynchronize(self.gpu_tracker.streams[ii]))

        self._cumsum_offsets()
        self._allocate_tracking_memory()

        for ii in range(self.ngpus):
            nseeds_gpu, block, grid = self._switch_device(ii)
            if nseeds_gpu == 0:
                continue
            self.gpu_tracker.dg.generateStreamlines(ii, nseeds_gpu, block, grid, self)
        for ii in range(self.ngpus):
            checkCudaErrors(runtime.cudaStreamSynchronize(self.gpu_tracker.streams[ii]))

        self._cleanup()

    def get_buffer_size(self):
        buffer_size = 0
        for ii in range(self.ngpus):
            lens = self.sline_lens[ii]
            for jj in range(self.nSlines[ii]):
                buffer_size += lens[jj] * 3 * REAL_SIZE
        return math.ceil(buffer_size / MEGABYTE)

    def as_generator(self):
        def _yield_slines():
            for ii in range(self.ngpus):
                this_sls = self.slines[ii]
                this_len = self.sline_lens[ii]

                for jj in range(self.nSlines[ii]):
                    npts = this_len[jj]

                    yield np.asarray(this_sls[jj], dtype=REAL_DTYPE)[:npts]

        return _yield_slines()

    def gen_bin_indices(self, bin_starts, bin_len):
        bin_edges = np.append(bin_starts, bin_starts[-1] + bin_len)
        bin_indices = {k: [] for k in range(len(bin_starts))}
        
        for ii in range(self.ngpus):
            scaled_lens = self.sline_lens[ii] * self.gpu_tracker.step_size
            assignments = np.digitize(scaled_lens, bin_edges) - 1
            
            for k in range(len(bin_starts)):
                jj_indices = np.where(assignments == k)[0]
                if jj_indices.size > 0:
                    bin_indices[k].append((ii, jj_indices))

        return bin_indices

    def as_array_sequence_group(self, bin_indices, bin_start):
        relevant_blocks = bin_indices[bin_start]

        def _yield_slines():
            for ii, jj_array in relevant_blocks:
                gpu_slines = self.slines[ii]
                gpu_lens = self.sline_lens[ii]
                for jj in jj_array:
                    npts = gpu_lens[jj]
                    yield np.asarray(gpu_slines[jj], dtype=REAL_DTYPE)[:npts]

        def _get_buffer_size():
            total_pts = 0
            for ii, jj_array in relevant_blocks:
                total_pts += np.sum(self.sline_lens[ii][jj_array])

            return math.ceil((total_pts * 3 * REAL_SIZE) / MEGABYTE)

        return ArraySequence(_yield_slines(), _get_buffer_size())

    def as_array_sequence(self):
        return ArraySequence(self.as_generator(), self.get_buffer_size())
