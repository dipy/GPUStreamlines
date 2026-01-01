import numpy as np
import ctypes
from cuda.bindings import runtime
from nibabel.streamlines.array_sequence import ArraySequence
import logging

from cutils import (
    REAL_SIZE,
    REAL_DTYPE,
    REAL3_DTYPE,
    MAX_SLINE_LEN,
    EXCESS_ALLOC_FACT,
    THR_X_SL,
    THR_X_BL,
    div_up,
    checkCudaErrors,
)


logger = logging.getLogger("GPUStreamlines")


class SeedBatchPropagator:
    def __init__(
            self,
            gpu_tracker):
        self.gpu_tracker = gpu_tracker

        self.nSlines_old = np.zeros(self.ngpus, dtype=np.int32)
        self.nSlines = np.zeros(self.ngpus, dtype=np.int32)
        self.slines = np.zeros(self.ngpus, dtype=ctypes.c_void_p)
        self.sline_lens = np.zeros(self.ngpus, dtype=ctypes.c_void_p)

        self.seeds_d = np.empty(self.ngpus, dtype=ctypes.c_void_p)
        self.slineSeed_d = np.empty(self.ngpus, dtype=ctypes.c_void_p)
        self.slinesOffs_d = np.empty(self.ngpus, dtype=ctypes.c_void_p)
        self.shDirTemp0_d = np.empty(self.ngpus, dtype=ctypes.c_void_p)
        self.slineLen_d = np.empty(self.ngpus, dtype=ctypes.c_void_p)
        self.sline_d = np.empty(self.ngpus, dtype=ctypes.c_void_p)

    def _switch_device(self, n):
        checkCudaErrors(runtime.cudaSetDevice(n))

        nseeds_gpu =  min(
            self.nseeds_per_gpu, max(0, self.nseeds - n * self.nseeds_per_gpu))
        block = (THR_X_SL, THR_X_BL//THR_X_SL, 1)
        grid = (div_up(nseeds_gpu, THR_X_BL//THR_X_SL), 1, 1)

        return nseeds_gpu, block, grid

    def _get_sl_buffer_size(self, n):
        return REAL_SIZE*2*3*MAX_SLINE_LEN*self.nSlines[n]

    def _allocate_seed_memory(self):
        # Move seeds to GPU
        for ii in range(self.ngpus):
            nseeds_gpu, _, _ = self._switch_device(ii)
            self.seeds_d[ii] = checkCudaErrors(runtime.cudaMalloc(
                REAL_SIZE*3*nseeds_gpu))
            checkCudaErrors(runtime.cudaMemcpy(
                self.seeds_d[ii],
                self.seeds[ii*self.nseeds_per_gpu:(ii+1)*self.nseeds_per_gpu].ctypes.data,
                REAL_SIZE*3*nseeds_gpu,
                runtime.cudaMemcpyHostToDevice))

        for ii in range(self.ngpus):
            nseeds_gpu, block, grid = self._switch_device(ii)
            # Streamline offsets
            self.slinesOffs_d[ii] = checkCudaErrors(runtime.cudaMalloc(
                np.uint64().nbytes * (nseeds_gpu + 1)))
            # Initial directions from each seed
            self.shDirTemp0_d[ii] = checkCudaErrors(runtime.cudaMalloc(
                REAL3_DTYPE.nbytes * self.samplm_nr * grid[0] * block[1]))

    def _cumsum_offsets(self):
        for ii in range(self.ngpus):
            nseeds_gpu, _, _ = self._switch_device(ii)
            if (nseeds_gpu == 0):
                self.nSlines[ii] = 0
                continue

            slinesOffs_h = np.empty(nseeds_gpu + 1, dtype=np.int32)
            checkCudaErrors(runtime.cudaMemcpy(
                slinesOffs_h.ctypes.data,
                self.slinesOffs_d[ii],
                slinesOffs_h.nbytes * (nseeds_gpu + 1),
                runtime.cudaMemcpyDeviceToHost))

            slinesOffs_h = np.concatenate((
                [0], np.cumsum(slinesOffs_h[:-1], dtype=slinesOffs_h.dtype)))
            self.nSlines[ii] = int(slinesOffs_h[-1])

            checkCudaErrors(runtime.cudaMemcpy(
                self.slinesOffs_d[ii],
                slinesOffs_h.ctypes.data,
                self.slinesOffs_d.size * (nseeds_gpu + 1),
                runtime.cudaMemcpyHostToDevice))

    def _allocate_tracking_memory(self):
        for ii in range(self.ngpus):
            self._switch_device(ii)

            self.slineSeed_d[ii] = checkCudaErrors(runtime.cudaMalloc(
                self.nSlines[ii] * np.int32().nbytes))
            checkCudaErrors(runtime.cudaMemset(
                self.slineSeed_d[ii],
                -1,
                self.nSlines[ii] * np.int32().nbytes))

            if self.nSlines[ii] > EXCESS_ALLOC_FACT*self.nSlines_old[ii]:
                if self.slines[ii]:
                    checkCudaErrors(runtime.cudaFreeHost(
                        self.slines[ii]))
                if self.sline_lens[ii]:
                    checkCudaErrors(runtime.cudaFreeHost(
                        self.sline_lens[ii]))
                self.slines[ii] = 0  # Nullptr
                self.sline_lens[ii] = 0  # Nullptr

            buffer_size = self._get_sl_buffer_size(ii)
            logger.debug(f"Streamline buffer size: {buffer_size}")

            if not self.slines[ii]:
                self.slines[ii] = checkCudaErrors(runtime.cudaMallocHost(
                    buffer_size))
            if not self.slines_lens[ii]:
                self.slines_lens[ii] = checkCudaErrors(runtime.cudaMallocHost(
                    np.int32().nbytes*EXCESS_ALLOC_FACT*self.nSlines[ii]))

        for ii in range(self.ngpus):
            self._switch_device(ii)
            buffer_size = self._get_sl_buffer_size(ii)

            self.slineLen_d[ii] = checkCudaErrors(runtime.cudaMalloc(
                np.int32().nbytes * self.nSlines[ii]))
            self.sline_d[ii] = checkCudaErrors(runtime.cudaMalloc(
                buffer_size))

    def _cleanup(self):
        for ii in range(self.ngpus):
            self._switch_device(ii)
            checkCudaErrors(runtime.cudaMemcpyAsync(
                self.slines[ii],
                self.sline_d[ii],
                self._get_sl_buffer_size(ii),
                runtime.cudaMemcpyDeviceToHost,
                self.gpu_tracker.streams[ii]))
            checkCudaErrors(runtime.cudaMemcpyAsync(
                self.sline_lens[ii],
                self.slineLen_d[ii],
                np.int32().nbytes*self.nSlines[ii],
                runtime.cudaMemcpyDeviceToHost,
                self.gpu_tracker.streams[ii]))

        for ii in range(self.ngpus):
            self._switch_device(ii)
            checkCudaErrors(runtime.cudaStreamSynchronize(
                self.gpu_tracker.streams[ii]))
            checkCudaErrors(runtime.cudaFree(self.seeds_d[ii]))
            checkCudaErrors(runtime.cudaFree(self.slineSeed_d[ii]))
            checkCudaErrors(runtime.cudaFree(self.slinesOffs_d[ii]))
            checkCudaErrors(runtime.cudaFree(self.shDirTemp0_d[ii]))
            checkCudaErrors(runtime.cudaFree(self.slineLen_d[ii]))
            checkCudaErrors(runtime.cudaFree(self.sline_d[ii]))

        self.nSlines_old = self.nSlines.copy()
        self.rng_offset += self.nseeds

    def propagate(self, seeds):
        self.seeds = seeds
        self.nseeds = len(seeds)
        self.nseeds_per_gpu = (self.nseeds + self.gpu_tracker.ngpus - 1) // self.gpu_tracker.ngpus

        self._seeds_to_gpu()
        self._allocate_seed_memory()

        for ii in range(self.ngpus):
            nseeds_gpu, block, grid = self._switch_device(ii)
            if (nseeds_gpu == 0):
                continue

            getNumStreamlines() # TODO: these will each be classes you can pass in

        self._cumsum_offsets()
        self._allocate_tracking_memory()

        for ii in range(self.ngpus):
            nseeds_gpu, block, grid = self._switch_device(ii)
            if (nseeds_gpu == 0):
                continue

            mergeStreamlines() # TODO

        self._cleanup()

    def as_array_sequence(self):  # TODO: optimize memory usage here? also, direct to trx?
        buffer_size = 0
        for ii in range(self.ngpus):
            lens = self.sline_lens[ii]
            for jj in range(self.nSlines[ii]):
                buffer_size += lens[jj] * 3 * REAL_SIZE

        def _yield_slines():
            for ii in range(self.ngpus):
                this_sls = self.slines[ii]
                this_len = self.sline_lens[ii]

                for jj in range(self.nSlines[ii]):
                    npts = this_len[jj]
                    offset = jj * 3 * 2 * MAX_SLINE_LEN

                    sl = np.asarray(
                        this_sls[offset : offset + npts * 3],
                        dtype=REAL_DTYPE)
                    sl = sl.reshape((npts, 3))
                    yield sl

        return ArraySequence(_yield_slines, buffer_size)
