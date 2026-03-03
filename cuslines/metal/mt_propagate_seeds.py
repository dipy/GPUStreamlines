"""Metal seed batch propagator — mirrors cuslines/cuda_python/cu_propagate_seeds.py.

Unified memory advantage: no cudaMemcpy needed. Seeds and results live in
shared CPU/GPU buffers.
"""

import numpy as np
import math
import gc
import logging

from nibabel.streamlines.array_sequence import ArraySequence, MEGABYTE

from cuslines.metal.mutils import (
    REAL_SIZE,
    REAL_DTYPE,
    REAL3_SIZE,
    MAX_SLINE_LEN,
    EXCESS_ALLOC_FACT,
    THR_X_SL,
    THR_X_BL,
    BLOCK_Y,
    div_up,
)

logger = logging.getLogger("GPUStreamlines")


class MetalSeedBatchPropagator:
    def __init__(self, gpu_tracker, minlen=0, maxlen=np.inf):
        self.gpu_tracker = gpu_tracker
        self.minlen = minlen
        self.maxlen = maxlen

        self.nSlines = 0
        self.nSlines_old = 0
        self.slines = None
        self.sline_lens = None

        # Metal buffers
        self.seeds_buf = None
        self.slinesOffs_buf = None
        self.shDirTemp0_buf = None
        self.slineSeed_buf = None
        self.slineLen_buf = None
        self.sline_buf = None

        # Backing numpy arrays (unified memory — these ARE the GPU data)
        self._seeds_arr = None
        self._slinesOffs_arr = None
        self._shDirTemp0_arr = None
        self._slineSeed_arr = None
        self._slineLen_arr = None
        self._sline_arr = None

    def _get_sl_buffer_size(self):
        return REAL_SIZE * 2 * 3 * MAX_SLINE_LEN * int(self.nSlines)

    def _allocate_seed_memory(self, seeds):
        from cuslines.metal.mt_tractography import (
            _make_shared_buffer, _make_dynamic_buffer, _buffer_as_array,
        )

        nseeds = len(seeds)
        device = self.gpu_tracker.device
        block = (THR_X_SL, BLOCK_Y, 1)
        grid = (div_up(nseeds, BLOCK_Y), 1, 1)

        # Seeds — copy into Metal shared buffer
        seeds_arr = np.ascontiguousarray(seeds, dtype=REAL_DTYPE)
        self.seeds_buf = _make_shared_buffer(device, seeds_arr)

        # Streamline offsets — dynamic buffer (GPU writes, CPU reads for prefix sum)
        offs_nbytes = (nseeds + 1) * np.dtype(np.int32).itemsize
        self.slinesOffs_buf = _make_dynamic_buffer(device, offs_nbytes)
        self._slinesOffs_arr = _buffer_as_array(
            self.slinesOffs_buf, np.int32, (nseeds + 1,)
        )
        self._slinesOffs_arr[:] = 0

        # Initial directions from each seed
        shdir_size = self.gpu_tracker.samplm_nr * grid[0] * block[1]
        shdir_nbytes = shdir_size * 3 * REAL_SIZE
        self.shDirTemp0_buf = _make_dynamic_buffer(device, shdir_nbytes)

        return nseeds, block, grid

    def _cumsum_offsets(self, nseeds):
        """CPU-side prefix sum on offsets — no memcpy needed with unified memory."""
        offs = self._slinesOffs_arr

        # Exclusive prefix sum: shift cumsum right, insert 0 at start
        counts = offs[:nseeds].copy()
        np.cumsum(counts, out=offs[1:nseeds + 1])
        offs[0] = 0
        self.nSlines = int(offs[nseeds])

    def _allocate_tracking_memory(self):
        from cuslines.metal.mt_tractography import (
            _make_dynamic_buffer, _buffer_as_array,
        )

        device = self.gpu_tracker.device

        if self.nSlines > EXCESS_ALLOC_FACT * self.nSlines_old:
            self.slines = None
            self.sline_lens = None
            gc.collect()

        if self.slines is None:
            self.slines = np.empty(
                (EXCESS_ALLOC_FACT * self.nSlines, MAX_SLINE_LEN * 2, 3),
                dtype=REAL_DTYPE,
            )
        if self.sline_lens is None:
            self.sline_lens = np.empty(
                EXCESS_ALLOC_FACT * self.nSlines, dtype=np.int32
            )

        # Seed-to-streamline mapping — dynamic buffer (GPU writes seed indices)
        seed_nbytes = self.nSlines * np.dtype(np.int32).itemsize
        self.slineSeed_buf = _make_dynamic_buffer(device, seed_nbytes)
        self._slineSeed_arr = _buffer_as_array(
            self.slineSeed_buf, np.int32, (self.nSlines,)
        )
        self._slineSeed_arr[:] = -1

        # Streamline lengths — dynamic buffer (GPU writes lengths)
        len_nbytes = self.nSlines * np.dtype(np.int32).itemsize
        self.slineLen_buf = _make_dynamic_buffer(device, len_nbytes)
        self._slineLen_arr = _buffer_as_array(
            self.slineLen_buf, np.int32, (self.nSlines,)
        )
        self._slineLen_arr[:] = 0

        # Streamline output buffer — dynamic buffer (GPU writes streamline points)
        buffer_count = 2 * 3 * MAX_SLINE_LEN * self.nSlines
        sline_nbytes = buffer_count * REAL_SIZE
        self.sline_buf = _make_dynamic_buffer(device, sline_nbytes)
        self._sline_arr = _buffer_as_array(
            self.sline_buf, REAL_DTYPE, (buffer_count,)
        )

    def _copy_results(self):
        """With unified memory, results are already in CPU-accessible memory.
        Just reshape/copy into the output arrays."""
        if self.nSlines == 0:
            return

        # Reshape the flat sline buffer into (nSlines, MAX_SLINE_LEN*2, 3)
        sline_view = self._sline_arr.reshape(self.nSlines, MAX_SLINE_LEN * 2, 3)
        self.slines[:self.nSlines] = sline_view
        self.sline_lens[:self.nSlines] = self._slineLen_arr

    def propagate(self, seeds):
        self.nseeds = len(seeds)

        nseeds, block, grid = self._allocate_seed_memory(seeds)

        # Pass 1: count streamlines per seed
        self.gpu_tracker.dg.getNumStreamlines(nseeds, block, grid, self)

        # Prefix sum offsets (no memcpy — unified memory)
        self._cumsum_offsets(nseeds)

        if self.nSlines == 0:
            self.nSlines_old = self.nSlines
            self.gpu_tracker.rng_offset += self.nseeds
            return

        self._allocate_tracking_memory()

        # Pass 2: generate streamlines
        self.gpu_tracker.dg.generateStreamlines(nseeds, block, grid, self)

        # Copy results (trivial with unified memory)
        self._copy_results()

        self.nSlines_old = self.nSlines
        self.gpu_tracker.rng_offset += self.nseeds

    def get_buffer_size(self):
        buffer_size = 0
        lens = self.sline_lens
        for jj in range(self.nSlines):
            if lens[jj] < self.minlen or lens[jj] > self.maxlen:
                continue
            buffer_size += lens[jj] * 3 * REAL_SIZE
        return math.ceil(buffer_size / MEGABYTE)

    def as_generator(self):
        def _yield_slines():
            sls = self.slines
            lens = self.sline_lens
            for jj in range(self.nSlines):
                npts = lens[jj]
                if npts < self.minlen or npts > self.maxlen:
                    continue
                yield np.asarray(sls[jj], dtype=REAL_DTYPE)[:npts]

        return _yield_slines()

    def as_array_sequence(self):
        return ArraySequence(self.as_generator(), self.get_buffer_size())
