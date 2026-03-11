"""WebGPU seed batch propagator — mirrors cuslines/metal/mt_propagate_seeds.py.

Key difference from Metal: no unified memory. After each GPU pass, results
must be read back explicitly via device.queue.read_buffer() (~3 readbacks per
seed batch, matching CUDA's cudaMemcpy pattern).
"""

import numpy as np
import math
import gc
import logging

from nibabel.streamlines.array_sequence import ArraySequence, MEGABYTE

from cuslines.webgpu.wgutils import (
    REAL_SIZE,
    REAL_DTYPE,
    REAL3_SIZE,
    MAX_SLINE_LEN,
    EXCESS_ALLOC_FACT,
    THR_X_SL,
    THR_X_BL,
    BLOCK_Y,
    div_up,
    create_buffer_from_data,
    create_empty_buffer,
    read_buffer,
    write_buffer,
)

logger = logging.getLogger("GPUStreamlines")


class WebGPUSeedBatchPropagator:
    def __init__(self, gpu_tracker, minlen=0, maxlen=np.inf):
        self.gpu_tracker = gpu_tracker
        self.minlen = minlen
        self.maxlen = maxlen

        self.nSlines = 0
        self.nSlines_old = 0
        self.slines = None
        self.sline_lens = None

        # GPU buffers
        self.seeds_buf = None
        self.slinesOffs_buf = None
        self.shDirTemp0_buf = None
        self.slineSeed_buf = None
        self.slineLen_buf = None
        self.sline_buf = None

    def _get_sl_buffer_size(self):
        return REAL_SIZE * 2 * 3 * MAX_SLINE_LEN * int(self.nSlines)

    def _allocate_seed_memory(self, seeds):
        nseeds = len(seeds)
        device = self.gpu_tracker.device
        block = (THR_X_SL, BLOCK_Y, 1)
        grid = (div_up(nseeds, BLOCK_Y), 1, 1)

        # Seeds — upload to GPU
        seeds_arr = np.ascontiguousarray(seeds, dtype=REAL_DTYPE)
        self.seeds_buf = create_buffer_from_data(device, seeds_arr.ravel(), label="seeds")

        # Streamline offsets — GPU writes counts, CPU reads for prefix sum
        offs_nbytes = (nseeds + 1) * np.dtype(np.int32).itemsize
        self.slinesOffs_buf = create_empty_buffer(device, offs_nbytes, label="slinesOffs")
        # Zero-initialize
        zeros = np.zeros(nseeds + 1, dtype=np.int32)
        write_buffer(device, self.slinesOffs_buf, zeros)

        # Initial directions from each seed
        shdir_size = self.gpu_tracker.samplm_nr * grid[0] * block[1]
        shdir_nbytes = shdir_size * 3 * REAL_SIZE
        self.shDirTemp0_buf = create_empty_buffer(device, shdir_nbytes, label="shDirTemp0")

        return nseeds, block, grid

    def _cumsum_offsets(self, nseeds):
        """Read offsets from GPU, do CPU prefix sum, write back."""
        device = self.gpu_tracker.device

        # Readback 1: streamline counts per seed
        offs = read_buffer(device, self.slinesOffs_buf, dtype=np.int32)

        # Exclusive prefix sum: shift cumsum right, insert 0 at start
        counts = offs[:nseeds].copy()
        result = np.empty(nseeds + 1, dtype=np.int32)
        result[0] = 0
        np.cumsum(counts, out=result[1:nseeds + 1])
        self.nSlines = int(result[nseeds])

        # Write back prefix-summed offsets to GPU
        write_buffer(device, self.slinesOffs_buf, result)

    def _allocate_tracking_memory(self):
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

        # Seed-to-streamline mapping
        seed_nbytes = self.nSlines * np.dtype(np.int32).itemsize
        self.slineSeed_buf = create_empty_buffer(device, seed_nbytes, label="slineSeed")
        write_buffer(device, self.slineSeed_buf, np.full(self.nSlines, -1, dtype=np.int32))

        # Streamline lengths
        len_nbytes = self.nSlines * np.dtype(np.int32).itemsize
        self.slineLen_buf = create_empty_buffer(device, len_nbytes, label="slineLen")
        write_buffer(device, self.slineLen_buf, np.zeros(self.nSlines, dtype=np.int32))

        # Streamline output buffer (flat f32: nSlines * MAX_SLINE_LEN * 2 * 3)
        buffer_count = 2 * 3 * MAX_SLINE_LEN * self.nSlines
        sline_nbytes = buffer_count * REAL_SIZE

        max_binding = device.limits["max-storage-buffer-binding-size"]
        if sline_nbytes > max_binding:
            max_slines = max_binding // (2 * 3 * MAX_SLINE_LEN * REAL_SIZE)
            raise RuntimeError(
                f"Streamline buffer ({sline_nbytes / 1e9:.1f} GB, "
                f"{self.nSlines} streamlines) exceeds WebGPU storage buffer "
                f"binding limit ({max_binding / 1e9:.1f} GB). "
                f"Reduce --chunk-size (current batch produced {self.nSlines} "
                f"streamlines from {self.nseeds} seeds; max ~{max_slines} "
                f"streamlines fit in a single buffer)."
            )

        self.sline_buf = create_empty_buffer(device, sline_nbytes, label="sline")

    def _copy_results(self):
        """Read GPU results back to CPU arrays."""
        if self.nSlines == 0:
            return
        device = self.gpu_tracker.device

        # Readback 2: streamline points
        sline_data = read_buffer(device, self.sline_buf, dtype=REAL_DTYPE)
        sline_view = sline_data.reshape(self.nSlines, MAX_SLINE_LEN * 2, 3)
        self.slines[:self.nSlines] = sline_view

        # Readback 3: streamline lengths
        self.sline_lens[:self.nSlines] = read_buffer(
            device, self.slineLen_buf, dtype=np.int32
        )[:self.nSlines]

    def propagate(self, seeds):
        self.nseeds = len(seeds)

        nseeds, block, grid = self._allocate_seed_memory(seeds)

        # Pass 1: count streamlines per seed
        self.gpu_tracker.dg.getNumStreamlines(nseeds, block, grid, self)

        # Prefix sum offsets (requires GPU→CPU readback)
        self._cumsum_offsets(nseeds)

        if self.nSlines == 0:
            self.nSlines_old = self.nSlines
            self.gpu_tracker.rng_offset += self.nseeds
            return

        self._allocate_tracking_memory()

        # Pass 2: generate streamlines
        self.gpu_tracker.dg.generateStreamlines(nseeds, block, grid, self)

        # Read results back from GPU
        self._copy_results()

        self.nSlines_old = self.nSlines
        self.gpu_tracker.rng_offset += self.nseeds

    def get_buffer_size(self):
        lens = self.sline_lens[:self.nSlines]
        mask = (lens >= self.minlen) & (lens <= self.maxlen)
        buffer_size = int(lens[mask].sum()) * 3 * REAL_SIZE
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
