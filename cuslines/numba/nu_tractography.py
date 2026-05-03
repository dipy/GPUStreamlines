import math
from math import radians

import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from nibabel.streamlines.array_sequence import ArraySequence, MEGABYTE
from tqdm import tqdm

from cuslines.generic_tracker import GenericTracker
from cuslines.numba_njit.num_streamlines_numba import getNumStreamlinesProb_generator
from cuslines.numba_njit.generate_streamlines_numba import genStreamlinesMergeProb_generator
from cuslines.numba.nu_globals import MAX_SLINE_LEN, REAL_DTYPE


class CPUProbDirectionGetter:
    pass

class CPUPttDirectionGetter:
    def __init__(self):
        raise NotImplementedError(
        "Only CPU detected. Only ProbDirectionGetter implemented on CPU. \n"
        "Either switch to ProbDirectionGetter or use a backend. Install either:\n"
        "  - CUDA: pip install 'cuslines[cu13]' (NVIDIA GPU)\n"
        "  - Metal: pip install 'cuslines[metal]' (Apple Silicon)\n"
        "  - WebGPU: pip install 'cuslines[webgpu]' (cross-platform)")

class CPUBootDirectionGetter:
    def __init__(self):
        raise NotImplementedError(
        "Only CPU detected. Only ProbDirectionGetter implemented on CPU. \n"
        "Either switch to ProbDirectionGetter or use a backend. Install either:\n"
        "  - CUDA: pip install 'cuslines[cu13]' (NVIDIA GPU)\n"
        "  - Metal: pip install 'cuslines[metal]' (Apple Silicon)\n"
        "  - WebGPU: pip install 'cuslines[webgpu]' (cross-platform)")

class SeedBatchPropagator:
    def __init__(self, cpu_tracker, minlen: int = 0, maxlen: float = np.inf):
        self.cpu_tracker = cpu_tracker
        self.minlen = minlen
        self.maxlen = maxlen

        self.nSlines    = 0
        self.slines     = None
        self.sline_lens = None

    def _get_num_streamlines(self, seeds):
        t = self.cpu_tracker
        nseed = len(seeds)

        shDir0      = np.zeros((nseed * t.dimt, 3), dtype=REAL_DTYPE)
        slineOutOff = np.zeros(nseed + 1, dtype=np.int32)

        getNumStreamlinesProb = getNumStreamlinesProb_generator(
            t.dimx,
            t.dimy,
            t.dimz,
            t.dimt,
            t.relative_peak_thresh,
            t.min_separation_angle,
            t.nedges,
            t.full_basis,
        )
        getNumStreamlinesProb(
            seeds,
            t.dataf,
            t.sphere_vertices,
            t.sphere_edges,
            shDir0,
            slineOutOff,
        )

        __pval = slineOutOff[0]
        slineOutOff[0] = 0
        for jj in range(1, nseed + 1):
            __cval = slineOutOff[jj]
            slineOutOff[jj] = slineOutOff[jj - 1] + __pval
            __pval = __cval

        return shDir0, slineOutOff

    def _generate_streamlines(self, seeds, shDir0, slineOutOff):
        t = self.cpu_tracker
        nSlines = int(slineOutOff[-1])

        slineSeed = np.full(nSlines, -1, dtype=np.int32)
        slineLen  = np.zeros(nSlines, dtype=np.int32)
        sline     = np.zeros((nSlines * MAX_SLINE_LEN * 2, 3), dtype=REAL_DTYPE)

        genStreamlinesMergeProb = genStreamlinesMergeProb_generator(
            t.dimx,
            t.dimy,
            t.dimz,
            t.dimt,
            t.full_basis,
            t.step_size,
            t.max_angle,
            t.tc_threshold,
        )
        genStreamlinesMergeProb(
            seeds,
            t.dataf,
            t.metric_map,
            t.sphere_vertices,
            t.sphere_edges,
            slineOutOff,
            shDir0,
            slineSeed,
            slineLen,
            sline,
        )
        return nSlines, slineLen, sline

    def propagate(self, seeds: np.ndarray):
        """
        Run full two-phase tracking for `seeds` (float32[N, 3]).
        Results stored in self.slines, self.sline_lens, self.nSlines.
        """
        seeds = np.ascontiguousarray(seeds, dtype=REAL_DTYPE)

        shDir0, slineOutOff = self._get_num_streamlines(seeds)
        nSlines, slineLen, sline = self._generate_streamlines(seeds, shDir0, slineOutOff)

        self.nSlines    = nSlines
        self.sline_lens = slineLen
        self.slines = sline

    def get_buffer_size(self) -> int:
        """Return estimated buffer size in MB (mirrors GPU version)."""
        if self.sline_lens is None:
            return 0
        total_pts = sum(
            l for l in self.sline_lens[:self.nSlines]
            if self.minlen <= l <= self.maxlen
        )
        return math.ceil(total_pts * 3 * REAL_DTYPE(0).itemsize / MEGABYTE)

    def as_generator(self):
        def _yield_slines():
            slines     = self.slines
            sline_lens = self.sline_lens
            step       = MAX_SLINE_LEN * 2   # points allocated per streamline

            for i in range(self.nSlines):
                npts = int(sline_lens[i])
                if npts < self.minlen or npts > self.maxlen:
                    continue
                yield np.asarray(slines[i * step : i * step + npts], dtype=REAL_DTYPE)
        return _yield_slines()

    def as_array_sequence(self) -> ArraySequence:
        return ArraySequence(self.as_generator(), self.get_buffer_size())


class CPUTracker(GenericTracker):
    """
    CPU probabilistic tractography tracker.

    Parameters
    ----------
    dg :  DirectionGetter
        Direction getter to use.
        Can only be CPUProbDirectionGetter.
        Maintained to match API with other backends.
    dataf : np.ndarray, shape (dimx, dimy, dimz, dimt)
        ODF volume.
    stop_map : np.ndarray, shape (dimx, dimy, dimz)
        Stopping metric (e.g. GFA or FA).
    stop_threshold : float
        Voxels with stop_map <= stop_threshold are endpoints.
    sphere_vertices : np.ndarray, shape (dimt, 3)
        Unit sphere vertices.
    sphere_edges : np.ndarray, shape (num_edges, 2)
        Sphere adjacency list (int32).
    max_angle : float
        Maximum turning angle in radians. Default: radians(60).
    step_size : float
        Step size in voxels. Default: 0.5.
    min_pts : int
        Minimum streamline length (points) to keep. Default: 0.
    max_pts : float
        Maximum streamline length (points) to keep. Default: inf.
    relative_peak_thresh : float
        Relative peak threshold for direction selection. Default: 0.25.
    min_separation_angle : float
        Minimum separation angle (radians) between peaks. Default: radians(45).
    ngpus : int, optional
        Ignored. Maintained to match API with other backends.
        default: 1
    rng_seed : int, optional
        Seed for random number generator
        default: 0
    rng_offset : int, optional
        Ignored. Maintained to match API with other backends.
        default: 0
    chunk_size : int
        Seeds per propagate() call in generate_sft(). Default: 100000.
    """

    def __init__(
        self,
        dg: object,
        dataf: np.ndarray,
        stop_map: np.ndarray,
        stop_threshold: float,
        sphere_vertices: np.ndarray,
        sphere_edges: np.ndarray,
        full_basis: bool = False,
        max_angle: float = radians(60),
        step_size: float = 0.5,
        min_pts: int = 0,
        max_pts: float = np.inf,
        relative_peak_thresh: float = 0.25,
        min_separation_angle: float = radians(45),
        ngpus: int = 1,
        rng_seed: int = 0,
        rng_offset: int = 0,
        chunk_size: int = 100000,
    ):
        self.dataf           = np.ascontiguousarray(dataf,           dtype=REAL_DTYPE)
        self.metric_map      = np.ascontiguousarray(stop_map,        dtype=REAL_DTYPE)
        self.sphere_vertices = np.ascontiguousarray(sphere_vertices, dtype=REAL_DTYPE)
        self.sphere_edges    = np.ascontiguousarray(sphere_edges,    dtype=np.int32)

        self.full_basis = full_basis
        self.dimx, self.dimy, self.dimz, self.dimt = dataf.shape

        self.max_angle             = float(max_angle)
        self.tc_threshold          = float(stop_threshold)
        self.step_size             = float(step_size)
        self.relative_peak_thresh  = float(relative_peak_thresh)
        self.min_separation_angle  = float(min_separation_angle)
        self.chunk_size            = int(chunk_size)
        self.nedges                = int(sphere_edges.shape[0])

        if rng_seed != 0:
            np.random.seed(rng_seed)

        self.seed_propagator = SeedBatchPropagator(
            cpu_tracker=self, minlen=min_pts, maxlen=max_pts
        )
