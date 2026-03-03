"""WebGPU direction getters — mirrors cuslines/metal/mt_direction_getters.py.

Compiles WGSL shaders at runtime and dispatches compute passes via wgpu-py.
"""

import numpy as np
import struct
from abc import ABC, abstractmethod
import logging
from importlib.resources import files
from time import time

from cuslines.boot_utils import prepare_opdt, prepare_csa

from cuslines.webgpu.wgutils import (
    REAL_SIZE,
    REAL_DTYPE,
    REAL3_SIZE,
    BLOCK_Y,
    THR_X_SL,
    div_up,
    create_buffer_from_data,
)

logger = logging.getLogger("GPUStreamlines")


class WebGPUDirectionGetter(ABC):
    """Abstract base for WebGPU direction getters."""

    @abstractmethod
    def getNumStreamlines(self, nseeds_gpu, block, grid, sp):
        pass

    @abstractmethod
    def generateStreamlines(self, nseeds_gpu, block, grid, sp):
        pass

    def setup_device(self, device, has_subgroups=True):
        """Called once when WebGPUTracker allocates resources."""
        pass

    def compile_program(self, device, has_subgroups=True):
        start_time = time()
        logger.info("Compiling WebGPU/WGSL shaders...")

        shader_dir = files("cuslines").joinpath("wgsl_shaders")

        # Read shader files in dependency order and concatenate.
        # WGSL has no #include mechanism, so we concatenate all source files.
        source_parts = []

        # Note: wgpu-native/Naga enables subgroup operations via device features
        # rather than WGSL's `enable subgroups;` directive (not yet supported in
        # Naga). Subgroup builtins are available when the "subgroup" feature is
        # requested at device creation time.

        # Foundation files
        foundation_files = [
            "globals.wgsl",
            "types.wgsl",
            "philox_rng.wgsl",
        ]

        # Utility files
        utility_files = [
            "utils.wgsl",
            "warp_sort.wgsl",
            "tracking_helpers.wgsl",
        ]

        # Direction-getter-specific files
        dg_files = self._shader_files()

        # Main kernel file(s)
        kernel_files = self._kernel_files()

        all_files = foundation_files + utility_files + dg_files + kernel_files

        for fname in all_files:
            path = shader_dir.joinpath(fname)
            with open(path, "r") as f:
                source_parts.append(f"// ── {fname} ──\n")
                source_parts.append(f.read())

        full_source = "\n".join(source_parts)

        shader_module = device.create_shader_module(code=full_source)
        self.shader_module = shader_module
        logger.info("WGSL shaders compiled in %.2f seconds", time() - start_time)

    def _shader_files(self):
        """Return list of additional .wgsl files needed by this direction getter."""
        return []

    def _kernel_files(self):
        """Return list of kernel .wgsl files. Override for standalone kernels like boot."""
        return ["generate_streamlines.wgsl"]

    def _make_pipeline(self, device, entry_point):
        pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={
                "module": self.shader_module,
                "entry_point": entry_point,
            },
        )
        return pipeline

    def _dispatch_kernel(self, pipeline, bind_groups, grid, device):
        """Submit a compute pass with the given pipeline and bind groups."""
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        for idx, bg in enumerate(bind_groups):
            compute_pass.set_bind_group(idx, bg)
        compute_pass.dispatch_workgroups(grid[0], grid[1], grid[2])
        compute_pass.end()
        device.queue.submit([encoder.finish()])


class WebGPUProbDirectionGetter(WebGPUDirectionGetter):
    """Probabilistic direction getter for WebGPU."""

    def __init__(self):
        self.shader_module = None
        self.getnum_pipeline = None
        self.gen_pipeline = None

    def _shader_files(self):
        return []

    def setup_device(self, device, has_subgroups=True):
        self.compile_program(device, has_subgroups)
        self.getnum_pipeline = self._make_pipeline(device, "getNumStreamlinesProb_k")
        self.gen_pipeline = self._make_pipeline(device, "genStreamlinesMergeProb_k")

    def _make_params_bytes(self, sp, nseeds_gpu, for_gen=False):
        gt = sp.gpu_tracker
        rng_seed = gt.rng_seed
        rng_seed_lo = rng_seed & 0xFFFFFFFF
        rng_seed_hi = (rng_seed >> 32) & 0xFFFFFFFF

        # ProbTrackingParams struct layout (must match WGSL struct)
        # float max_angle, tc_threshold, step_size, relative_peak_thresh, min_separation_angle
        # int rng_seed_lo, rng_seed_hi, rng_offset, nseed
        # int dimx, dimy, dimz, dimt, samplm_nr, num_edges, model_type
        values = [
            gt.max_angle,
            gt.tc_threshold if for_gen else 0.0,
            gt.step_size if for_gen else 0.0,
            gt.relative_peak_thresh,
            gt.min_separation_angle,
            rng_seed_lo,
            rng_seed_hi,
            gt.rng_offset if for_gen else 0,
            nseeds_gpu,
            gt.dimx, gt.dimy, gt.dimz, gt.dimt,
            gt.samplm_nr, gt.nedges, 2,  # model_type = PROB
        ]
        # 5 floats + 11 ints = 64 bytes
        return struct.pack("5f11i", *values)

    def getNumStreamlines(self, nseeds_gpu, block, grid, sp):
        gt = sp.gpu_tracker
        device = gt.device
        params_bytes = self._make_params_bytes(sp, nseeds_gpu, for_gen=False)

        # Create params buffer from packed bytes
        params_buf = device.create_buffer_with_data(
            data=params_bytes, usage="STORAGE | COPY_SRC"
        )

        # With layout="auto", only bindings actually used by the entry point
        # are included. getNumStreamlinesProb_k uses:
        #   Group 0: params(0), seeds(1), dataf(2), sphere_vertices(4), sphere_edges(5)
        #            metric_map(3) is NOT used by getNum
        #   Group 1: slineOutOff(0), shDir0(1)
        #            slineSeed(2), slineLen(3), sline(4) NOT used by getNum
        bg0 = device.create_bind_group(
            layout=self.getnum_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": params_buf}},
                {"binding": 1, "resource": {"buffer": sp.seeds_buf}},
                {"binding": 2, "resource": {"buffer": gt.dataf_buf}},
                {"binding": 4, "resource": {"buffer": gt.sphere_vertices_buf}},
                {"binding": 5, "resource": {"buffer": gt.sphere_edges_buf}},
            ],
        )

        bg1 = device.create_bind_group(
            layout=self.getnum_pipeline.get_bind_group_layout(1),
            entries=[
                {"binding": 0, "resource": {"buffer": sp.slinesOffs_buf}},
                {"binding": 1, "resource": {"buffer": sp.shDirTemp0_buf}},
            ],
        )

        self._dispatch_kernel(self.getnum_pipeline, [bg0, bg1], grid, device)

    def generateStreamlines(self, nseeds_gpu, block, grid, sp):
        gt = sp.gpu_tracker
        device = gt.device
        params_bytes = self._make_params_bytes(sp, nseeds_gpu, for_gen=True)

        params_buf = device.create_buffer_with_data(
            data=params_bytes, usage="STORAGE | COPY_SRC"
        )

        # Group 0: params, seeds, dataf, metric_map, sphere_vertices, sphere_edges
        bg0 = device.create_bind_group(
            layout=self.gen_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": params_buf}},
                {"binding": 1, "resource": {"buffer": sp.seeds_buf}},
                {"binding": 2, "resource": {"buffer": gt.dataf_buf}},
                {"binding": 3, "resource": {"buffer": gt.metric_map_buf}},
                {"binding": 4, "resource": {"buffer": gt.sphere_vertices_buf}},
                {"binding": 5, "resource": {"buffer": gt.sphere_edges_buf}},
            ],
        )

        # Group 1: slineOutOff, shDir0, slineSeed, slineLen, sline
        bg1 = device.create_bind_group(
            layout=self.gen_pipeline.get_bind_group_layout(1),
            entries=[
                {"binding": 0, "resource": {"buffer": sp.slinesOffs_buf}},
                {"binding": 1, "resource": {"buffer": sp.shDirTemp0_buf}},
                {"binding": 2, "resource": {"buffer": sp.slineSeed_buf}},
                {"binding": 3, "resource": {"buffer": sp.slineLen_buf}},
                {"binding": 4, "resource": {"buffer": sp.sline_buf}},
            ],
        )

        self._dispatch_kernel(self.gen_pipeline, [bg0, bg1], grid, device)


class WebGPUPttDirectionGetter(WebGPUProbDirectionGetter):
    """PTT direction getter for WebGPU."""

    def _shader_files(self):
        return ["disc.wgsl", "ptt.wgsl"]

    def setup_device(self, device, has_subgroups=True):
        self.compile_program(device, has_subgroups)
        # PTT reuses Prob's getNum kernel for initial direction finding
        self.getnum_pipeline = self._make_pipeline(device, "getNumStreamlinesProb_k")
        # PTT has its own gen kernel
        self.gen_pipeline = self._make_pipeline(device, "genStreamlinesMergePtt_k")

    def _make_params_bytes(self, sp, nseeds_gpu, for_gen=False):
        gt = sp.gpu_tracker
        rng_seed = gt.rng_seed
        rng_seed_lo = rng_seed & 0xFFFFFFFF
        rng_seed_hi = (rng_seed >> 32) & 0xFFFFFFFF
        values = [
            gt.max_angle,
            gt.tc_threshold if for_gen else 0.0,
            gt.step_size if for_gen else 0.0,
            gt.relative_peak_thresh,
            gt.min_separation_angle,
            rng_seed_lo,
            rng_seed_hi,
            gt.rng_offset if for_gen else 0,
            nseeds_gpu,
            gt.dimx, gt.dimy, gt.dimz, gt.dimt,
            gt.samplm_nr, gt.nedges, 3,  # model_type = PTT
        ]
        return struct.pack("5f11i", *values)


class WebGPUBootDirectionGetter(WebGPUDirectionGetter):
    """Bootstrap direction getter for WebGPU."""

    def __init__(
        self,
        model_type: str,
        min_signal: float,
        H: np.ndarray,
        R: np.ndarray,
        delta_b: np.ndarray,
        delta_q: np.ndarray,
        sampling_matrix: np.ndarray,
        b0s_mask: np.ndarray,
    ):
        self.model_type_str = model_type.upper()
        if self.model_type_str == "OPDT":
            self.model_type = 0
        elif self.model_type_str == "CSA":
            self.model_type = 1
        else:
            raise ValueError(f"Invalid model_type {model_type}, must be 'OPDT' or 'CSA'")

        self.H = np.ascontiguousarray(H, dtype=REAL_DTYPE)
        self.R = np.ascontiguousarray(R, dtype=REAL_DTYPE)
        self.delta_b = np.ascontiguousarray(delta_b, dtype=REAL_DTYPE)
        self.delta_q = np.ascontiguousarray(delta_q, dtype=REAL_DTYPE)
        self.delta_nr = int(delta_b.shape[0])
        self.min_signal = np.float32(min_signal)
        self.sampling_matrix = np.ascontiguousarray(sampling_matrix, dtype=REAL_DTYPE)
        self.b0s_mask = np.ascontiguousarray(b0s_mask, dtype=np.int32)

        self.shader_module = None
        self.getnum_pipeline = None
        self.gen_pipeline = None

        # Buffers created on setup_device
        self.H_buf = None
        self.R_buf = None
        self.delta_b_buf = None
        self.delta_q_buf = None
        self.b0s_mask_buf = None
        self.sampling_matrix_buf = None

    @classmethod
    def from_dipy_opdt(cls, gtab, sphere, sh_order_max=6, full_basis=False,
                       sh_lambda=0.006, min_signal=1):
        return cls(**prepare_opdt(gtab, sphere, sh_order_max, full_basis,
                                  sh_lambda, min_signal))

    @classmethod
    def from_dipy_csa(cls, gtab, sphere, sh_order_max=6, full_basis=False,
                      sh_lambda=0.006, min_signal=1):
        return cls(**prepare_csa(gtab, sphere, sh_order_max, full_basis,
                                 sh_lambda, min_signal))

    def _shader_files(self):
        return ["boot.wgsl"]

    def _kernel_files(self):
        # boot.wgsl is self-contained (has its own buffer bindings, params, entry points)
        return []

    def setup_device(self, device, has_subgroups=True):
        self.compile_program(device, has_subgroups)
        self.getnum_pipeline = self._make_pipeline(device, "getNumStreamlinesBoot_k")
        self.gen_pipeline = self._make_pipeline(device, "genStreamlinesMergeBoot_k")

        # Upload boot-specific data to GPU
        self.H_buf = create_buffer_from_data(device, self.H.ravel(), label="H")
        self.R_buf = create_buffer_from_data(device, self.R.ravel(), label="R")
        self.delta_b_buf = create_buffer_from_data(device, self.delta_b.ravel(), label="delta_b")
        self.delta_q_buf = create_buffer_from_data(device, self.delta_q.ravel(), label="delta_q")
        self.b0s_mask_buf = create_buffer_from_data(device, self.b0s_mask, label="b0s_mask")
        self.sampling_matrix_buf = create_buffer_from_data(
            device, self.sampling_matrix.ravel(), label="sampling_matrix"
        )

    def _make_params_bytes(self, sp, nseeds_gpu, for_gen=False):
        gt = sp.gpu_tracker
        rng_seed = gt.rng_seed
        rng_seed_lo = rng_seed & 0xFFFFFFFF
        rng_seed_hi = (rng_seed >> 32) & 0xFFFFFFFF

        # BootTrackingParams struct layout (must match WGSL struct in boot.wgsl)
        # float max_angle, tc_threshold, step_size, relative_peak_thresh,
        #       min_separation_angle, min_signal
        # int rng_seed_lo, rng_seed_hi, rng_offset, nseed
        # int dimx, dimy, dimz, dimt, samplm_nr, num_edges, delta_nr, model_type
        values = [
            gt.max_angle,
            gt.tc_threshold if for_gen else 0.0,
            gt.step_size if for_gen else 0.0,
            gt.relative_peak_thresh,
            gt.min_separation_angle,
            float(self.min_signal),
            rng_seed_lo,
            rng_seed_hi,
            gt.rng_offset if for_gen else 0,
            nseeds_gpu,
            gt.dimx, gt.dimy, gt.dimz, gt.dimt,
            gt.samplm_nr, gt.nedges, self.delta_nr, self.model_type,
        ]
        # 6 floats + 12 ints
        return struct.pack("6f12i", *values)

    def getNumStreamlines(self, nseeds_gpu, block, grid, sp):
        gt = sp.gpu_tracker
        device = gt.device
        params_bytes = self._make_params_bytes(sp, nseeds_gpu, for_gen=False)

        params_buf = device.create_buffer_with_data(
            data=params_bytes, usage="STORAGE | COPY_SRC"
        )

        # Boot getNum uses 3 bind groups. With layout="auto", only bindings
        # reachable from the entry point are included:
        #   Group 0: params(0), seeds(1), dataf(2), sphere_vertices(4), sphere_edges(5)
        #            metric_map(3) NOT used by getNum (only by tracker_boot → check_point_fn)
        #   Group 1: H(0), R(1), delta_b(2), delta_q(3), sampling_matrix(4), b0s_mask(5)
        #   Group 2: slineOutOff(0), shDir0(1)
        #            slineSeed(2), slineLen(3), sline(4) NOT used by getNum
        bg0 = device.create_bind_group(
            layout=self.getnum_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": params_buf}},
                {"binding": 1, "resource": {"buffer": sp.seeds_buf}},
                {"binding": 2, "resource": {"buffer": gt.dataf_buf}},
                {"binding": 4, "resource": {"buffer": gt.sphere_vertices_buf}},
                {"binding": 5, "resource": {"buffer": gt.sphere_edges_buf}},
            ],
        )

        bg1 = device.create_bind_group(
            layout=self.getnum_pipeline.get_bind_group_layout(1),
            entries=[
                {"binding": 0, "resource": {"buffer": self.H_buf}},
                {"binding": 1, "resource": {"buffer": self.R_buf}},
                {"binding": 2, "resource": {"buffer": self.delta_b_buf}},
                {"binding": 3, "resource": {"buffer": self.delta_q_buf}},
                {"binding": 4, "resource": {"buffer": self.sampling_matrix_buf}},
                {"binding": 5, "resource": {"buffer": self.b0s_mask_buf}},
            ],
        )

        bg2 = device.create_bind_group(
            layout=self.getnum_pipeline.get_bind_group_layout(2),
            entries=[
                {"binding": 0, "resource": {"buffer": sp.slinesOffs_buf}},
                {"binding": 1, "resource": {"buffer": sp.shDirTemp0_buf}},
            ],
        )

        self._dispatch_kernel(self.getnum_pipeline, [bg0, bg1, bg2], grid, device)

    def generateStreamlines(self, nseeds_gpu, block, grid, sp):
        gt = sp.gpu_tracker
        device = gt.device
        params_bytes = self._make_params_bytes(sp, nseeds_gpu, for_gen=True)

        params_buf = device.create_buffer_with_data(
            data=params_bytes, usage="STORAGE | COPY_SRC"
        )

        # Gen kernel uses all 17 buffers across 3 bind groups
        # Group 0: params, seeds, dataf, metric_map, sphere_vertices, sphere_edges
        bg0 = device.create_bind_group(
            layout=self.gen_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": params_buf}},
                {"binding": 1, "resource": {"buffer": sp.seeds_buf}},
                {"binding": 2, "resource": {"buffer": gt.dataf_buf}},
                {"binding": 3, "resource": {"buffer": gt.metric_map_buf}},
                {"binding": 4, "resource": {"buffer": gt.sphere_vertices_buf}},
                {"binding": 5, "resource": {"buffer": gt.sphere_edges_buf}},
            ],
        )

        # Group 1: H, R, delta_b, delta_q, sampling_matrix, b0s_mask
        bg1 = device.create_bind_group(
            layout=self.gen_pipeline.get_bind_group_layout(1),
            entries=[
                {"binding": 0, "resource": {"buffer": self.H_buf}},
                {"binding": 1, "resource": {"buffer": self.R_buf}},
                {"binding": 2, "resource": {"buffer": self.delta_b_buf}},
                {"binding": 3, "resource": {"buffer": self.delta_q_buf}},
                {"binding": 4, "resource": {"buffer": self.sampling_matrix_buf}},
                {"binding": 5, "resource": {"buffer": self.b0s_mask_buf}},
            ],
        )

        # Group 2: slineOutOff, shDir0, slineSeed, slineLen, sline
        bg2 = device.create_bind_group(
            layout=self.gen_pipeline.get_bind_group_layout(2),
            entries=[
                {"binding": 0, "resource": {"buffer": sp.slinesOffs_buf}},
                {"binding": 1, "resource": {"buffer": sp.shDirTemp0_buf}},
                {"binding": 2, "resource": {"buffer": sp.slineSeed_buf}},
                {"binding": 3, "resource": {"buffer": sp.slineLen_buf}},
                {"binding": 4, "resource": {"buffer": sp.sline_buf}},
            ],
        )

        self._dispatch_kernel(self.gen_pipeline, [bg0, bg1, bg2], grid, device)
