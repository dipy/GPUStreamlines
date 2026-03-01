"""Metal direction getters — mirrors cuslines/cuda_python/cu_direction_getters.py.

Compiles MSL shaders at runtime and dispatches kernel launches via
MTLComputeCommandEncoder.
"""

import numpy as np
import struct
from abc import ABC, abstractmethod
import logging
from importlib.resources import files
from time import time

from cuslines.boot_utils import prepare_opdt, prepare_csa

from cuslines.metal.mutils import (
    REAL_SIZE,
    REAL_DTYPE,
    REAL3_SIZE,
    BLOCK_Y,
    THR_X_SL,
    div_up,
    checkMetalError,
)

logger = logging.getLogger("GPUStreamlines")


class MetalGPUDirectionGetter(ABC):
    """Abstract base for Metal direction getters."""

    # Soft angular weighting factor for bootstrap direction getters.
    # 0.0 = disabled (match CPU behavior), 0.5 = moderate bias toward
    # current trajectory at fiber crossings.
    angular_weight = 0.0

    @abstractmethod
    def getNumStreamlines(self, nseeds_gpu, block, grid, sp):
        pass

    @abstractmethod
    def generateStreamlines(self, nseeds_gpu, block, grid, sp):
        pass

    def setup_device(self, device):
        """Called once when GPUTracker allocates resources."""
        pass

    def compile_program(self, device):
        import Metal
        import re

        start_time = time()
        logger.info("Compiling Metal shaders...")

        shader_dir = files("cuslines").joinpath("metal_shaders")

        # Read header files in dependency order and inline them.
        # Metal's runtime compiler doesn't support include search paths,
        # so we prepend all headers and strip #include "..." directives.
        header_files = [
            "globals.h",
            "types.h",
            "philox_rng.h",
        ]
        # Add disc.h if boot.metal or ptt.metal is in the shader set
        if "boot.metal" in self._shader_files() or "ptt.metal" in self._shader_files():
            header_files.append("disc.h")

        source_parts = []
        for fname in header_files:
            path = shader_dir.joinpath(fname)
            with open(path, "r") as f:
                source_parts.append(f"// ── {fname} ──\n")
                source_parts.append(f.read())

        # Metal source files
        metal_files = [
            "utils.metal",
            "warp_sort.metal",
            "tracking_helpers.metal",
        ]
        metal_files += self._shader_files()
        metal_files.append("generate_streamlines_metal.metal")

        for fname in metal_files:
            path = shader_dir.joinpath(fname)
            with open(path, "r") as f:
                src = f.read()
            # Strip local #include directives (headers already inlined above)
            src = re.sub(r'#include\s+"[^"]*"', '', src)
            source_parts.append(f"// ── {fname} ──\n")
            source_parts.append(src)

        full_source = "\n".join(source_parts)

        # Prepend compile-time constants
        enable = 1 if self.angular_weight > 0 else 0
        defines = (
            f"#define ENABLE_ANGULAR_WEIGHT {enable}\n"
            f"#define ANGULAR_WEIGHT {self.angular_weight:.2f}f\n"
        )
        full_source = defines + full_source

        options = Metal.MTLCompileOptions.new()
        options.setFastMathEnabled_(True)

        library, error = device.newLibraryWithSource_options_error_(
            full_source, options, None
        )
        if error is not None:
            raise RuntimeError(f"Metal shader compilation failed: {error}")

        self.library = library
        logger.info("Metal shaders compiled in %.2f seconds", time() - start_time)

    def _shader_files(self):
        """Return list of additional .metal files needed by this direction getter."""
        return []

    def _make_pipeline(self, device, kernel_name):
        import Metal

        fn = self.library.newFunctionWithName_(kernel_name)
        if fn is None:
            raise RuntimeError(f"Metal kernel '{kernel_name}' not found in library")
        pipeline, error = device.newComputePipelineStateWithFunction_error_(fn, None)
        if error is not None:
            raise RuntimeError(f"Failed to create pipeline for '{kernel_name}': {error}")
        return pipeline

    @staticmethod
    def _check_cmd_buf(cmd_buf, kernel_name=""):
        """Check command buffer status after waitUntilCompleted."""
        import Metal

        status = cmd_buf.status()
        if status == Metal.MTLCommandBufferStatusError:
            error = cmd_buf.error()
            raise RuntimeError(
                f"Metal command buffer error in {kernel_name}: {error}"
            )


class MetalProbDirectionGetter(MetalGPUDirectionGetter):
    """Probabilistic direction getter for Metal."""

    def __init__(self):
        self.library = None
        self.getnum_pipeline = None
        self.gen_pipeline = None

    def _shader_files(self):
        return []

    def setup_device(self, device):
        self.compile_program(device)
        self.getnum_pipeline = self._make_pipeline(device, "getNumStreamlinesProb_k")
        self.gen_pipeline = self._make_pipeline(device, "genStreamlinesMergeProb_k")

    def _make_params_bytes(self, sp, nseeds_gpu, for_gen=False):
        gt = sp.gpu_tracker
        rng_seed = gt.rng_seed
        rng_seed_lo = rng_seed & 0xFFFFFFFF
        rng_seed_hi = (rng_seed >> 32) & 0xFFFFFFFF

        # ProbTrackingParams struct layout (must match Metal struct)
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
        # 5 floats + 11 ints
        return struct.pack("5f11i", *values)

    def getNumStreamlines(self, nseeds_gpu, block, grid, sp):
        import Metal

        gt = sp.gpu_tracker
        params_bytes = self._make_params_bytes(sp, nseeds_gpu, for_gen=False)

        cmd_buf = gt.command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self.getnum_pipeline)

        encoder.setBytes_length_atIndex_(params_bytes, len(params_bytes), 0)
        encoder.setBuffer_offset_atIndex_(sp.seeds_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(gt.dataf_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(gt.sphere_vertices_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(gt.sphere_edges_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(sp.shDirTemp0_buf, 0, 5)
        encoder.setBuffer_offset_atIndex_(sp.slinesOffs_buf, 0, 6)

        threads_per_group = Metal.MTLSize(block[0], block[1], block[2])
        groups = Metal.MTLSize(grid[0], grid[1], grid[2])
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(groups, threads_per_group)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        self._check_cmd_buf(cmd_buf, "getNumStreamlinesProb_k")

    def generateStreamlines(self, nseeds_gpu, block, grid, sp):
        import Metal

        gt = sp.gpu_tracker
        params_bytes = self._make_params_bytes(sp, nseeds_gpu, for_gen=True)

        cmd_buf = gt.command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self.gen_pipeline)

        encoder.setBytes_length_atIndex_(params_bytes, len(params_bytes), 0)
        encoder.setBuffer_offset_atIndex_(sp.seeds_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(gt.dataf_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(gt.metric_map_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(gt.sphere_vertices_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(gt.sphere_edges_buf, 0, 5)
        encoder.setBuffer_offset_atIndex_(sp.slinesOffs_buf, 0, 6)
        encoder.setBuffer_offset_atIndex_(sp.shDirTemp0_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(sp.slineSeed_buf, 0, 8)
        encoder.setBuffer_offset_atIndex_(sp.slineLen_buf, 0, 9)
        encoder.setBuffer_offset_atIndex_(sp.sline_buf, 0, 10)

        threads_per_group = Metal.MTLSize(block[0], block[1], block[2])
        groups = Metal.MTLSize(grid[0], grid[1], grid[2])
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(groups, threads_per_group)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        self._check_cmd_buf(cmd_buf, "genStreamlinesMergeProb_k")


class MetalPttDirectionGetter(MetalProbDirectionGetter):
    """PTT direction getter for Metal."""

    def _shader_files(self):
        return ["ptt.metal"]

    def setup_device(self, device):
        self.compile_program(device)
        # PTT reuses Prob's getNum kernel for initial direction finding
        self.getnum_pipeline = self._make_pipeline(device, "getNumStreamlinesProb_k")
        # PTT has its own gen kernel with parallel transport frame tracking
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


class MetalBootDirectionGetter(MetalGPUDirectionGetter):
    """Bootstrap direction getter for Metal."""

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

        self.library = None
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
        return ["boot.metal"]

    def setup_device(self, device):
        from cuslines.metal.mt_tractography import _make_shared_buffer

        self.compile_program(device)
        self.getnum_pipeline = self._make_pipeline(device, "getNumStreamlinesBoot_k")
        self.gen_pipeline = self._make_pipeline(device, "genStreamlinesMergeBoot_k")

        # Create shared buffers for boot-specific data
        self.H_buf = _make_shared_buffer(device, self.H)
        self.R_buf = _make_shared_buffer(device, self.R)
        self.delta_b_buf = _make_shared_buffer(device, self.delta_b)
        self.delta_q_buf = _make_shared_buffer(device, self.delta_q)
        self.b0s_mask_buf = _make_shared_buffer(device, self.b0s_mask)
        self.sampling_matrix_buf = _make_shared_buffer(device, self.sampling_matrix)

    def _make_params_bytes(self, sp, nseeds_gpu, for_gen=False):
        gt = sp.gpu_tracker
        rng_seed = gt.rng_seed
        rng_seed_lo = rng_seed & 0xFFFFFFFF
        rng_seed_hi = (rng_seed >> 32) & 0xFFFFFFFF

        # BootTrackingParams struct layout (must match Metal struct in boot.metal)
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

    def _boot_sh_pool_bytes(self, gt):
        """Compute dynamic threadgroup memory size for boot kernels."""
        n32dimt = ((gt.dimt + 31) // 32) * 32
        sh_per_row = 2 * n32dimt + 2 * max(n32dimt, gt.samplm_nr)
        return BLOCK_Y * sh_per_row * REAL_SIZE  # bytes

    def getNumStreamlines(self, nseeds_gpu, block, grid, sp):
        import Metal

        gt = sp.gpu_tracker
        params_bytes = self._make_params_bytes(sp, nseeds_gpu, for_gen=False)

        cmd_buf = gt.command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self.getnum_pipeline)

        # Buffer bindings match getNumStreamlinesBoot_k signature in boot.metal
        encoder.setBytes_length_atIndex_(params_bytes, len(params_bytes), 0)
        encoder.setBuffer_offset_atIndex_(sp.seeds_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(gt.dataf_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.H_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(self.R_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(self.delta_b_buf, 0, 5)
        encoder.setBuffer_offset_atIndex_(self.delta_q_buf, 0, 6)
        encoder.setBuffer_offset_atIndex_(self.b0s_mask_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(self.sampling_matrix_buf, 0, 8)
        encoder.setBuffer_offset_atIndex_(gt.sphere_vertices_buf, 0, 9)
        encoder.setBuffer_offset_atIndex_(gt.sphere_edges_buf, 0, 10)
        encoder.setBuffer_offset_atIndex_(sp.shDirTemp0_buf, 0, 11)
        encoder.setBuffer_offset_atIndex_(sp.slinesOffs_buf, 0, 12)

        # Dynamic threadgroup memory (replaces CUDA extern __shared__)
        encoder.setThreadgroupMemoryLength_atIndex_(self._boot_sh_pool_bytes(gt), 0)

        threads_per_group = Metal.MTLSize(block[0], block[1], block[2])
        groups = Metal.MTLSize(grid[0], grid[1], grid[2])
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(groups, threads_per_group)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        self._check_cmd_buf(cmd_buf, "getNumStreamlinesBoot_k")

    def generateStreamlines(self, nseeds_gpu, block, grid, sp):
        import Metal

        gt = sp.gpu_tracker
        params_bytes = self._make_params_bytes(sp, nseeds_gpu, for_gen=True)

        cmd_buf = gt.command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self.gen_pipeline)

        # Buffer bindings match genStreamlinesMergeBoot_k signature in boot.metal
        encoder.setBytes_length_atIndex_(params_bytes, len(params_bytes), 0)
        encoder.setBuffer_offset_atIndex_(sp.seeds_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(gt.dataf_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(gt.metric_map_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(gt.sphere_vertices_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(gt.sphere_edges_buf, 0, 5)
        encoder.setBuffer_offset_atIndex_(self.H_buf, 0, 6)
        encoder.setBuffer_offset_atIndex_(self.R_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(self.delta_b_buf, 0, 8)
        encoder.setBuffer_offset_atIndex_(self.delta_q_buf, 0, 9)
        encoder.setBuffer_offset_atIndex_(self.sampling_matrix_buf, 0, 10)
        encoder.setBuffer_offset_atIndex_(self.b0s_mask_buf, 0, 11)
        encoder.setBuffer_offset_atIndex_(sp.slinesOffs_buf, 0, 12)
        encoder.setBuffer_offset_atIndex_(sp.shDirTemp0_buf, 0, 13)
        encoder.setBuffer_offset_atIndex_(sp.slineSeed_buf, 0, 14)
        encoder.setBuffer_offset_atIndex_(sp.slineLen_buf, 0, 15)
        encoder.setBuffer_offset_atIndex_(sp.sline_buf, 0, 16)

        # Dynamic threadgroup memory (replaces CUDA extern __shared__)
        encoder.setThreadgroupMemoryLength_atIndex_(self._boot_sh_pool_bytes(gt), 0)

        threads_per_group = Metal.MTLSize(block[0], block[1], block[2])
        groups = Metal.MTLSize(grid[0], grid[1], grid[2])
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(groups, threads_per_group)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        self._check_cmd_buf(cmd_buf, "genStreamlinesMergeBoot_k")
