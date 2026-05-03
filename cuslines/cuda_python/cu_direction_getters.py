import logging
import math
from abc import ABC, abstractmethod
from importlib.resources import files
from time import time

import numpy as np
from cuda.bindings import runtime
from cuda.bindings.runtime import cudaMemcpyKind
from cuda.cccl import get_include_paths
from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch
from cuda.pathfinder import find_nvidia_header_directory
from scipy.spatial import KDTree

from cuslines.boot_utils import prepare_csa, prepare_opdt
from cuslines.cuda_python.cutils import (
    BLOCK_Y,
    REAL3_DTYPE_AS_STR,
    REAL_DTYPE,
    REAL_DTYPE_AS_STR,
    ModelType,
    checkCudaErrors,
    EXCESS_ALLOC_FACT,
    MAX_SLINES_PER_SEED,
    MAX_SLINE_LEN,
    PMF_THRESHOLD_P,
    REAL_SIZE,
    THR_X_BL,
    THR_X_SL,
)

logger = logging.getLogger("GPUStreamlines")


class GPUDirectionGetter(ABC):
    @abstractmethod
    def getNumStreamlines(self, n, nseeds_gpu, block, grid, sp):
        pass

    @abstractmethod
    def generateStreamlines(self, n, nseeds_gpu, block, grid, sp):
        pass

    def set_macros(self, gpu_tracker):
        pass

    def allocate_on_gpu(self, n):
        pass

    def deallocate_on_gpu(self, n):
        pass

    def compile_program(self, gpu_tracker, debug: bool = False):
        start_time = time()
        logger.info("Compiling GPUStreamlines")

        cuslines_cuda = files("cuslines").joinpath("cuda_c")

        if debug:
            program_opts = {
                "ptxas_options": ["-O0", "-v"],
                "device_code_optimize": True,
                "debug": True,
                "lineinfo": True,
            }
        else:
            program_opts = {"ptxas_options": ["-O3"]}

        n32dimt = ((gpu_tracker.dimt + 31) // 32) * 32
        self.macros = {
            "__NVRTC__": None,
            "DIMX": str(int(gpu_tracker.dimx)),
            "DIMY": str(int(gpu_tracker.dimy)),
            "DIMZ": str(int(gpu_tracker.dimz)),
            "DIMT": str(int(gpu_tracker.dimt)),
            "N32DIMT": str(int(n32dimt)),
            "STEP_SIZE": str(float(gpu_tracker.step_size)),
            "MAX_ANGLE": str(float(gpu_tracker.max_angle)),
            "TC_THRESHOLD": str(float(gpu_tracker.tc_threshold)),
            "RELATIVE_PEAK_THRESH": str(float(gpu_tracker.relative_peak_thresh)),
            "MIN_SEPARATION_ANGLE": str(float(gpu_tracker.min_separation_angle)),
            "RNG_SEED": str(int(gpu_tracker.rng_seed)),
            "SAMPLM_NR": str(int(gpu_tracker.samplm_nr)),
            "NUM_EDGES": str(int(gpu_tracker.nedges)),
            "FULL_BASIS": "1" if gpu_tracker.full_basis else "0",
            "EXCESS_ALLOC_FACT": str(int(EXCESS_ALLOC_FACT)),
            "MAX_SLINES_PER_SEED": str(int(MAX_SLINES_PER_SEED)),
            "MAX_SLINE_LEN": str(int(MAX_SLINE_LEN)),
            "PMF_THRESHOLD_P": str(float(PMF_THRESHOLD_P)),
            "REAL_SIZE": str(int(REAL_SIZE)),
            "THR_X_BL": str(int(THR_X_BL)),
            "THR_X_SL": str(int(THR_X_SL)),
        }
        self.set_macros(gpu_tracker)
        optional_macros = [
            "log2_width",
            "width_mask",
            "probe_step_size",
            "max_curvature",
            "probe_quality",
            "step_frac",
        ]
        for name in optional_macros:
            if name.upper() not in self.macros:
                self.macros[name.upper()] = "0"
        if debug:
            self.macros["DEBUG"] = None

        program_options = ProgramOptions(
            name="cuslines",
            use_fast_math=True,
            std="c++17",
            define_macro=[
                f"{k}={v}" if v is not None else k for k, v in self.macros.items()
            ],
            include_path=[
                str(cuslines_cuda),
                find_nvidia_header_directory("cudart"),
                find_nvidia_header_directory("curand"),
                get_include_paths().libcudacxx,
            ],
            **program_opts,
        )

        # Here we assume all devices are the same,
        # so we compile once for any current device.
        # I think this is reasonable
        dev = Device()
        dev.set_current()
        cuda_path = cuslines_cuda.joinpath("generate_streamlines_cuda.cu")
        with open(cuda_path, "r") as f:
            prog = Program(f.read(), code_type="c++", options=program_options)
        self.module = prog.compile(
            "cubin",
            name_expressions=(
                self.getnum_kernel_name,
                self.genstreamlines_kernel_name,
            ),
        )
        logger.info(
            "GPUStreamlines compiled successfully in %.2f seconds", time() - start_time
        )


class BootDirectionGetter(GPUDirectionGetter):
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
        if model_type.upper() == "OPDT":
            self.model_type = int(ModelType.OPDT)
        elif model_type.upper() == "CSA":
            self.model_type = int(ModelType.CSA)
        else:
            raise ValueError(
                f"Invalid model_type {model_type}, must be one of 'OPDT', 'CSA'"
            )

        self.H = np.ascontiguousarray(H, dtype=REAL_DTYPE)
        self.R = np.ascontiguousarray(R, dtype=REAL_DTYPE)
        self.delta_b = np.ascontiguousarray(delta_b, dtype=REAL_DTYPE)
        self.delta_q = np.ascontiguousarray(delta_q, dtype=REAL_DTYPE)
        self.delta_nr = int(delta_b.shape[0])
        self.min_signal = REAL_DTYPE(min_signal)
        self.sampling_matrix = np.ascontiguousarray(sampling_matrix, dtype=REAL_DTYPE)
        self.b0s_mask = np.ascontiguousarray(b0s_mask, dtype=np.int32)

        self.H_d = []
        self.R_d = []
        self.delta_b_d = []
        self.delta_q_d = []
        self.b0s_mask_d = []
        self.sampling_matrix_d = []

        self.getnum_kernel_name = f"getNumStreamlinesBoot_k<{THR_X_SL},{BLOCK_Y},{REAL_DTYPE_AS_STR},{REAL3_DTYPE_AS_STR}>"
        self.genstreamlines_kernel_name = f"genStreamlinesMergeBoot_k<{THR_X_SL},{BLOCK_Y},{model_type.upper()},{REAL_DTYPE_AS_STR},{REAL3_DTYPE_AS_STR}>"

    @classmethod
    def from_dipy_opdt(
        cls,
        gtab,
        sphere,
        sh_order_max=6,
        full_basis=False,
        sh_lambda=0.006,
        min_signal=1,
    ):
        return cls(
            **prepare_opdt(
                gtab, sphere, sh_order_max, full_basis, sh_lambda, min_signal
            )
        )

    @classmethod
    def from_dipy_csa(
        cls,
        gtab,
        sphere,
        sh_order_max=6,
        full_basis=False,
        sh_lambda=0.006,
        min_signal=1,
    ):
        return cls(
            **prepare_csa(gtab, sphere, sh_order_max, full_basis, sh_lambda, min_signal)
        )

    def allocate_on_gpu(self, n):
        self.H_d.append(checkCudaErrors(runtime.cudaMalloc(REAL_SIZE * self.H.size)))
        self.R_d.append(checkCudaErrors(runtime.cudaMalloc(REAL_SIZE * self.R.size)))
        self.delta_b_d.append(
            checkCudaErrors(runtime.cudaMalloc(REAL_SIZE * self.delta_b.size))
        )
        self.delta_q_d.append(
            checkCudaErrors(runtime.cudaMalloc(REAL_SIZE * self.delta_q.size))
        )
        self.b0s_mask_d.append(
            checkCudaErrors(runtime.cudaMalloc(np.int32().nbytes * self.b0s_mask.size))
        )
        self.sampling_matrix_d.append(
            checkCudaErrors(runtime.cudaMalloc(REAL_SIZE * self.sampling_matrix.size))
        )

        checkCudaErrors(
            runtime.cudaMemcpy(
                self.H_d[n],
                self.H.ctypes.data,
                REAL_SIZE * self.H.size,
                cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        )
        checkCudaErrors(
            runtime.cudaMemcpy(
                self.R_d[n],
                self.R.ctypes.data,
                REAL_SIZE * self.R.size,
                cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        )
        checkCudaErrors(
            runtime.cudaMemcpy(
                self.delta_b_d[n],
                self.delta_b.ctypes.data,
                REAL_SIZE * self.delta_b.size,
                cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        )
        checkCudaErrors(
            runtime.cudaMemcpy(
                self.delta_q_d[n],
                self.delta_q.ctypes.data,
                REAL_SIZE * self.delta_q.size,
                cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        )
        checkCudaErrors(
            runtime.cudaMemcpy(
                self.b0s_mask_d[n],
                self.b0s_mask.ctypes.data,
                np.int32().nbytes * self.b0s_mask.size,
                cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        )
        checkCudaErrors(
            runtime.cudaMemcpy(
                self.sampling_matrix_d[n],
                self.sampling_matrix.ctypes.data,
                REAL_SIZE * self.sampling_matrix.size,
                cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        )

    def deallocate_on_gpu(self, n):
        if self.H_d[n]:
            checkCudaErrors(runtime.cudaFree(self.H_d[n]))
        if self.R_d[n]:
            checkCudaErrors(runtime.cudaFree(self.R_d[n]))
        if self.delta_b_d[n]:
            checkCudaErrors(runtime.cudaFree(self.delta_b_d[n]))
        if self.delta_q_d[n]:
            checkCudaErrors(runtime.cudaFree(self.delta_q_d[n]))
        if self.b0s_mask_d[n]:
            checkCudaErrors(runtime.cudaFree(self.b0s_mask_d[n]))
        if self.sampling_matrix_d[n]:
            checkCudaErrors(runtime.cudaFree(self.sampling_matrix_d[n]))

    def _shared_mem_bytes(self, sp):
        return (
            REAL_SIZE
            * BLOCK_Y
            * 2
            * (
                sp.gpu_tracker.n32dimt
                + max(sp.gpu_tracker.n32dimt, sp.gpu_tracker.samplm_nr)
            )
            + np.int32().nbytes * BLOCK_Y * sp.gpu_tracker.samplm_nr
        )

    def getNumStreamlines(self, n, nseeds_gpu, block, grid, sp):
        ker = self.module.get_kernel(self.getnum_kernel_name)
        shared_memory = self._shared_mem_bytes(sp)
        config = LaunchConfig(block=block, grid=grid, shmem_size=shared_memory)

        launch(
            sp.gpu_tracker.streams[n],
            config,
            ker,
            self.model_type,
            self.min_signal,
            nseeds_gpu,
            sp.seeds_d[n],
            sp.gpu_tracker.dataf_d[n],
            self.H_d[n],
            self.R_d[n],
            self.delta_nr,
            self.delta_b_d[n],
            self.delta_q_d[n],
            self.b0s_mask_d[n],
            self.sampling_matrix_d[n],
            sp.gpu_tracker.sphere_vertices_d[n],
            sp.gpu_tracker.sphere_edges_d[n],
            sp.shDirTemp0_d[n],
            sp.slinesOffs_d[n],
        )

    def generateStreamlines(self, n, nseeds_gpu, block, grid, sp):
        ker = self.module.get_kernel(self.genstreamlines_kernel_name)
        shared_memory = self._shared_mem_bytes(sp)
        config = LaunchConfig(block=block, grid=grid, shmem_size=shared_memory)

        launch(
            sp.gpu_tracker.streams[n],
            config,
            ker,
            sp.gpu_tracker.rng_offset + n * nseeds_gpu,
            nseeds_gpu,
            sp.seeds_d[n],
            sp.gpu_tracker.dataf_d[n],
            sp.gpu_tracker.metric_map_d[n].getPtr(),
            sp.gpu_tracker.sphere_vertices_d[n],
            sp.gpu_tracker.sphere_edges_d[n],
            self.min_signal,
            self.delta_nr,
            self.H_d[n],
            self.R_d[n],
            self.delta_b_d[n],
            self.delta_q_d[n],
            self.sampling_matrix_d[n],
            self.b0s_mask_d[n],
            sp.slinesOffs_d[n],
            sp.shDirTemp0_d[n],
            sp.slineSeed_d[n],
            sp.slineLen_d[n],
            sp.sline_d[n],
        )


class ProbDirectionGetter(GPUDirectionGetter):
    def __init__(self):
        self.getnum_kernel_name = f"getNumStreamlinesProb_k<{THR_X_SL},{BLOCK_Y},{REAL_DTYPE_AS_STR},{REAL3_DTYPE_AS_STR}>"
        self.genstreamlines_kernel_name = f"genStreamlinesMergeProb_k<{THR_X_SL},{BLOCK_Y},PROB,const {REAL_DTYPE_AS_STR} *__restrict__,{REAL_DTYPE_AS_STR},{REAL3_DTYPE_AS_STR}>"

    def getNumStreamlines(self, n, nseeds_gpu, block, grid, sp):
        ker = self.module.get_kernel(self.getnum_kernel_name)
        shared_memory = (
            REAL_SIZE * BLOCK_Y * sp.gpu_tracker.n32dimt
            + np.int32().nbytes * BLOCK_Y * sp.gpu_tracker.n32dimt
        )
        config = LaunchConfig(block=block, grid=grid, shmem_size=shared_memory)

        if isinstance(sp.gpu_tracker.dataf_d[n], runtime.cudaTextureObject_t):
            dataf_d_n = sp.gpu_tracker.dataf_d[n].getPtr()
        else:
            dataf_d_n = sp.gpu_tracker.dataf_d[n]

        launch(
            sp.gpu_tracker.streams[n],
            config,
            ker,
            nseeds_gpu,
            sp.seeds_d[n],
            dataf_d_n,
            sp.gpu_tracker.sphere_vertices_d[n],
            sp.gpu_tracker.sphere_edges_d[n],
            sp.shDirTemp0_d[n],
            sp.slinesOffs_d[n],
        )

    def generateStreamlines(self, n, nseeds_gpu, block, grid, sp):
        ker = self.module.get_kernel(self.genstreamlines_kernel_name)
        shared_memory = REAL_SIZE * BLOCK_Y * sp.gpu_tracker.n32dimt
        config = LaunchConfig(block=block, grid=grid, shmem_size=shared_memory)

        launch(
            sp.gpu_tracker.streams[n],
            config,
            ker,
            sp.gpu_tracker.rng_offset + n * nseeds_gpu,
            nseeds_gpu,
            sp.seeds_d[n],
            sp.gpu_tracker.dataf_d[n],
            sp.gpu_tracker.metric_map_d[n].getPtr(),
            sp.gpu_tracker.sphere_vertices_d[n],
            sp.gpu_tracker.sphere_edges_d[n],
            sp.slinesOffs_d[n],
            sp.shDirTemp0_d[n],
            sp.slineSeed_d[n],
            sp.slineLen_d[n],
            sp.sline_d[n],
        )


class PttDirectionGetter(ProbDirectionGetter):
    def __init__(
        self,
        odf_lut_res: int = 128,
        probe_length: float = 0.25,
        target_short_step: float = 0.025,
        probe_quality: int = 8,
    ):
        """
        Use Parallel Transport Tractography

        Parameters
        ----------
        odf_lut_res: int
            Resolution of the ODF lookup table.
            Default: 128
        probe_length: int
            Length of probing steps as fraction of voxel size.
            Default: 0.25
        target_short_step : float
            Length of target short steps as fraction of voxel size.
            Actual short probing steps will be chosen such that step size
            is an integer multiple of the short step.
            Default: 0.025
        probe_quality : float
            Number of probing steps.
            Default: 8
        """
        self.getnum_kernel_name = f"getNumStreamlinesPtt_k<{THR_X_SL},{BLOCK_Y},{REAL_DTYPE_AS_STR},{REAL3_DTYPE_AS_STR}>"
        self.genstreamlines_kernel_name = f"genStreamlinesMergeProb_k<{THR_X_SL},{BLOCK_Y},PTT,const cudaTextureObject_t *__restrict__,{REAL_DTYPE_AS_STR},{REAL3_DTYPE_AS_STR}>"
        self.odf_lut_res = odf_lut_res
        self.sphere_vertices_lut_h = None
        self.sphere_vertices_lut_d = []
        self.sphere_vertices_lut_array_d = []

        self.probe_length = probe_length
        self.probe_quality = probe_quality
        self.target_short_step = target_short_step

    def set_macros(self, gpu_tracker):
        self.macros["LOG2_WIDTH"] = str(int(self.log2_width))
        self.macros["WIDTH_MASK"] = str(int(self.width_mask))
        self.macros["PROBE_QUALITY"] = str(float(self.probe_quality))
        self.macros["PROBE_STEP_SIZE"] = str(float(self.probe_length))
        self.macros["STEP_FRAC"] = str(
            int(np.round(gpu_tracker.step_size / self.target_short_step))
        )
        self.macros["MAX_CURVATURE"] = str(
            float(8.0 * np.sin(gpu_tracker.max_angle / 2.0))
        )

    def allocate_on_gpu(self, n):
        if REAL_SIZE != 4:
            raise ValueError(
                ("PTT on CUDA uses texture memory which only supports 32-bit floats")
            )

        channelDesc = checkCudaErrors(
            runtime.cudaCreateChannelDesc(
                32, 0, 0, 0, runtime.cudaChannelFormatKind.cudaChannelFormatKindFloat
            )
        )
        extent = runtime.make_cudaExtent(
            self.odf_lut_res, self.odf_lut_res, self.odf_lut_res
        )
        sphere_vertices_array = checkCudaErrors(
            runtime.cudaMalloc3DArray(channelDesc, extent, 0)
        )

        copyParams = runtime.cudaMemcpy3DParms()
        copyParams.srcPtr = runtime.make_cudaPitchedPtr(
            self.sphere_vertices_lut_h.ctypes.data,
            self.odf_lut_res * 4,
            self.odf_lut_res,
            self.odf_lut_res,
        )

        copyParams.dstArray = sphere_vertices_array
        copyParams.extent = extent
        copyParams.kind = cudaMemcpyKind.cudaMemcpyHostToDevice
        checkCudaErrors(runtime.cudaMemcpy3D(copyParams))

        resDesc = runtime.cudaResourceDesc()
        resDesc.resType = runtime.cudaResourceType.cudaResourceTypeArray
        resDesc.res.array.array = sphere_vertices_array

        texDesc = runtime.cudaTextureDesc()
        texDesc.addressMode[0] = runtime.cudaTextureAddressMode.cudaAddressModeClamp
        texDesc.addressMode[1] = runtime.cudaTextureAddressMode.cudaAddressModeClamp
        texDesc.addressMode[2] = runtime.cudaTextureAddressMode.cudaAddressModeClamp
        texDesc.filterMode = runtime.cudaTextureFilterMode.cudaFilterModePoint
        texDesc.readMode = runtime.cudaTextureReadMode.cudaReadModeElementType
        texDesc.normalizedCoords = 1

        texObj = checkCudaErrors(
            runtime.cudaCreateTextureObject(resDesc, texDesc, None)
        )
        self.sphere_vertices_lut_d.append(texObj)
        self.sphere_vertices_lut_array_d.append(sphere_vertices_array)

    def deallocate_on_gpu(self, n):
        if self.sphere_vertices_lut_d[n]:
            checkCudaErrors(
                runtime.cudaDestroyTextureObject(self.sphere_vertices_lut_d[n]),
                hard_error=False,
            )
        if self.sphere_vertices_lut_array_d[n]:
            checkCudaErrors(
                runtime.cudaFreeArray(self.sphere_vertices_lut_array_d[n]),
                hard_error=False,
            )

    def prepare_data(self, dataf, stop_map, stop_threshold, sphere_vertices):
        dimx, dimy, dimz, dimt = dataf.shape
        dataf = dataf.clip(min=0)

        # zeros outside of tracking mask helps with probing
        dataf[stop_map < stop_threshold, :] = 0

        # normalize ODFs to max of 1
        odf_sums = dataf.max(axis=3, keepdims=True)
        nonzero_mask = odf_sums > 0
        np.divide(dataf, odf_sums, out=dataf, where=nonzero_mask)

        # This rearrangement is for cuda texture memory
        # In particular, for texture memory, we want each dimension
        # to be less than 65,535, so we tile t across x and y
        # additionally, we then make the tiles in the x dim
        # a power of 2 to ensure it is fast to calculate indices
        # into the tiles
        ideal_tiles_per_row = math.ceil(math.sqrt(dimt))
        self.log2_width = math.ceil(math.log2(ideal_tiles_per_row))
        tiles_per_row = 1 << self.log2_width
        self.width_mask = tiles_per_row - 1
        tiles_per_col = math.ceil(dimt / tiles_per_row)
        total_slots = tiles_per_row * tiles_per_col
        if dimt < total_slots:
            padding = np.zeros((dimx, dimy, dimz, total_slots - dimt), dtype=np.float32)
            data_f_rearranged = np.concatenate([dataf, padding], axis=3)
        else:
            data_f_rearranged = dataf

        total_memory_usage_gb = (
            (tiles_per_row * dimx) * (tiles_per_col * dimy) * dimz * 4 / 1e9
        )
        logger.info(
            (
                f"For PTT, we will allocate a 3D texture of size "
                f"{tiles_per_row * dimx}x{tiles_per_col * dimy}x{dimz} "
                "to store the ODFs on the GPU. This will be in 4 byte floats and use "
                f"{total_memory_usage_gb:.2f} GB of GPU memory. "
                "If this is too near your total GPU memory, it will error"
            )
        )
        data_f_rearranged = data_f_rearranged.reshape(
            dimx, dimy, dimz, tiles_per_col, tiles_per_row
        )
        data_f_rearranged = data_f_rearranged.transpose(2, 3, 1, 4, 0).reshape(
            dimz, tiles_per_col * dimy, tiles_per_row * dimx
        )
        data_f_rearranged = np.ascontiguousarray(data_f_rearranged, dtype=np.float32)

        # Generate a 3D LUT that maps each point in a 128x128x128 grid to
        # the index of the closest sphere vertex
        coords = np.linspace(-1, 1, self.odf_lut_res)
        grid_x, grid_y, grid_z = np.meshgrid(coords, coords, coords, indexing="ij")
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)

        tree = KDTree(sphere_vertices)
        _, closest_indices = tree.query(grid_points)
        lut = closest_indices.reshape(
            (self.odf_lut_res, self.odf_lut_res, self.odf_lut_res)
        )
        lut = np.ascontiguousarray(lut, dtype=np.float32)
        self.sphere_vertices_lut_h = lut

        return data_f_rearranged

    def generateStreamlines(self, n, nseeds_gpu, block, grid, sp):
        ker = self.module.get_kernel(self.genstreamlines_kernel_name)
        shared_memory = 0
        config = LaunchConfig(block=block, grid=grid, shmem_size=shared_memory)

        launch(
            sp.gpu_tracker.streams[n],
            config,
            ker,
            sp.gpu_tracker.rng_offset + n * nseeds_gpu,
            nseeds_gpu,
            sp.seeds_d[n],
            sp.gpu_tracker.dataf_d[n].getPtr(),
            sp.gpu_tracker.metric_map_d[n].getPtr(),
            self.sphere_vertices_lut_d[n].getPtr(),
            sp.gpu_tracker.sphere_edges_d[n],
            sp.slinesOffs_d[n],
            sp.shDirTemp0_d[n],
            sp.slineSeed_d[n],
            sp.slineLen_d[n],
            sp.sline_d[n],
        )
