import numpy as np
from abc import ABC, abstractmethod
import logging
import ctypes
from importlib.resources import files
from time import time

from cuda.core import Device, LaunchConfig, Program, launch, ProgramOptions
from cuda.pathfinder import find_nvidia_header_directory
from cuda.cccl import get_include_paths
from cuda.bindings import runtime
from cuda.bindings.runtime import cudaMemcpyKind

from cuslines.cuda_python.cutils import (
    REAL_SIZE,
    REAL_DTYPE,
    REAL_DTYPE_AS_STR,
    REAL3_DTYPE_AS_STR,
    REAL_DTYPE_AS_CTYPE,
    checkCudaErrors,
    ModelType,
    THR_X_SL,
    BLOCK_Y,
    DEV_PTR,
)

logger = logging.getLogger("GPUStreamlines")


class GPUDirectionGetter(ABC):
    @abstractmethod
    def getNumStreamlines(self, n, nseeds_gpu, block, grid, sp):
        pass

    @abstractmethod
    def generateStreamlines(self):
        pass

    def allocate_on_gpu(self, n):
        pass

    def deallocate_on_gpu(self, n):
        pass

    def compile_program(self, debug: bool = False):
        start_time = time()
        logger.info("Compiling GPUStreamlines")

        cuslines_cuda = files("cuslines")

        if debug:
            program_opts = {
                "ptxas_options": ["-O0", "-v"],
                "device_code_optimize": True,
                "debug": True,
                "lineinfo": True,
            }
        else:
            program_opts = {
                "ptxas_options": ["-O3"]
            }

        program_options = ProgramOptions(
            name="cuslines",
            use_fast_math=True,
            std="c++17",
            define_macro="__NVRTC__",
            include_path=[
                str(cuslines_cuda),
                find_nvidia_header_directory("cudart"),
                find_nvidia_header_directory("curand"),
                get_include_paths().libcudacxx],
            **program_opts)

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
            ))
        logger.info("GPUStreamlines compiled successfully in %.2f seconds", time() - start_time)


class _BootCtx(ctypes.Structure):
    _fields_ = [
        ("min_signal", REAL_DTYPE_AS_CTYPE),
        ("delta_nr", ctypes.c_int32),
        ("H", ctypes.c_void_p),
        ("R", ctypes.c_void_p),
        ("delta_b", ctypes.c_void_p),
        ("delta_q", ctypes.c_void_p),
        ("sampling_matrix", ctypes.c_void_p),
        ("b0s_mask", ctypes.c_void_p),
    ]


class BootDirectionGetter(GPUDirectionGetter):
    def __init__(  # TODO: Maybe accept a dipy thing and extract arrays here? maybe as a from_ function?
            self,
            model_type: str,
            min_signal: float,
            H: np.ndarray,
            R: np.ndarray,
            delta_b: np.ndarray,
            delta_q: np.ndarray,
            sampling_matrix: np.ndarray,
            b0s_mask: np.ndarray):
        if model_type.upper() == "OPDT":
            self.model_type = int(ModelType.OPDT)
        elif model_type.upper() == "CSA":
            self.model_type = int(ModelType.CSA)
        else:
            raise ValueError(f"Invalid model_type {model_type}, must be one of 'OPDT', 'CSA'")

        self.H = np.ascontiguousarray(H, dtype=REAL_DTYPE)
        self.R = np.ascontiguousarray(R, dtype=REAL_DTYPE)
        self.delta_b = np.ascontiguousarray(delta_b, dtype=REAL_DTYPE)
        self.delta_q = np.ascontiguousarray(delta_q, dtype=REAL_DTYPE)
        self.delta_nr = int(delta_b.shape[0])
        self.min_signal = REAL_DTYPE(min_signal)
        self.sampling_matrix = np.ascontiguousarray(sampling_matrix, dtype=REAL_DTYPE)
        self.b0s_mask = np.ascontiguousarray(b0s_mask, dtype=np.int32)
        self.ctx_h = []

        self.H_d = []
        self.R_d = []
        self.delta_b_d = []
        self.delta_q_d = []
        self.b0s_mask_d = []
        self.sampling_matrix_d = []
        self.ctx_d = []

        self.getnum_kernel_name = f"getNumStreamlinesBoot_k<{THR_X_SL},{BLOCK_Y},{REAL_DTYPE_AS_STR},{REAL3_DTYPE_AS_STR}>"
        self.genstreamlines_kernel_name = f"genStreamlinesMerge_k<{THR_X_SL},{BLOCK_Y},{model_type.upper()},{REAL_DTYPE_AS_STR},{REAL3_DTYPE_AS_STR}>"
        self.compile_program()

    def allocate_on_gpu(self, n):
        self.H_d.append(
            checkCudaErrors(runtime.cudaMalloc(
                REAL_SIZE*self.H.size)))
        self.R_d.append(
            checkCudaErrors(runtime.cudaMalloc(
                REAL_SIZE*self.R.size)))
        self.delta_b_d.append(
            checkCudaErrors(runtime.cudaMalloc(
                REAL_SIZE*self.delta_b.size)))
        self.delta_q_d.append(
            checkCudaErrors(runtime.cudaMalloc(
                REAL_SIZE*self.delta_q.size)))
        self.b0s_mask_d.append(
            checkCudaErrors(runtime.cudaMalloc(
                np.int32().nbytes*self.b0s_mask.size)))
        self.sampling_matrix_d.append(
            checkCudaErrors(runtime.cudaMalloc(
                REAL_SIZE*self.sampling_matrix.size)))
        self.ctx_d.append(
            checkCudaErrors(runtime.cudaMalloc(
                ctypes.sizeof(_BootCtx))))
        self.ctx_h.append(_BootCtx(
            min_signal=self.min_signal,
            H=self.H_d[n],
            R=self.R_d[n],
            delta_b=self.delta_b_d[n],
            delta_q=self.delta_q_d[n],
            sampling_matrix=self.sampling_matrix_d[n],
            b0s_mask=self.b0s_mask_d[n],
        ))

        checkCudaErrors(runtime.cudaMemcpy(
            self.H_d[n],
            self.H.ctypes.data,
            REAL_SIZE*self.H.size,
            cudaMemcpyKind.cudaMemcpyHostToDevice))
        checkCudaErrors(runtime.cudaMemcpy(
            self.R_d[n],
            self.R.ctypes.data,
            REAL_SIZE*self.R.size,
            cudaMemcpyKind.cudaMemcpyHostToDevice))
        checkCudaErrors(runtime.cudaMemcpy(
            self.delta_b_d[n],
            self.delta_b.ctypes.data,
            REAL_SIZE*self.delta_b.size,
            cudaMemcpyKind.cudaMemcpyHostToDevice))
        checkCudaErrors(runtime.cudaMemcpy(
            self.delta_q_d[n],
            self.delta_q.ctypes.data,
            REAL_SIZE*self.delta_q.size,
            cudaMemcpyKind.cudaMemcpyHostToDevice))
        checkCudaErrors(runtime.cudaMemcpy(
            self.b0s_mask_d[n],
            self.b0s_mask.ctypes.data,
            np.int32().nbytes*self.b0s_mask.size,
            cudaMemcpyKind.cudaMemcpyHostToDevice))
        checkCudaErrors(runtime.cudaMemcpy(
            self.sampling_matrix_d[n],
            self.sampling_matrix.ctypes.data,
            REAL_SIZE*self.sampling_matrix.size,
            cudaMemcpyKind.cudaMemcpyHostToDevice))
        checkCudaErrors(runtime.cudaMemcpy(
            self.ctx_d[n],
            ctypes.byref(self.ctx_h[n]),
            ctypes.sizeof(_BootCtx),
            cudaMemcpyKind.cudaMemcpyHostToDevice
        ))

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
        if self.ctx_d[n]:
            checkCudaErrors(runtime.cudaFree(self.ctx_d[n]))

    def _shared_mem_bytes(self, sp):
        return REAL_SIZE*BLOCK_Y*2*(
            sp.gpu_tracker.n32dimt + max(sp.gpu_tracker.n32dimt, sp.gpu_tracker.samplm_nr)) + \
                np.int32().nbytes*BLOCK_Y*sp.gpu_tracker.samplm_nr

    def getNumStreamlines(self, n, nseeds_gpu, block, grid, sp):
        ker = self.module.get_kernel(self.getnum_kernel_name)
        shared_memory = self._shared_mem_bytes(sp)
        config = LaunchConfig(block=block, grid=grid, shmem_size=shared_memory)

        launch(
            sp.gpu_tracker.streams[n], config, ker,
            self.model_type,
            sp.gpu_tracker.max_angle,
            sp.gpu_tracker.min_separation_angle,
            sp.gpu_tracker.relative_peak_thresh,
            sp.gpu_tracker.rng_seed,
            nseeds_gpu,
            sp.seeds_d[n],
            sp.gpu_tracker.dimx,
            sp.gpu_tracker.dimy,
            sp.gpu_tracker.dimz,
            sp.gpu_tracker.dimt,
            sp.gpu_tracker.dataf_d[n].handle,
            self.H_d[n],
            self.R_d[n],
            self.delta_nr,
            self.delta_b_d[n],
            self.delta_q_d[n],
            self.b0s_mask_d[n],
            sp.gpu_tracker.samplm_nr,
            self.sampling_matrix_d[n],
            sp.gpu_tracker.sphere_vertices_d[n],
            sp.gpu_tracker.sphere_edges_d[n],
            sp.gpu_tracker.nedges,
            sp.shDirTemp0_d[n],
            sp.slinesOffs_d[n])

    def generateStreamlines(self, n, nseeds_gpu, block, grid, sp):
        ker = self.module.get_kernel(self.genstreamlines_kernel_name)
        shared_memory = self._shared_mem_bytes(sp)
        config = LaunchConfig(block=block, grid=grid, shmem_size=shared_memory)

        launch(
            sp.gpu_tracker.streams[n], config, ker,
            sp.gpu_tracker.max_angle,
            sp.gpu_tracker.tc_threshold,
            sp.gpu_tracker.step_size,
            sp.gpu_tracker.relative_peak_thresh,
            sp.gpu_tracker.min_separation_angle,
            sp.gpu_tracker.rng_seed,
            sp.gpu_tracker.rng_offset + n*nseeds_gpu,
            nseeds_gpu,
            sp.seeds_d[n],
            sp.gpu_tracker.dimx,
            sp.gpu_tracker.dimy,
            sp.gpu_tracker.dimz,
            sp.gpu_tracker.dimt,
            sp.gpu_tracker.dataf_d[n].handle,
            sp.gpu_tracker.metric_map_d[n],
            self.ctx_d[n],
            sp.gpu_tracker.samplm_nr,
            sp.gpu_tracker.sphere_vertices_d[n],
            sp.gpu_tracker.sphere_edges_d[n],
            sp.gpu_tracker.nedges,
            sp.slinesOffs_d[n],
            sp.shDirTemp0_d[n],
            sp.slineSeed_d[n],
            sp.slineLen_d[n],
            sp.sline_d[n]
        )


class ProbDirectionGetter(GPUDirectionGetter):
    def __init__(self):
        self.getnum_kernel_name = f"getNumStreamlinesProb_k<{THR_X_SL},{BLOCK_Y},{REAL_DTYPE_AS_STR},{REAL3_DTYPE_AS_STR}>"
        self.genstreamlines_kernel_name = f"genStreamlinesMerge_k<{THR_X_SL},{BLOCK_Y},PROB,{REAL_DTYPE_AS_STR},{REAL3_DTYPE_AS_STR}>"
        self.compile_program()

    def getNumStreamlines(self, n, nseeds_gpu, block, grid, sp):
        ker = self.module.get_kernel(self.getnum_kernel_name)
        shared_memory = REAL_SIZE*BLOCK_Y*sp.gpu_tracker.n32dimt + \
            np.int32().nbytes*BLOCK_Y*sp.gpu_tracker.n32dimt
        config = LaunchConfig(block=block, grid=grid, shmem_size=shared_memory)

        launch(
            sp.gpu_tracker.streams[n], config, ker,
            sp.gpu_tracker.max_angle,
            sp.gpu_tracker.relative_peak_thresh,
            sp.gpu_tracker.min_separation_angle,
            sp.gpu_tracker.rng_seed,
            nseeds_gpu,
            sp.seeds_d[n],
            sp.gpu_tracker.dimx,
            sp.gpu_tracker.dimy,
            sp.gpu_tracker.dimz,
            sp.gpu_tracker.dimt,
            sp.gpu_tracker.dataf_d[n].handle,
            sp.gpu_tracker.sphere_vertices_d[n],
            sp.gpu_tracker.sphere_edges_d[n],
            sp.gpu_tracker.nedges,
            sp.shDirTemp0_d[n],
            sp.slinesOffs_d[n])

    def _shared_mem_bytes(self, sp):
        return REAL_SIZE * BLOCK_Y * sp.gpu_tracker.n32dimt

    def generateStreamlines(self, n, nseeds_gpu, block, grid, sp):
        ker = self.module.get_kernel(self.genstreamlines_kernel_name)
        shared_memory = self._shared_mem_bytes(sp)
        config = LaunchConfig(block=block, grid=grid, shmem_size=shared_memory)

        launch(
            sp.gpu_tracker.streams[n], config, ker,
            sp.gpu_tracker.max_angle,
            sp.gpu_tracker.tc_threshold,
            sp.gpu_tracker.step_size,
            sp.gpu_tracker.relative_peak_thresh,
            sp.gpu_tracker.min_separation_angle,
            sp.gpu_tracker.rng_seed,
            sp.gpu_tracker.rng_offset + n*nseeds_gpu,
            nseeds_gpu,
            sp.seeds_d[n],
            sp.gpu_tracker.dimx,
            sp.gpu_tracker.dimy,
            sp.gpu_tracker.dimz,
            sp.gpu_tracker.dimt,
            sp.gpu_tracker.dataf_d[n].handle,
            sp.gpu_tracker.metric_map_d[n],
            int(0),
            sp.gpu_tracker.samplm_nr,
            sp.gpu_tracker.sphere_vertices_d[n],
            sp.gpu_tracker.sphere_edges_d[n],
            sp.gpu_tracker.nedges,
            sp.slinesOffs_d[n],
            sp.shDirTemp0_d[n],
            sp.slineSeed_d[n],
            sp.slineLen_d[n],
            sp.sline_d[n]
        )



class PttDirectionGetter(ProbDirectionGetter):
    def __init__(self):
        self.getnum_kernel_name = f"getNumStreamlinesProb_k<{THR_X_SL},{BLOCK_Y},{REAL_DTYPE_AS_STR},{REAL3_DTYPE_AS_STR}>"
        self.genstreamlines_kernel_name = f"genStreamlinesMerge_k<{THR_X_SL},{BLOCK_Y},PTT,{REAL_DTYPE_AS_STR},{REAL3_DTYPE_AS_STR}>"
        self.compile_program()

    def _shared_mem_bytes(self, sp):
        return 0

