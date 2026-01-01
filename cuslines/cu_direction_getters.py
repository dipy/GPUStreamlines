import numpy as np
from abc import ABC, abstractmethod
import logging
from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

from cuda.bindings import runtime
from cuda.core import Device

from cutils import (
    REAL_SIZE,
    REAL_DTYPE,
    checkCudaErrors,
)


__all__ = [
    "ProbDirectionGetter",
    "PTTDirectionGetter",
    "BootDirectionGetter"
]


logger = logging.getLogger("GPUStreamlines")


_program = None


def _compile_program(debug=False):  # TODO: compile kernels individually as needed
    if _program is None:
        logger.info("Compiling GPUStreamlines")
        dev = Device()
        dev.set_current()

        if debug:
            comp_kwargs = {
                "debug": True,
                "lineinfo": True,
                "device_code_optimize": True,
                "ptxas_options": ["-v", "-O0"]
            }
        else:
            comp_kwargs = {"ptxas_options": ["-O3"]}
        program_options = ProgramOptions(  # include_path maybe needed here?
            name="GPUStreamlines",
            arch=f"sm_{dev.arch}",
            use_fast_math=True,
            extra_device_vectorization=True,
            std="c++11",
            **comp_kwargs
        )
        prog = Program(code, code_type="c++", options=program_options)
        _program = prog.compile("cubin", name_expressions=("vector_add<float>",))


class _GPUDirectionGetter(ABC):
    @abstractmethod
    def get_direction(self):
        pass

    @abstractmethod
    def get_num_streamlines(self):
        pass

    @abstractmethod
    def allocate_on_gpu(self):
        pass

    @abstractmethod
    def deallocate_on_gpu(self):
        pass


class BootDirectionGetter(_GPUDirectionGetter):
    def __init__(  # TODO: Maybe accept a dipy thing and extract arrays here? maybe as a from_ function?
            self,
            min_signal: float,
            H: np.ndarray,
            R: np.ndarray,
            delta_b: np.ndarray,
            delta_q: np.ndarray,
            sampling_matrix: np.ndarray,
            b0s_mask: np.ndarray):
        for name, arr, dt in [
                ("H", H, REAL_DTYPE),
                ("R", R, REAL_DTYPE),
                ("delta_b", delta_b, REAL_DTYPE),
                ("delta_q", delta_q, REAL_DTYPE),
                ("b0s_mask", b0s_mask, np.int32),
                ("sampling_matrix", sampling_matrix, REAL_DTYPE)]:
            if arr.dtype != dt:
                raise TypeError(f"{name} must have dtype {dt}, got {arr.dtype}")
            if not arr.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")

        self.H = H
        self.R = R
        self.delta_b = delta_b
        self.delta_q = delta_q
        self.delta_nr = int(delta_b.shape[0])
        self.min_signal = REAL_DTYPE(min_signal)
        self.sampling_matrix = sampling_matrix

        self.H_d = []
        self.R_d = []
        self.delta_b_d = []
        self.delta_q_d = []
        self.b0s_mask_d = []
        self.sampling_matrix_d = []

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

        checkCudaErrors(runtime.cudaMemcpy(
            self.H_d[n],
            self.H.ctypes.data,
            REAL_SIZE*self.H.size,
            runtime.cudaMemcpyHostToDevice))
        checkCudaErrors(runtime.cudaMemcpy(
            self.R_d[n],
            self.R.ctypes.data,
            REAL_SIZE*self.R.size,
            runtime.cudaMemcpyHostToDevice))
        checkCudaErrors(runtime.cudaMemcpy(
            self.delta_b_d[n],
            self.delta_b.ctypes.data,
            REAL_SIZE*self.delta_b.size,
            runtime.cudaMemcpyHostToDevice))
        checkCudaErrors(runtime.cudaMemcpy(
            self.delta_q_d[n],
            self.delta_q.ctypes.data,
            REAL_SIZE*self.delta_q.size,
            runtime.cudaMemcpyHostToDevice))
        checkCudaErrors(runtime.cudaMemcpy(
            self.b0s_mask_d[n],
            self.b0s_mask.ctypes.data,
            np.int32().nbytes*self.b0s_mask.size,
            runtime.cudaMemcpyHostToDevice))
        checkCudaErrors(runtime.cudaMemcpy(
            self.sampling_matrix_d[n],
            self.sampling_matrix.ctypes.data,
            REAL_SIZE*self.sampling_matrix.size,
            runtime.cudaMemcpyHostToDevice))

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

    def getNumStreamlines(self):
       pass

    def generateStreamlines(self):
       pass




// Precompute number of streamlines before allocating memory
if (!((model_type == PTT) || (model_type == PROB))) {
    shSizeGNS = sizeof(REAL)*(THR_X_BL/THR_X_SL)*(2*n32dimt + 2*MAX(n32dimt, samplm_nr)) + // for get_direction_boot_d
                sizeof(int)*samplm_nr;						      // for peak_directions_d	
    getNumStreamlinesBoot_k<THR_X_SL,
                            THR_X_BL/THR_X_SL>
                            <<<grid, block, shSizeGNS>>>(
                                    model_type,
                                    max_angle,
                                    min_signal,
                                    relative_peak_thresh,
                                    min_separation_angle,
                                    rng_seed,
                                    nseeds_gpu,
                                    reinterpret_cast<const REAL3 *>(seeds_d[n]),
                                    dimx,
                                    dimy,
                                    dimz,
                                    dimt,
                                    dataf_d[n],
                                    H_d[n],
                                    R_d[n],
                                    delta_nr,
                                    delta_b_d[n],
                                    delta_q_d[n],
                                    b0s_mask_d[n],
                                    samplm_nr,
                                    sampling_matrix_d[n],
                                    reinterpret_cast<const REAL3 *>(sphere_vertices_d[n]),
                                    reinterpret_cast<const int2 *>(sphere_edges_d[n]),
                                    nedges,
                                    shDirTemp0_d[n],
                                    slinesOffs_d[n]);
} else {
    shSizeGNS = sizeof(REAL)*(THR_X_BL/THR_X_SL)*n32dimt + sizeof(int)*(THR_X_BL/THR_X_SL)*n32dimt;
    getNumStreamlinesProb_k<THR_X_SL,
                            THR_X_BL/THR_X_SL>
                            <<<grid, block, shSizeGNS>>>(
                                    max_angle,
                                    relative_peak_thresh,
                                    min_separation_angle,
                                    rng_seed,
                                    nseeds_gpu,
                                    reinterpret_cast<const REAL3 *>(seeds_d[n]),
                                    dimx,
                                    dimy,
                                    dimz,
                                    dimt,
                                    dataf_d[n],
                                    reinterpret_cast<const REAL3 *>(sphere_vertices_d[n]),
                                    reinterpret_cast<const int2 *>(sphere_edges_d[n]),
                                    nedges,
                                    shDirTemp0_d[n],
                                    slinesOffs_d[n]);
}
    

  //#pragma omp parallel for
  for (int n = 0; n < ngpus; ++n) {
    CHECK_CUDA(cudaSetDevice(n));
    int nseeds_gpu = std::min(nseeds_per_gpu, std::max(0, nseeds - n*nseeds_per_gpu));
    if (nseeds_gpu == 0) continue;
    dim3 block(THR_X_SL, THR_X_BL/THR_X_SL);
    dim3 grid(DIV_UP(nseeds_gpu, THR_X_BL/THR_X_SL));
#if 0
    std::cerr << "GPU " << n << ": ";
    std::cerr << "Generating " << nSlines_h[n] << " streamlines (from " << nseeds_gpu << " seeds)" << std::endl; 
#endif

    //fprintf(stderr, "Launching kernel with %u blocks of size (%u, %u)\n", grid.x, block.x, block.y);
    switch(model_type) {
        case OPDT:
            genStreamlinesMerge_k<THR_X_SL, THR_X_BL/THR_X_SL, OPDT> <<<grid, block, shSizeGNS, streams[n]>>>(
                max_angle, min_signal, tc_threshold, step_size, relative_peak_thresh, min_separation_angle,
                rng_seed, rng_offset + n*nseeds_per_gpu, nseeds_gpu, reinterpret_cast<const REAL3 *>(seeds_d[n]),
                dimx, dimy, dimz, dimt, dataf_d[n], H_d[n], R_d[n], delta_nr, delta_b_d[n], delta_q_d[n],
                b0s_mask_d[n], metric_map_d[n], samplm_nr, sampling_matrix_d[n],
                reinterpret_cast<const REAL3 *>(sphere_vertices_d[n]), reinterpret_cast<const int2 *>(sphere_edges_d[n]),
                nedges, slinesOffs_d[n], shDirTemp0_d[n], slineSeed_d[n], slineLen_d[n], sline_d[n]);
            break;

        case CSA:
            genStreamlinesMerge_k<THR_X_SL, THR_X_BL/THR_X_SL, CSA> <<<grid, block, shSizeGNS, streams[n]>>>(
                max_angle, min_signal, tc_threshold, step_size, relative_peak_thresh, min_separation_angle,
                rng_seed, rng_offset + n*nseeds_per_gpu, nseeds_gpu, reinterpret_cast<const REAL3 *>(seeds_d[n]),
                dimx, dimy, dimz, dimt, dataf_d[n], H_d[n], R_d[n], delta_nr, delta_b_d[n], delta_q_d[n],
                b0s_mask_d[n], metric_map_d[n], samplm_nr, sampling_matrix_d[n],
                reinterpret_cast<const REAL3 *>(sphere_vertices_d[n]), reinterpret_cast<const int2 *>(sphere_edges_d[n]),
                nedges, slinesOffs_d[n], shDirTemp0_d[n], slineSeed_d[n], slineLen_d[n], sline_d[n]);
            break;

        case PROB:
            // Shared memory requirements are smaller for probabilistic for main run
            // than for preliminary run
            shSizeGNS = sizeof(REAL)*(THR_X_BL/THR_X_SL)*n32dimt;
            genStreamlinesMerge_k<THR_X_SL, THR_X_BL/THR_X_SL, PROB> <<<grid, block, shSizeGNS, streams[n]>>>(
                max_angle, min_signal, tc_threshold, step_size, relative_peak_thresh, min_separation_angle,
                rng_seed, rng_offset + n*nseeds_per_gpu, nseeds_gpu, reinterpret_cast<const REAL3 *>(seeds_d[n]),
                dimx, dimy, dimz, dimt, dataf_d[n], H_d[n], R_d[n], delta_nr, delta_b_d[n], delta_q_d[n],
                b0s_mask_d[n], metric_map_d[n], samplm_nr, sampling_matrix_d[n],
                reinterpret_cast<const REAL3 *>(sphere_vertices_d[n]), reinterpret_cast<const int2 *>(sphere_edges_d[n]),
                nedges, slinesOffs_d[n], shDirTemp0_d[n], slineSeed_d[n], slineLen_d[n], sline_d[n]);
            break;

        case PTT:
            shSizeGNS = 0; // PTT uses exclusively static shared memory
            genStreamlinesMerge_k<THR_X_SL, THR_X_BL/THR_X_SL, PTT> <<<grid, block, shSizeGNS, streams[n]>>>(
                max_angle, min_signal, tc_threshold, step_size, relative_peak_thresh, min_separation_angle,
                rng_seed, rng_offset + n*nseeds_per_gpu, nseeds_gpu, reinterpret_cast<const REAL3 *>(seeds_d[n]),
                dimx, dimy, dimz, dimt, dataf_d[n], H_d[n], R_d[n], delta_nr, delta_b_d[n], delta_q_d[n],
                b0s_mask_d[n], metric_map_d[n], samplm_nr, sampling_matrix_d[n],
                reinterpret_cast<const REAL3 *>(sphere_vertices_d[n]), reinterpret_cast<const int2 *>(sphere_edges_d[n]),
                nedges, slinesOffs_d[n], shDirTemp0_d[n], slineSeed_d[n], slineLen_d[n], sline_d[n]);
            break;

        default:
            printf("FATAL: Invalid Model Type.\n");
            break;
    }

    CHECK_ERROR("genStreamlinesMerge_k");
  }


