from cuda.bindings import driver, nvrtc, runtime
# TODO: this would be better if only using CUDA core

import numpy as np
import logging

import re
import os


logger = logging.getLogger("GPUStreamlines")


# We extract REAL_DTYPE, MAX_SLINE_LEN from globals.h
# Maybe there is a more elegant way of doing this?
dir_path = os.path.dirname(os.path.abspath(__file__))
globals_path = os.path.join(dir_path, "globals.h")
with open(globals_path, 'r') as f:
    content = f.read()

defines = dict(re.findall(r"#define\s+(\w+)\s+([^\s/]+)", content))
REAL_SIZE = int(defines["REAL_SIZE"])
if REAL_SIZE == 4:
    REAL_DTYPE = np.float32
elif REAL_SIZE == 8:
    REAL_DTYPE = np.float64
else:
    raise NotImplementedError(f"Unsupported REAL_SIZE={REAL_SIZE} in globals.h")
MAX_SLINE_LEN = int(defines["MAX_SLINE_LEN"])


def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


class GPUTracker:
    def __init__(
        self,
        model_type: ModelType,
        max_angle: float,
        min_signal: float,
        tc_threshold: float,
        step_size: float,
        relative_peak_thresh: float,
        min_separation_angle: float,
        dataf: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        delta_b: np.ndarray,
        delta_q: np.ndarray, # TODO: some of these only needed for boot
        b0s_mask: np.ndarray,
        metric_map: np.ndarray,
        sampling_matrix: np.ndarray,
        sphere_vertices: np.ndarray,
        sphere_edges: np.ndarray,
        ngpus: int = 1,
        rng_seed: int = 0,
        rng_offset: int = 0,
    ):
        for name, arr, dt in [
            ("dataf", dataf, REAL_DTYPE),
            ("H", H, REAL_DTYPE),
            ("R", R, REAL_DTYPE),
            ("delta_b", delta_b, REAL_DTYPE),
            ("delta_q", delta_q, REAL_DTYPE),
            ("b0s_mask", b0s_mask, np.int32),
            ("metric_map", metric_map, REAL_DTYPE),
            ("sampling_matrix", sampling_matrix, REAL_DTYPE),
            ("sphere_vertices", sphere_vertices, REAL_DTYPE),
            ("sphere_edges", sphere_edges, np.int32),
        ]:
            if arr.dtype != dt:
                raise TypeError(f"{name} must have dtype {dt}, got {arr.dtype}")
            if not arr.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")

        self.dataf = dataf
        self.H = H
        self.R = R
        self.delta_b = delta_b
        self.delta_q = delta_q
        self.b0s_mask = b0s_mask
        self.metric_map = metric_map
        self.sampling_matrix = sampling_matrix
        self.sphere_vertices = sphere_vertices
        self.sphere_edges = sphere_edges

        self.dimx, self.dimy, self.dimz, self.dimt = dataf.shape
        self.nedges = int(sphere_edges.shape[0])
        self.delta_nr = int(delta_b.shape[0])
        self.samplm_nr = int(sampling_matrix.shape[0])

        self.model_type = int(model_type)
        self.max_angle = REAL_DTYPE(max_angle)
        self.min_signal = REAL_DTYPE(min_signal)
        self.tc_threshold = REAL_DTYPE(tc_threshold)
        self.step_size = REAL_DTYPE(step_size)
        self.relative_peak_thresh = REAL_DTYPE(relative_peak_thresh)
        self.min_separation_angle = REAL_DTYPE(min_separation_angle)

        self.ngpus = int(ngpus)
        self.rng_seed = int(rng_seed)
        self.rng_offset = int(rng_offset)

        self.nSlines_old = []
        self.slines = []
        self.sline_lens = []

        checkCudaErrors(driver.cuInit(0))
        avail = checkCudaErrors(runtime.cudaGetDeviceCount())
        if self.ngpus > avail:
            raise RuntimeError(f"Requested {self.ngpus} GPUs but only {avail} available")

        logger.info("Creating GPUTracker with %d GPUs...", self.ngpus)

        self.dataf_pts = []
        self.H_pts = []
        self.R_pts = []
        self.delta_b_pts = []
        self.delta_q_pts = []
        self.b0s_mask_pts = []
        self.metric_map_pts = []
        self.sampling_matrix_pts = []
        self.sphere_vertices_pts = []
        self.sphere_edges_pts = []

        for ii in range(self.ngpus):
            checkCudaErrors(runtime.cudaSetDevice(ii))
            self.dataf_pts.append( # TODO: put this in texture memory?
                checkCudaErrors(runtime.cudaMallocManaged(
                    REAL_SIZE*self.dataf.size, 
                    runtime.cudaMemAttachGlobal)))
            checkCudaErrors(runtime.cudaMemAdvise(
                self.dataf_pts[ii],
                REAL_SIZE*self.dataf.size,
                runtime.cudaMemAdviseSetPreferredLocation,
                ii))
            self.H_pts.append(
                checkCudaErrors(runtime.cudaMalloc(
                    REAL_SIZE*self.H.size)))
            self.R_pts.append(
                checkCudaErrors(runtime.cudaMalloc(
                    REAL_SIZE*self.R.size)))
            self.delta_b_pts.append(
                checkCudaErrors(runtime.cudaMalloc(
                    REAL_SIZE*self.delta_b.size)))
            self.delta_q_pts.append(
                checkCudaErrors(runtime.cudaMalloc(
                    REAL_SIZE*self.delta_q.size)))
            self.b0s_mask_pts.append(
                checkCudaErrors(runtime.cudaMalloc(
                    np.int32().nbytes*self.b0s_mask.size)))
            self.metric_map_pts.append(
                checkCudaErrors(runtime.cudaMalloc(
                    REAL_SIZE*self.metric_map.size)))
            self.sampling_matrix_pts.append(
                checkCudaErrors(runtime.cudaMalloc(
                    REAL_SIZE*self.sampling_matrix.size)))
            self.sphere_vertices_pts.append(
                checkCudaErrors(runtime.cudaMalloc(
                    REAL_SIZE*self.sphere_vertices.size)))
            self.sphere_edges_pts.append(
                checkCudaErrors(runtime.cudaMalloc(
                    np.int32().nbytes*self.sphere_edges.size)))
            
            checkCudaErrors(runtime.cudaMemcpy(
                self.dataf_pts[ii],
                self.dataf.ctypes.data,
                REAL_SIZE*self.dataf.size,
                runtime.cudaMemcpyHostToDevice))
            checkCudaErrors(runtime.cudaMemcpy(
                self.H_pts[ii],
                self.H.ctypes.data,
                REAL_SIZE*self.H.size,
                runtime.cudaMemcpyHostToDevice))
            checkCudaErrors(runtime.cudaMemcpy(
                self.R_pts[ii],
                self.R.ctypes.data,
                REAL_SIZE*self.R.size,
                runtime.cudaMemcpyHostToDevice))
            checkCudaErrors(runtime.cudaMemcpy(
                self.delta_b_pts[ii],
                self.delta_b.ctypes.data,
                REAL_SIZE*self.delta_b.size,
                runtime.cudaMemcpyHostToDevice))
            checkCudaErrors(runtime.cudaMemcpy(
                self.delta_q_pts[ii],
                self.delta_q.ctypes.data,
                REAL_SIZE*self.delta_q.size,
                runtime.cudaMemcpyHostToDevice))
            checkCudaErrors(runtime.cudaMemcpy(
                self.b0s_mask_pts[ii],
                self.b0s_mask.ctypes.data,
                np.int32().nbytes*self.b0s_mask.size,
                runtime.cudaMemcpyHostToDevice))
            checkCudaErrors(runtime.cudaMemcpy(
                self.metric_map_pts[ii],
                self.metric_map.ctypes.data,
                REAL_SIZE*self.metric_map.size,
                runtime.cudaMemcpyHostToDevice))
            checkCudaErrors(runtime.cudaMemcpy(
                self.sampling_matrix_pts[ii],
                self.sampling_matrix.ctypes.data,
                REAL_SIZE*self.sampling_matrix.size,
                runtime.cudaMemcpyHostToDevice))
            checkCudaErrors(runtime.cudaMemcpy(
                self.sphere_vertices_pts[ii],
                self.sphere_vertices.ctypes.data,
                REAL_SIZE*self.sphere_vertices.size,
                runtime.cudaMemcpyHostToDevice))
            checkCudaErrors(runtime.cudaMemcpy(
                self.sphere_edges_pts[ii],
                self.sphere_edges.ctypes.data,
                np.int32().nbytes*self.sphere_edges.size,
                runtime.cudaMemcpyHostToDevice))

        self.streams = []
        for ii in range(self.ngpus):
            checkCudaErrors(runtime.cudaSetDevice(ii))
            self.streams.append(
                checkCudaErrors(runtime.cudaStreamCreateWithFlags(
                    runtime.cudaStreamNonBlocking)))

    def generate_streamlines(self, seeds):  # TODO: location this is going should be these arguments
        nseeds = len(seeds)
        nseeds_per_gpu = (nseeds + self.ngpus - 1) // self.ngpus

        seeds_ptrs = []

        for ii in range(self.ngpus):
            nseeds_gpu = min(nseeds_per_gpu, max(0, nseeds - ii * nseeds_per_gpu))
            checkCudaErrors(runtime.cudaSetDevice(ii))
            seeds_ptrs.append(checkCudaErrors(runtime.cudaMalloc(
                REAL_SIZE*3*nseeds_gpu)))
            checkCudaErrors(runtime.cudaMemcpy(
                seeds_ptrs[ii],
                seeds[ii*nseeds_per_gpu:(ii+1)*nseeds_per_gpu].ctypes.data,
                REAL_SIZE*3*nseeds_gpu,
                runtime.cudaMemcpyHostToDevice))
        
        nSlines = [0] * self.ngpus  # TODO: figure out what this is doing
        # TODO:
    #   // Call GPU routine
    #   generate_streamlines_cuda_mgpu(model_type_, max_angle_, min_signal_, tc_threshold_, step_size_,
    #                                  relative_peak_thresh_, min_separation_angle_,
    #                                  nseeds, seeds_d,
    #                                  dimx_, dimy_, dimz_, dimt_,
    #                                  dataf_d, H_d, R_d, delta_nr_, delta_b_d, delta_q_d, b0s_mask_d, metric_map_d, samplm_nr_, sampling_matrix_d,
    #                                  sphere_vertices_d, sphere_edges_d, nedges_,
    #                                  slines_, slinesLen_, nSlines, nSlines_old_, rng_seed_, rng_offset_, ngpus_,
    #                                  streams_);

        self.nSlines_old = nSlines.copy()  # TODO: figure out what this is doing
        self.rng_offset += nseeds

        nSlines_total = 0
        for ii in range(self.ngpus):
            checkCudaErrors(runtime.cudaFree(seeds_ptrs[ii]))
            nSlines_total += nSlines[ii]


        # TODO
    #   std::vector<py::array_t<REAL>> slines_list;
    #   slines_list.reserve(nSlines_total);
    #   for (int n = 0; n < ngpus_; ++n) {
    #     for (int i = 0; i < nSlines[n]; ++i) {
    #       REAL* sl = new REAL[slinesLen_[n][i]*3];
    #       std::memcpy(sl, slines_[n] + i*3*2*MAX_SLINE_LEN, slinesLen_[n][i]*3*sizeof(*sl));
    #       auto sl_arr = py::array_t<REAL>({slinesLen_[n][i], 3}, // shape
    #                                       {3*sizeof(REAL), sizeof(REAL)}, // strides
    #                                       sl,
    #                                       cleanup(sl));
    #       slines_list.push_back(sl_arr);
    #     }
    #   }

    #   return slines_list;

    # }