/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstring>
#include <iostream>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <cuda_runtime.h>

// #define USE_NVTX

#include "globals.h"
#include "cudamacro.h"
#include "generate_streamlines_cuda.h"

using np_array = py::array_t<REAL>;
using np_array_int = py::array_t<int>;

using np_array_cast = py::array_t<REAL, py::array::c_style | py::array::forcecast>;
using np_array_int_cast = py::array_t<int, py::array::c_style | py::array::forcecast>;

// Handle to cleanup returned host allocations when associated Python object is destroyed
template <typename T>
py::capsule cleanup(T* ptr) {
  return py::capsule(ptr, [](void *f) {
           T *g = reinterpret_cast<T *>(f);
           delete [] g;
         });
}

class GPUTracker {
  public:
    GPUTracker(ModelType model_type,
               double max_angle,
               double min_signal,
               double tc_threshold,
               double step_size,
               double relative_peak_thresh,
               double min_separation_angle,
               np_array_cast dataf,
               np_array_cast H,
               np_array_cast R,
               np_array_cast delta_b,
               np_array_cast delta_q,
               np_array_int_cast b0s_mask,
               np_array_cast metric_map,
               np_array_cast sampling_matrix,
               np_array_cast sphere_vertices,
               np_array_int_cast sphere_edges,
               int ngpus = 1,
               int rng_seed = 0,
               int rng_offset = 0) {

      // Get info structs from numpy objects
      auto dataf_info = dataf.request();
      auto H_info = H.request();
      auto R_info = R.request();
      auto delta_b_info = delta_b.request();
      auto delta_q_info = delta_q.request();
      auto b0s_mask_info = b0s_mask.request();
      auto metric_map_info = metric_map.request();
      auto sampling_matrix_info = sampling_matrix.request();
      auto sphere_vertices_info = sphere_vertices.request();
      auto sphere_edges_info = sphere_edges.request();

      dimx_ = dataf_info.shape[0];
      dimy_ = dataf_info.shape[1];
      dimz_ = dataf_info.shape[2];
      dimt_ = dataf_info.shape[3];
      nedges_ = sphere_edges_info.shape[0];

      delta_nr_ = delta_b_info.shape[0];
      samplm_nr_ = sampling_matrix_info.shape[0];

// No longer needed
#if 0
      // Error checking for template parameters.
      // TODO: Need to make kernel more general.
      if (delta_b_info.shape[0] != 28 ||
          sampling_matrix_info.shape[0] != 181 ||
          dataf_info.shape[3] > 160) {
          std::cout << delta_b_info.shape[0] << " " << sampling_matrix_info.shape[0] << " " << dataf_info.shape[3] << std::endl;
          throw std::invalid_argument("Input data dimensions not currently supported.");
      }
#endif

      // Get number of GPUs
      int ngpus_avail;
      CHECK_CUDA(cudaGetDeviceCount(&ngpus_avail));
      if (ngpus > ngpus_avail) {
        throw std::runtime_error("Requested to use more GPUs than available on system.");
      }

      std::cerr << "Creating GPUTracker with " << ngpus << " GPUs..." << std::endl;
      ngpus_ = ngpus;

      model_type_ = model_type;
      max_angle_ = max_angle;
      min_signal_ = min_signal;
      tc_threshold_ = tc_threshold;
      step_size_ = step_size;
      relative_peak_thresh_ = relative_peak_thresh,
      min_separation_angle_ = min_separation_angle,

      // Allocate/copy constant problem data on GPUs
      dataf_d.resize(ngpus_, nullptr);
      H_d.resize(ngpus_, nullptr);
      R_d.resize(ngpus_, nullptr);
      delta_b_d.resize(ngpus_, nullptr);
      delta_q_d.resize(ngpus_, nullptr);
      b0s_mask_d.resize(ngpus_, nullptr);
      metric_map_d.resize(ngpus_, nullptr);
      sampling_matrix_d.resize(ngpus_, nullptr);
      sphere_vertices_d.resize(ngpus_, nullptr);
      sphere_edges_d.resize(ngpus_, nullptr);

      //#pragma omp parallel for
      for (int n = 0; n < ngpus_; ++n) {
        CHECK_CUDA(cudaSetDevice(n));
        CHECK_CUDA(cudaMallocManaged(&dataf_d[n], sizeof(*dataf_d[n]) * dataf_info.size));
        CUDA_MEM_ADVISE(dataf_d[n], sizeof(*dataf_d[n]) * dataf_info.size, cudaMemAdviseSetPreferredLocation, n);
        CHECK_CUDA(cudaMalloc(&H_d[n], sizeof(*H_d[n]) * H_info.size));
        CHECK_CUDA(cudaMalloc(&R_d[n], sizeof(*R_d[n]) * R_info.size));
        CHECK_CUDA(cudaMalloc(&delta_b_d[n], sizeof(*delta_b_d[n]) * delta_b_info.size));
        CHECK_CUDA(cudaMalloc(&delta_q_d[n], sizeof(*delta_q_d[n]) * delta_q_info.size));
        CHECK_CUDA(cudaMalloc(&b0s_mask_d[n], sizeof(*b0s_mask_d[n]) * b0s_mask_info.size));
        CHECK_CUDA(cudaMalloc(&metric_map_d[n], sizeof(*metric_map_d[n]) * metric_map_info.size));
        CHECK_CUDA(cudaMalloc(&sampling_matrix_d[n], sizeof(*sampling_matrix_d[n]) * sampling_matrix_info.size));
        CHECK_CUDA(cudaMalloc(&sphere_vertices_d[n], sizeof(*sphere_vertices_d[n]) * sphere_vertices_info.size));
        CHECK_CUDA(cudaMalloc(&sphere_edges_d[n], sizeof(*sphere_edges_d[n]) * sphere_edges_info.size));

        CHECK_CUDA(cudaMemcpy(dataf_d[n], dataf_info.ptr, sizeof(*dataf_d[n]) * dataf_info.size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(H_d[n], H_info.ptr, sizeof(*H_d[n]) * H_info.size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(R_d[n], R_info.ptr, sizeof(*R_d[n]) * R_info.size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(delta_b_d[n], delta_b_info.ptr, sizeof(*delta_b_d[n]) * delta_b_info.size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(delta_q_d[n], delta_q_info.ptr, sizeof(*delta_q_d[n]) * delta_q_info.size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(b0s_mask_d[n], b0s_mask_info.ptr, sizeof(*b0s_mask_d[n]) * b0s_mask_info.size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(metric_map_d[n], metric_map_info.ptr, sizeof(*metric_map_d[n]) * metric_map_info.size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(sampling_matrix_d[n], sampling_matrix_info.ptr, sizeof(*sampling_matrix_d[n]) * sampling_matrix_info.size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(sphere_vertices_d[n], sphere_vertices_info.ptr, sizeof(*sphere_vertices_d[n]) * sphere_vertices_info.size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(sphere_edges_d[n], sphere_edges_info.ptr, sizeof(*sphere_edges_d[n]) * sphere_edges_info.size, cudaMemcpyHostToDevice));
      }

      rng_seed_ = rng_seed;
      rng_offset_ = rng_offset;
      nSlines_old_.resize(ngpus_, 0);
      slines_.resize(ngpus_, nullptr);
      slinesLen_.resize(ngpus_, nullptr);

      streams_.resize(ngpus_);
      for (int n = 0; n < ngpus_; ++n) {
        CHECK_CUDA(cudaSetDevice(n));
        CHECK_CUDA(cudaStreamCreateWithFlags(&streams_[n], cudaStreamNonBlocking));
      }

    }

    ~GPUTracker() {
      std::cerr << "Destroy GPUTracker..." << std::endl;
      for (int n = 0; n < ngpus_; ++n) {
        CHECK_CUDA(cudaSetDevice(n));
        if (dataf_d[n]) CHECK_CUDA(cudaFree(dataf_d[n]));
        if (H_d[n]) CHECK_CUDA(cudaFree(H_d[n]));
        if (R_d[n]) CHECK_CUDA(cudaFree(R_d[n]));
        if (delta_b_d[n]) CHECK_CUDA(cudaFree(delta_b_d[n]));
        if (delta_q_d[n]) CHECK_CUDA(cudaFree(delta_q_d[n]));
        if (b0s_mask_d[n]) CHECK_CUDA(cudaFree(b0s_mask_d[n]));
        if (metric_map_d[n]) CHECK_CUDA(cudaFree(metric_map_d[n]));
        if (sampling_matrix_d[n]) CHECK_CUDA(cudaFree(sampling_matrix_d[n]));
        if (sphere_vertices_d[n]) CHECK_CUDA(cudaFree(sphere_vertices_d[n]));
        if (sphere_edges_d[n]) CHECK_CUDA(cudaFree(sphere_edges_d[n]));

        if (slines_[n]) CHECK_CUDA(cudaFreeHost(slines_[n]));
        if (slinesLen_[n]) CHECK_CUDA(cudaFreeHost(slinesLen_[n]));

        CHECK_CUDA(cudaStreamDestroy(streams_[n]));
      }
    }

    std::vector<py::array_t<REAL>> generate_streamlines(np_array seeds) {

      auto seeds_info = seeds.request();
      int nseeds = seeds_info.shape[0];

      std::vector<REAL*> seeds_d(ngpus_, nullptr);
      int nseeds_per_gpu = (nseeds + ngpus_ - 1) / ngpus_;

      //#pragma omp parallel for
      for (int n = 0; n < ngpus_; ++n) {
        int nseeds_gpu = std::min(nseeds_per_gpu, std::max(0, nseeds - n*nseeds_per_gpu));
        CHECK_CUDA(cudaSetDevice(n));
        CHECK_CUDA(cudaMalloc(&seeds_d[n], sizeof(*seeds_d[n]) * 3 * nseeds_gpu));
        CHECK_CUDA(cudaMemcpy(seeds_d[n], reinterpret_cast<REAL *>(seeds_info.ptr) + 3*n*nseeds_per_gpu, sizeof(*seeds_d[n]) * 3 * nseeds_gpu, cudaMemcpyHostToDevice));
      }

      std::vector<int> nSlines(ngpus_);

      // Call GPU routine
      generate_streamlines_cuda_mgpu(model_type_, max_angle_, min_signal_, tc_threshold_, step_size_,
                                     relative_peak_thresh_, min_separation_angle_,
                                     nseeds, seeds_d,
                                     dimx_, dimy_, dimz_, dimt_,
                                     dataf_d, H_d, R_d, delta_nr_, delta_b_d, delta_q_d, b0s_mask_d, metric_map_d, samplm_nr_, sampling_matrix_d,
                                     sphere_vertices_d, sphere_edges_d, nedges_,
                                     slines_, slinesLen_, nSlines, nSlines_old_, rng_seed_, rng_offset_, ngpus_,
                                     streams_);

      nSlines_old_ = nSlines;  //store number of slines for next set of seeds

      // Update rng_offset for next set of seeds
      rng_offset_ += nseeds;

      int nSlines_total = 0;
      for (int n = 0; n < ngpus_; ++n) {
        CHECK_CUDA(cudaFree(seeds_d[n]));
        nSlines_total += nSlines[n];
      }

      std::vector<py::array_t<REAL>> slines_list;
      slines_list.reserve(nSlines_total);
      for (int n = 0; n < ngpus_; ++n) {
        for (int i = 0; i < nSlines[n]; ++i) {
          REAL* sl = new REAL[slinesLen_[n][i]*3];
          std::memcpy(sl, slines_[n] + i*3*2*MAX_SLINE_LEN, slinesLen_[n][i]*3*sizeof(*sl));
          auto sl_arr = py::array_t<REAL>({slinesLen_[n][i], 3}, // shape
                                          {3*sizeof(REAL), sizeof(REAL)}, // strides
                                          sl,
                                          cleanup(sl));
          slines_list.push_back(sl_arr);
        }
      }

      return slines_list;

    }

    void dump_streamlines(std::string output_prefix, std::string voxel_order,
                          np_array_int roi_shape, np_array voxel_size, np_array vox_to_ras) {

      auto roi_shape_info = roi_shape.request();
      auto voxel_size_info = voxel_size.request();
      auto vox_to_ras_info = vox_to_ras.request();

      START_RANGE("filewrite", 0);

      //#pragma omp parallel for
      for (int n = 0; n < ngpus_; ++n) {
        std::stringstream ss;
        ss << output_prefix << "_" << std::to_string(n) <<  ".trk";
        write_trk(ss.str().c_str(), reinterpret_cast<int *>(roi_shape_info.ptr), reinterpret_cast<REAL *>(voxel_size_info.ptr),
                  voxel_order.c_str(), reinterpret_cast<REAL *>(vox_to_ras_info.ptr), nSlines_old_[n], slinesLen_[n],
                  reinterpret_cast<REAL3 *>(slines_[n]));
      }

      END_RANGE;
    }

  private:
    int ngpus_;
    int rng_seed_;
    int rng_offset_;
    int dimx_, dimy_, dimz_, dimt_;
    int nedges_;
    int delta_nr_, samplm_nr_;

    ModelType model_type_;
    double max_angle_;
    double tc_threshold_;
    double min_signal_;
    double step_size_;
    double relative_peak_thresh_;
    double min_separation_angle_;

    std::vector<int> nSlines_old_;
    std::vector<REAL*> slines_;
    std::vector<int*> slinesLen_;

    std::vector<REAL*> dataf_d;
    std::vector<REAL*> H_d;
    std::vector<REAL*> R_d;
    std::vector<REAL*> delta_b_d;
    std::vector<REAL*> delta_q_d;
    std::vector<int*> b0s_mask_d;
    std::vector<REAL*> metric_map_d;
    std::vector<REAL*> sampling_matrix_d;
    std::vector<REAL*> sphere_vertices_d;
    std::vector<int*> sphere_edges_d;

    std::vector<cudaStream_t> streams_;

};


PYBIND11_MODULE(cuslines, m) {
  m.attr("MAX_SLINE_LEN") = py::int_(MAX_SLINE_LEN);
  m.attr("REAL_SIZE") = py::int_(REAL_SIZE);

  py::enum_<ModelType>(m, "ModelType")
    .value("OPDT", OPDT)
    .value("CSA", CSA)
    .value("PROB", PROB)
    .value("PTT", PTT);

  py::class_<GPUTracker>(m, "GPUTracker")
    .def(py::init<ModelType, double, double, double, double,
                  double, double,
		              np_array_cast, np_array_cast,
                  np_array_cast, np_array_cast,
                  np_array_cast, np_array_int_cast,
                  np_array_cast, np_array_cast,
                  np_array_cast, np_array_int_cast,
                  int, int, int>(),
                  py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
                  py::arg().noconvert(), py::arg().noconvert(),
                  py::arg().noconvert(), py::arg().noconvert(),
                  py::arg().noconvert(), py::arg().noconvert(),
                  py::arg().noconvert(), py::arg().noconvert(),
                  py::arg().noconvert(), py::arg().noconvert(),
                  py::arg().noconvert(), py::arg().noconvert(),
                  py::arg("ngpus") = 1, py::arg("rng_seed") = 0,
                  py::arg("rng_offset") = 0)

    .def("generate_streamlines", &GPUTracker::generate_streamlines,
         "Generates streamline for dipy test case.")

    .def("dump_streamlines", &GPUTracker::dump_streamlines,
         "Dump streamlines to file.");
}

