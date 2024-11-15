cmake_minimum_required(VERSION 3.24)

project(cuslines LANGUAGES CUDA CXX VERSION 1.0)

# Build settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Werror=reorder")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug" )
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
else()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# CUDA
find_package(CUDAToolkit REQUIRED)

# Set default CUDA compute capabilities if unset
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  include(FindCUDA/select_compute_arch.cmake)
  cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Auto)
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_FLAGS})
endif()
message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    
    # Set OMP runtime based on compiler
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(OMP_RUNTIME "INTEL")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      set(OMP_RUNTIME "GNU")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
      set(OMP_RUNTIME "INTEL")
    endif()
    message(STATUS "OpenMP runtime used: ${OMP_RUNTIME}")
endif()

# Build library and pybind11 module
add_subdirectory(external/pybind11)
add_library(cuslines_kernels)
target_sources(cuslines_kernels
	       PRIVATE
	       ${CMAKE_SOURCE_DIR}/cuslines/generate_streamlines_cuda.cu)
set_target_properties(cuslines_kernels PROPERTIES OUTPUT_NAME cuslines_kernels
                                                  POSITION_INDEPENDENT_CODE TRUE)

pybind11_add_module(cuslines ${CMAKE_SOURCE_DIR}/cuslines/cuslines.cpp)
target_include_directories(cuslines PUBLIC "${CMAKE_SOURCE_DIR}/cuslines" "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(cuslines PRIVATE cuslines_kernels CUDA::cudart_static)

# Install
install(TARGETS cuslines cuslines_kernels LIBRARY DESTINATION .)
