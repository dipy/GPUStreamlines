cmake_minimum_required(VERSION 3.10)

#project definition
cmake_policy(SET CMP0104 NEW)
project(cuslines LANGUAGES CUDA CXX VERSION 1.0)

#include the external project stuff
include(ExternalProject)

#global settings
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Werror=reorder")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#determine type of build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

#now, add the right flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug" )
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
else()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

#decide whether to use CUDA or not
#find_package(CUDAToolkit REQUIRED)
if(NOT CUDA_COMPUTE_CAPABILITY)
  set(CUDA_COMPUTE_CAPABILITY 70)
endif()

#Find OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    
    #check which compiler we have to determine openmp runtime
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(OMP_RUNTIME "INTEL")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      set(OMP_RUNTIME "GNU")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
      set(OMP_RUNTIME "INTEL")
    endif()
    message(STATUS "OpenMP runtime used: ${OMP_RUNTIME}")
endif()

# add object lib
add_library(cuslines_kernels-object OBJECT
  "${CMAKE_SOURCE_DIR}/cuslines/generate_streamlines_cuda.cu"
)
target_include_directories( cuslines_kernels-object PUBLIC "${CMAKE_SOURCE_DIR}/cuslines" )
target_link_libraries(cuslines_kernels-object PUBLIC cudart)
set_target_properties(cuslines_kernels-object PROPERTIES POSITION_INDEPENDENT_CODE TRUE
                                                         CUDA_ARCHITECTURES "${CUDA_COMPUTE_CAPABILITY}")

##############################################
# PyBind11/Python build
##############################################
add_subdirectory(external/pybind11)

FIND_PACKAGE( PythonLibs )
INCLUDE_DIRECTORIES( ${PYTHON_INCLUDE_PATH} )

add_library(cuslines_kernels-shared STATIC $<TARGET_OBJECTS:cuslines_kernels-object>)
set_target_properties(cuslines_kernels-shared PROPERTIES OUTPUT_NAME cuslines_kernels 
                                                         POSITION_INDEPENDENT_CODE TRUE
                                                         CUDA_ARCHITECTURES "${CUDA_COMPUTE_CAPABILITY}")

pybind11_add_module(cuslines ${CMAKE_SOURCE_DIR}/cuslines/cuslines.cpp)
target_include_directories(cuslines PUBLIC "${CMAKE_SOURCE_DIR}/cuslines" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_link_libraries(cuslines PRIVATE cuslines_kernels-shared cudart)
set_target_properties(cuslines PROPERTIES CUDA_ARCHITECTURES "${CUDA_COMPUTE_CAPABILITY}")

# custom target for install
#add_custom_target(mvcmd_py ALL
#  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/cuslines/__init__.py" "${CMAKE_INSTALL_PREFIX}/cuslines/__init__.py"
#  DEPENDS "${CMAKE_SOURCE_DIR}/cuslines/__init__.py"
#)

# install
install(TARGETS cuslines cuslines_kernels-shared LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/cuslines")
install(FILES "${CMAKE_SOURCE_DIR}/cuslines/__init__.py" DESTINATION "${CMAKE_INSTALL_PREFIX}/cuslines")
install(FILES "${CMAKE_SOURCE_DIR}/run_dipy_gpu.py" DESTINATION "${CMAKE_INSTALL_PREFIX}/")
install(FILES "${CMAKE_SOURCE_DIR}/run_dipy_cpu.py" DESTINATION "${CMAKE_INSTALL_PREFIX}/")
install(FILES "${CMAKE_SOURCE_DIR}/run_dipy_gpu_hardi.py" DESTINATION "${CMAKE_INSTALL_PREFIX}/")
install(FILES "${CMAKE_SOURCE_DIR}/run_dipy_cpu_hardi.py" DESTINATION "${CMAKE_INSTALL_PREFIX}/")
