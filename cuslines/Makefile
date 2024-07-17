# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2
CUDA_HOME=/usr/local/cuda
CUDACC=$(CUDA_HOME)/bin/nvcc
CXX=g++
LD=g++

CXXFLAGS= -c -O3 -std=c++11 -fopenmp -fPIC `python3 -m pybind11 --includes` -I$(CUDA_HOME)/include

SMS ?= 60 70
CUDA_ARCH = $(foreach SM,$(SMS),-gencode arch=compute_$(SM),code=sm_$(SM))
LASTSM := $(lastword $(sort $(SMS)))
CUDA_ARCH += -gencode arch=compute_$(LASTSM),code=compute_$(LASTSM)
CUDACFLAGS=-c -O3 -lineinfo -Xptxas=-v -std=c++11 -Xcompiler -fPIC -Xcompiler=-fopenmp $(CUDA_ARCH)
LDFLAGS= -shared -fopenmp -L$(CUDA_HOME)/lib64 -lcudart -lnvToolsExt

all: cuslines

cuslines: generate_streamlines_cuda.o cuslines.o
	$(LD) generate_streamlines_cuda.o cuslines.o -o cuslines`python3-config --extension-suffix` $(LDFLAGS)

%.o : %.cu
	$(CUDACC) $(CUDACFLAGS) $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm *.o cuslines`python3-config --extension-suffix` __pycache__/*.pyc
