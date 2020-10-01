FROM nvcr.io/nvidia/cuda:11.0-devel

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# upgrade
RUN apt update && \
    apt install --assume-yes apt-transport-https \
    	ca-certificates gnupg \
	software-properties-common gcc git wget numactl
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    	      | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ focal main"
RUN apt install -y cmake libncurses5-dev libtinfo6

# Anaconda
RUN cd /opt && wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh \
    && bash /tmp/Anaconda3-2020.02-Linux-x86_64.sh -b -p /opt/anaconda \
    && rm -rf /tmp/Anaconda3-2020.02-Linux-x86_64.sh \
    && eval "$(/opt/anaconda/bin/conda shell.bash hook)"
ENV PATH /opt/anaconda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/anaconda/lib:${LD_LIBRARY_PATH}

# python prereqs
RUN pip install numpy scipy cython nibabel dipy tqdm

# copy stuff
COPY CMakeLists.txt /opt/GPUStreamlines/CMakeLists.txt
COPY run_dipy_gpu.py /opt/GPUStreamlines/run_dipy_gpu.py
COPY run_dipy_cpu.py /opt/GPUStreamlines/run_dipy_cpu.py
COPY run_dipy_gpu_hardi.py /opt/GPUStreamlines/run_dipy_gpu_hardi.py
COPY run_dipy_cpu_hardi.py /opt/GPUStreamlines/run_dipy_cpu_hardi.py
COPY merge_trk.sh /opt/exec/merge_trk.sh
COPY cuslines /opt/GPUStreamlines/cuslines
COPY external /opt/GPUStreamlines/external
RUN mkdir -p /opt/exec/output

# compile
RUN cd /opt/GPUStreamlines && mkdir build && cd build \
    &&  cmake -DCMAKE_INSTALL_PREFIX=/opt/exec/ \
      	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CXX_COMPILER=g++ \
	-DPYTHON_EXECUTABLE=$(which python) \
	.. \
    && make && make install

WORKDIR /opt/exec
