ARG NVIDIAVERSION=12.0.1-devel-ubuntu20.04
FROM nvidia/cuda:${NVIDIAVERSION}

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --assume-yes apt-transport-https \
	ca-certificates gnupg software-properties-common \
	gcc git wget curl numactl
RUN apt install -y cmake libncurses5-dev libtinfo6

RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0-linux-x86_64.sh \
    -O /tmp/cmake-install.sh \
    && chmod +x /tmp/cmake-install.sh \
    && mkdir /opt/cmake \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake \
    && rm /tmp/cmake-install.sh
ENV PATH /opt/cmake/bin:${PATH}

RUN curl -L "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
    -o "/tmp/Miniconda3.sh"
RUN bash /tmp/Miniconda3.sh -b -p /opt/anaconda
RUN rm -rf /tmp/Miniconda3.sh
RUN cd /opt && eval "$(/opt/anaconda/bin/conda shell.bash hook)"
ENV PATH /opt/anaconda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/anaconda/lib:${LD_LIBRARY_PATH}

# python prereqs
RUN conda install -c conda-forge git
RUN pip install numpy>=2.0.0
RUN pip install scipy>=1.13.0 cython nibabel dipy tqdm
RUN pip install pybind11

COPY . /opt/GPUStreamlines/

# compile
RUN cd /opt/GPUStreamlines && mkdir build && cd build \
    &&  cmake -DCMAKE_INSTALL_PREFIX=/opt/exec/ \
      	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CXX_COMPILER=g++ \
	-DPYTHON_EXECUTABLE=$(which python) \
	.. \
    && make && make install

WORKDIR /opt/exec
