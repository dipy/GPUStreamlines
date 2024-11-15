ARG NVIDIAVERSION=12.0.1-devel-ubuntu20.04
FROM nvidia/cuda:${NVIDIAVERSION}

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --assume-yes apt-transport-https \
	ca-certificates gnupg software-properties-common gcc git wget numactl cmake

RUN wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
RUN bash /tmp/Anaconda3-2022.10-Linux-x86_64.sh -b -p /opt/anaconda
RUN rm -rf /tmp/Anaconda3-2022.10-Linux-x86_64.sh
RUN cd /opt && eval "$(/opt/anaconda/bin/conda shell.bash hook)"
ENV PATH /opt/anaconda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/anaconda/lib:${LD_LIBRARY_PATH}

# python prereqs
RUN pip install numpy scipy cython nibabel dipy tqdm

COPY GPUStreamlines /opt/GPUStreamlines
RUN cd /opt/GPUStreamlines && pip install .

# compile
RUN cd /opt/GPUStreamlines && mkdir build && cd build \
    &&  cmake -DCMAKE_INSTALL_PREFIX=/opt/exec/ \
      	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CXX_COMPILER=g++ \
	-DPYTHON_EXECUTABLE=$(which python) \
	.. \
    && make && make install

WORKDIR /opt/exec
