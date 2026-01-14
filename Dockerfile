ARG NVIDIAVERSION=12.0.1-devel-ubuntu20.04
FROM nvidia/cuda:${NVIDIAVERSION}

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --assume-yes curl

RUN curl -L "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
    -o "/tmp/Miniconda3.sh"
RUN bash /tmp/Miniconda3.sh -b -p /opt/anaconda
RUN rm -rf /tmp/Miniconda3.sh
RUN cd /opt && eval "$(/opt/anaconda/bin/conda shell.bash hook)"
ENV PATH=/opt/anaconda/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH}

COPY . /opt/GPUStreamlines/
RUN cd /opt/GPUStreamlines && pip install .
