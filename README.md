# GPUStreamlines

## Installation
To install from pypi, simply run `pip install "cuslines[cu13]"` or `pip install "cuslines[cu12]"` depending on your CUDA version.

For Apple Silicon (M1/M2/M3/M4), install the Metal backend: `pip install "cuslines[metal]"`

To install from dev, simply run `pip install ".[cu13]"` or `pip install ".[cu12]"` (or `pip install ".[metal]"` on macOS) in the top-level repository directory.

The GPU backend is auto-detected at import time. On macOS with Apple Silicon, Metal is used; on Linux/Windows with an NVIDIA GPU, CUDA is used.

## Running the examples
This repository contains several example usage scripts.

The script `run_gpu_streamlines.py` demonstrates how to run any diffusion MRI dataset on the GPU. It can also run on the CPU for reference, if the argument `--device=cpu` is used. Use `--device=metal` to explicitly select the Metal backend on macOS. If no data is passed, it will download and use the HARDI dataset.

To run the baseline CPU example on a random set of 1000 seeds, this is the command and example output:
```
$ python run_gpu_streamlines.py --device=cpu --output-prefix small --nseeds 1000
parsing arguments
Fitting Tensor
Computing anisotropy measures (FA,MD,RGB)
slowadcodf
Bootstrap direction getter
streamline gen
Generated 2746 streamlines from 1000 seeds, time: 6.713643550872803 s
Saved streamlines to small.1_1.trk, time 0.22669768333435059 s
Completed processing 1000 seeds.
Initialization time: 12.355878829956055 sec
Streamline generation total time: 6.9404990673065186 sec
        Streamline processing: 6.713643550872803 sec
        File writing: 0.22669768333435059 sec
```

To run the same case on a single GPU, this is the command and example output:
```
$ python run_gpu_streamlines.py --output-prefix small --nseeds 1000 --ngpus 1
parsing arguments
Fitting Tensor
Computing anisotropy measures (FA,MD,RGB)
slowadcodf
Bootstrap direction getter
streamline gen
Creating GPUTracker with 1 GPUs...
Generated 2512 streamlines from 1000 seeds, time: 0.21228599548339844 s
Saved streamlines to small.1_1.trk, time 0.17112255096435547 s
Completed processing 1000 seeds.
Initialization time: 14.81659483909607 sec
Streamline generation total time: 0.3834989070892334 sec
        Streamline processing: 0.21228599548339844 sec
        File writing: 0.17112255096435547 sec
Destroy GPUTracker...
```

Note that if you experience memory errors, you can adjust the `--chunk-size` flag.

To run on more seeds, we suggest setting the `--write-method trx` flag in the GPU script to not get bottlenecked by writing files.

## Running on AWS with Docker
First, set up an AWS instance with GPU and ssh into it (we recommend a P3 instance with at least 1 V100 16 GB GPU and a Deep Learning AMI Ubuntu 18.04 v 33.0.). Then do the following:
1. Log in to GitHub docker registry:
```
$ docker login -u <github id> docker.pkg.github.com
```
2. Enter your GitHub access token. If you do not have one, create it on the GitHub general security settings page and enable package read access for that token.
3. Pull the container:
```
$ docker pull docker.pkg.github.com/dipy/gpustreamlines/gpustreamlines:latest
```
4. Run the code, mounting the current directory into the container for easy result retrieval:
```
$ docker run --gpus=all -v ${PWD}:/opt/exec/output:rw -it docker.pkg.github.com/dipy/gpustreamlines/gpustreamlines:latest \
 python /opt/GPUStreamlines/run_gpu_streamlines.py --ngpus 1 --output-prefix /opt/exec/output/hardi_gpu_full
```
