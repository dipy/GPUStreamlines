# GPUStreamlines

## Installation
When cloning the repo, please use "git clone --recursive" to pull all the requirements.

To install, simply run `pip install .` in the top-level repository directory.

## Running the examples
This repository contains several example usage scripts.

There are two example scripts using the HARDI dataset, `run_dipy_cpu_hardi.py` and `run_dipy_gpu_hardi.py`, which run on CPU and GPU respectively.

To run the baseline CPU example on a random set of 1000 seeds, this is the command and example output:
```
$ python run_dipy_cpu_hardi.py --chunk-size 100000 --output-prefix small --nseeds 1000
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
$ python run_dipy_gpu_hardi.py --chunk-size 100000 --output-prefix small --nseeds 1000 --ngpus 1
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

To run on more seeds, we suggest enabling the `--use-fast-write` flag in the GPU script to not get bottlenecked by writing files. Here is a comparison running on 500K seeds on 1 GPU with and without this flag:

Without `--use-fast-write`:
```
$ python run_dipy_gpu_hardi.py --chunk-size 100000 --output-prefix small --nseeds 500000 --ngpus 1
parsing arguments
Fitting Tensor
Computing anisotropy measures (FA,MD,RGB)
slowadcodf
Bootstrap direction getter
streamline gen
Creating GPUTracker with 1 GPUs...
Generated 143891 streamlines from 100000 seeds, time: 7.978902339935303 s
Saved streamlines to small.1_5.trk, time 11.439777851104736 s
Generated 151932 streamlines from 100000 seeds, time: 10.155118703842163 s
Saved streamlines to small.2_5.trk, time 12.438884019851685 s
Generated 146971 streamlines from 100000 seeds, time: 9.822870016098022 s
Saved streamlines to small.3_5.trk, time 12.377111673355103 s
Generated 153824 streamlines from 100000 seeds, time: 11.133368968963623 s
Saved streamlines to small.4_5.trk, time 13.317519187927246 s
Generated 162004 streamlines from 100000 seeds, time: 13.19784665107727 s
Saved streamlines to small.5_5.trk, time 14.21276593208313 s
Completed processing 500000 seeds.
Initialization time: 14.789637088775635 sec
Streamline generation total time: 116.0746865272522 sec
        Streamline processing: 52.28810667991638 sec
        File writing: 63.7860586643219 sec
Destroy GPUTracker...
```

With `--use-fast-write`:
```
$ python run_dipy_gpu_hardi.py --chunk-size 100000 --output-prefix small --nseeds 500000 --ngpus 1 --use-fast-write
parsing arguments
Fitting Tensor
Computing anisotropy measures (FA,MD,RGB)
slowadcodf
Bootstrap direction getter
streamline gen
Creating GPUTracker with 1 GPUs...
Generated 143891 streamlines from 100000 seeds, time: 7.962322473526001 s
Saved streamlines to small.1_5_*.trk, time 0.1053612232208252 s
Generated 151932 streamlines from 100000 seeds, time: 10.148677825927734 s
Saved streamlines to small.2_5_*.trk, time 0.1606450080871582 s
Generated 146971 streamlines from 100000 seeds, time: 9.811130285263062 s
Saved streamlines to small.3_5_*.trk, time 0.571892499923706 s
Generated 153824 streamlines from 100000 seeds, time: 11.186563968658447 s
Saved streamlines to small.4_5_*.trk, time 0.3091111183166504 s
Generated 162004 streamlines from 100000 seeds, time: 13.282683610916138 s
Saved streamlines to small.5_5_*.trk, time 0.7107999324798584 s
Completed processing 500000 seeds.
Initialization time: 14.705361366271973 sec
Streamline generation total time: 54.24975609779358 sec
        Streamline processing: 52.39137816429138 sec
        File writing: 1.8578097820281982 sec
Destroy GPUTracker...
```

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
 python run_dipy_gpu_hardi.py --chunk-size 100000 --ngpus 1 --output-prefix output/hardi_gpu_full --use-fast-write
```
5. The code produces a number of independent track files (one per processed "chunk"), but we have provided a merge script to combine them into a single trk file. To merge files, run:
```
$ docker run --gpus=all -v ${PWD}:/opt/exec/output:rw -it docker.pkg.github.com/dipy/gpustreamlines/gpustreamlines:latest \
 ./merge_trk.sh -o output/hardi_tracks.trk output/hardi_gpu_full*
```

