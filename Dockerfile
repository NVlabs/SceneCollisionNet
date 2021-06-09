FROM nvidia/cudagl:10.2-devel-ubuntu18.04

# env variables for tzdata install
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Vancouver

RUN apt-get update -y && \
    apt-get install -y \
        python3-pip \ 
        python3-dev \
        ninja-build \
        libcudnn8=8.1.1.33-1+cuda10.2 \
        libcudnn8-dev=8.1.1.33-1+cuda10.2 \
        libsm6 libxext6 libxrender-dev \
        freeglut3-dev \
        liboctomap-dev libfcl-dev \
        gifsicle libfreetype6-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get install -y python3.8 python3.8-dev python3-pip

# Install pytorch and pytorch scatter deps (need to match CUDA version)
RUN pip3 install --no-cache-dir \
    torch==1.8.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html \
    torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html

# Build and install pointnet2 dep
RUN mkdir -p SceneCollisionNet
COPY [ "./extern",  "SceneCollisionNet/extern" ]
RUN TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.5 8.0 8.6+PTX" python3 -m pip install --no-cache-dir SceneCollisionNet/extern/pointnet2

# Build and install kaolin dep (for SDF baselines)
RUN IGNORE_TORCH_VER=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.5 8.0 8.6+PTX" python3 -m pip install --no-cache-dir SceneCollisionNet/extern/kaolin

# Install all other python deps
COPY [ "./setup.py",  "SceneCollisionNet/setup.py" ]
RUN python3 -m pip install --no-cache-dir `python3 SceneCollisionNet/setup.py --list-bench`

# Install repo
COPY [ ".",  "SceneCollisionNet/" ]
RUN python3 -m pip install --no-cache-dir SceneCollisionNet/
