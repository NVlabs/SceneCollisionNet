# SceneCollisionNet
This repo contains the code for "[Object Rearrangement Using Learned Implicit Collision Functions](https://arxiv.org/abs/2011.10726)", an ICRA 2021 paper. For more information, please visit the [project website](https://research.nvidia.com/publication/2021-03_Object-Rearrangement-Using).

## Install and Setup
Clone and install the repo (we recommend a virtual environment, especially if training or benchmarking, to avoid dependency conflicts):
```shell
git clone --recursive https://github.com/mjd3/SceneCollisionNet.git
cd SceneCollisionNet
pip install -e .
```
These commands install the minimum dependencies needed for generating a mesh dataset and then training/benchmarking using Docker. If you instead wish to train or benchmark without using Docker, please first install an appropriate version of [PyTorch](https://pytorch.org/get-started/locally/) and corresponding version of [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) for your system. Then, execute these commands:
```shell
git clone --recursive https://github.com/mjd3/SceneCollisionNet.git
cd SceneCollisionNet
pip install -e .[train]
```
If benchmarking, replace `train` in the last command with `bench`.

To rollout the object rearrangement MPPI policy in a simulated tabletop environment, first download [Isaac Gym](https://developer.nvidia.com/isaac-gym) and place it in the `extern` folder within this repo. Next, follow the previous installation instructions for training, but replace the `train` option with `policy`.

To download the pretrained weights for benchmarking or policy rollout, run `bash scripts/download_weights.sh`.

## Generating a Mesh Dataset
To save time during training/benchmarking, meshes are preprocessed and mesh stable poses are calculated offline. SceneCollisionNet was trained using the [ACRONYM dataset](https://sites.google.com/nvidia.com/graspdataset). To use this dataset for training or benchmarking, download the ShapeNetSem meshes [here](https://shapenet.org/) (note: you must first register for an account) and the ACRONYM grasps [here](https://sites.google.com/nvidia.com/graspdataset). Next, build Manifold (an external library included as a submodule):
```shell
./scripts/install_manifold.sh
```

Then, use the following script to generate a preprocessed version of the ACRONYM dataset:
```shell
python tools/generate_acronym_dataset.py /path/to/shapenetsem/meshes /path/to/acronym datasets/shapenet
```

If you have your own set of meshes, run:
```shell
python tools/generate_mesh_dataset.py /path/to/meshes datasets/your_dataset_name
```
Note that this dataset will not include grasp data, which is not needed for training or benchmarking SceneCollisionNet, but is be used for rolling out the MPPI policy.

## Training/Benchmarking with Docker
First, install Docker and `nvidia-docker2` following the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian). Pull the SceneCollisionNet docker image from DockerHub (tag `scenecollisionnet`) or build locally using the provided Dockerfile (`docker build -t scenecollisionnet .`). Then, use the appropriate configuration `.yaml` file in `cfg` to set training or benchmarking parameters (note that cfg file paths are relative to the Docker container, not the local machine) and run one of the commands below (replacing paths with your local paths as needed; `-v` requires absolute paths).

### Train a SceneCollisionNet
Edit `cfg/train_scenecollisionnet.yaml`, then run:
```shell
docker run --gpus all --rm -it -v /path/to/dataset:/dataset:ro -v /path/to/models:/models:rw -v /path/to/cfg:/cfg:ro scenecollisionnet /SceneCollisionNet/scripts/train_scenecollisionnet_docker.sh
```

### Train a RobotCollisionNet
Edit `cfg/train_robotcollisionnet.yaml`, then run:
```shell
docker run --gpus all --rm -it -v /path/to/models:/models:rw -v /path/to/cfg:/cfg:ro scenecollisionnet /SceneCollisionNet/scripts/train_robotcollisionnet_docker.sh
```

### Benchmark a SceneCollisionNet
Edit `cfg/benchmark_scenecollisionnet.yaml`, then run:
```shell
docker run --gpus all --rm -it -v /path/to/dataset:/dataset:ro -v /path/to/models:/models:ro -v /path/to/cfg:/cfg:ro -v /path/to/benchmark_results:/benchmark:rw scenecollisionnet /SceneCollisionNet/scripts/benchmark_scenecollisionnet_docker.sh
```


### Benchmark a RobotCollisionNet
Edit `cfg/benchmark_robotcollisionnet.yaml`, then run:
```shell
docker run --gpus all --rm -it -v /path/to/models:/models:rw -v /path/to/cfg:/cfg:ro -v /path/to/benchmark_results:/benchmark:rw scenecollisionnet /SceneCollisionNet/scripts/train_robotcollisionnet_docker.sh
```

### Loss Plots
To get loss plots while training, run:
```shell
docker exec -d <container_name> python3 tools/loss_plots.py /models/<model_name>/log.csv
```

### Benchmark FCL or SDF Baselines
Edit `cfg/benchmark_baseline.yaml`, then run:
```shell
docker run --gpus all --rm -it -v /path/to/dataset:/dataset:ro -v /path/to/benchmark_results:/benchmark:rw -v /path/to/cfg:/cfg:ro scenecollisionnet /SceneCollisionNet/scripts/benchmark_baseline_docker.sh
```

## Training/Benchmarking without Docker
First, install system dependencies. The system dependencies listed assume an Ubuntu 18.04 install with NVIDIA drivers >= 450.80.02 and CUDA 10.2. You can adjust the dependencies accordingly for different driver/CUDA versions. Note that the NVIDIA drivers come packaged with EGL, which is used during training and benchmarking for headless rendering on the GPU.

### System Dependencies
See Dockerfile for a full list. For training/benchmarking, you will need:
```
python3-dev
python3-pip
ninja-build
libcudnn8=8.1.1.33-1+cuda10.2
libcudnn8-dev=8.1.1.33-1+cuda10.2
libsm6
libxext6
libxrender-dev
freeglut3-dev
liboctomap-dev
libfcl-dev
gifsicle
libfreetype6-dev
libpng-dev
```

### Python Dependencies
Follow the instructions above to install the necessary dependencies for your use case (either the `train`, `bench`, or `policy` options).

### Train a SceneCollisionNet
Edit `cfg/train_scenecollisionnet.yaml`, then run:
```shell
PYOPENGL_PLATFORM=egl python tools/train_scenecollisionnet.py
```

### Train a RobotCollisionNet
Edit `cfg/train_robotcollisionnet.yaml`, then run:
```shell
python tools/train_robotcollisionnet.py
```

### Benchmark a SceneCollisionNet
Edit `cfg/benchmark_scenecollisionnet.yaml`, then run:
```shell
PYOPENGL_PLATFORM=egl python tools/benchmark_scenecollisionnet.py
```

### Benchmark a RobotCollisionNet
Edit `cfg/benchmark_robotcollisionnet.yaml`, then run:
```shell
python tools/benchmark_robotcollisionnet.py
```

### Benchmark FCL or SDF Baselines
Edit `cfg/benchmark_baseline.yaml`, then run:
```shell
PYOPENGL_PLATFORM=egl python tools/benchmark_baseline.py
```

## Policy Rollout
To view a rearrangement MPPI policy rollout in a simulated Isaac Gym tabletop environment, run the following command (note that this requires a local machine with an available GPU and display):
```shell
python tools/rollout_policy.py --self-coll-nn weights/self_coll_nn --scene-coll-nn weights/scene_coll_nn --control-frequency 1
```
There are many possible options for this command that can be viewed using the `--help` command line argument and set with the appropriate argument. If you get `RuntimeError: CUDA out of memory`, try reducing the horizon (`--mppi-horizon`, default 40), number of trajectories (`--mppi-num-rollouts`, default 200) or collision steps (`--mppi-collision-steps`, default 10). Note that this may affect policy performance.

## Citation
If you use this code in your own research, please consider citing:
```
@inproceedings{danielczuk2021object,
  title={Object Rearrangement Using Learned Implicit Collision Functions},
  author={Danielczuk, Michael and Mousavian, Arsalan and Eppner, Clemens and Fox, Dieter},
  booktitle={Proc. IEEE Int. Conf. Robotics and Automation (ICRA)},
  year={2021}
}
```
