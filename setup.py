"""
Setup of codebase
Author: Michael Danielczuk
"""
import os
import sys

from setuptools import find_packages, setup

root_dir = os.path.dirname(os.path.realpath(__file__))

requirements_default = set(
    [
        "numpy",  # Basic math/array utilities
        "tqdm",  # Progress bars
        "h5py",  # Reading/writing dataset info
        "trimesh",  # Mesh loading/utilities
    ]
)

requirements_train = set(
    [
        "pyrender",  # Scene rendering
        "python-fcl",  # Scene GT collisions
        "urdfpy",  # Robot FK
        "torch>=1.5",  # Training/Benchmarking
        "torch-scatter",  # For collision models
        f"pointnet2 @ file://localhost{root_dir}/extern/pointnet2",  # Network modules
        "autolab_core",  # used for loading cfg files and Image classes
    ]
)

requirements_bench = requirements_train.union(
    [
        "matplotlib",  # Plotting
        "seaborn",  # Fancy plotting
        "pandas",  # Loss plotting made easy
        "natsort",  # Loss plotting sort files
        "pygifsicle",  # Benchmarking GIFs
        "imageio",  # Benchmarking GIFs
    ]
)

requirements_policy = requirements_train.union(
    [
        f"tracikpy @ file://localhost{root_dir}/extern/tracikpy",  # Fast IK solutions
        "KNN-CUDA @ https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl",  # CUDA impl of KNN
        f"isaacgym @ file://localhost{root_dir}/extern/isaacgym/python",  # Isaac Gym Simulation
    ]
)

# if someone wants to output a requirements file
# `python setup.py --list-train > requirements.txt`
if "--list-train" in sys.argv:
    print("\n".join(requirements_train.union(requirements_default)))
    exit()
elif "--list-bench" in sys.argv:
    print("\n".join(requirements_bench.union(requirements_default)))
    exit()

# load __version__ without importing anything
version_file = os.path.join(
    os.path.dirname(__file__), "scenecollisionnet/version.py"
)
with open(version_file, "r") as f:
    # use eval to get a clean string of version from file
    __version__ = eval(f.read().strip().split("=")[-1])

# load README.md as long_description
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r") as f:
        long_description = f.read()

setup(
    name="scenecollisionnet",
    version=__version__,
    description="SceneCollisionNet ICRA21 Paper Code",
    long_description=long_description,
    author="Michael Danielczuk",
    author_email="mdanielczuk@berkeley.edu",
    license="MIT Software License",
    url="https://github.com/mjd3/SceneCollisionNet",
    keywords="robotics computer vision",
    classifiers=[
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(),
    install_requires=list(requirements_default),
    extras_require={
        "train": list(requirements_train),
        "bench": list(requirements_bench),
        "policy": list(requirements_policy),
    },
)
