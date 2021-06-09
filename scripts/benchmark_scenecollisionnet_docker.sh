#!/bin/bash

cd /SceneCollisionNet
PYOPENGL_PLATFORM=egl python3 tools/benchmark_scenecollisionnet.py --cfg /cfg/benchmark_scenecollisionnet.yaml $@
