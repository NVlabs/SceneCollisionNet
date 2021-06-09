#!/bin/bash

cd /SceneCollisionNet
PYOPENGL_PLATFORM=egl python3 tools/benchmark_baseline.py --cfg /cfg/benchmark_baseline.yaml $@
