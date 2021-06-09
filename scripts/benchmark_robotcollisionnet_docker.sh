#!/bin/bash

cd /SceneCollisionNet
python3 tools/benchmark_robotcollisionnet.py --cfg /cfg/benchmark_robotcollisionnet.yaml $@
