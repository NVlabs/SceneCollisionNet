#!/bin/bash

cd /SceneCollisionNet
PYOPENGL_PLATFORM=egl python3 tools/train_scenecollisionnet.py --cfg /cfg/train_scenecollisionnet.yaml $@
