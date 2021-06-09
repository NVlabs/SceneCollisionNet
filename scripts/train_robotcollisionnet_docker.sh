#!/bin/bash

cd /SceneCollisionNet
python3 tools/train_robotcollisionnet.py --cfg /cfg/train_robotcollisionnet.yaml $@
