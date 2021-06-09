#!/bin/bash

# Install Manifold
cd extern/Manifold
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
