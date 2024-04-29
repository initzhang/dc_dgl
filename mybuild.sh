#!/bin/bash
git submodule update --init --recursive
mkdir -p build
cd build
cmake -DBUILD_TYPE=release -DUSE_CUDA=ON -DBUILD_TORCH=ON -DCMAKE_C_COMPILER=/usr/local/bin/gcc9 -DCMAKE_CXX_COMPILER=/usr/local/bin/g++9 ..
make -j64
cd ../python; python setup.py install
