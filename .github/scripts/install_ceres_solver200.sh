#!/bin/bash

sudo apt install libunwind-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev
wget https://github.com/ceres-solver/ceres-solver/archive/2.0.0.zip
unzip 2.0.0.zip
cd ceres-solver-2.0.0
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/ceres-solver200 ..
njobs=`lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l` # https://stackoverflow.com/a/23378780
make -j{njobs} # njobs - число потоков которое будет использоваться для компиляции (по числу ядер)
sudo make install
