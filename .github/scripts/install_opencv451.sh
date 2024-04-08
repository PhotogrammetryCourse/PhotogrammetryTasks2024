#!/bin/bash

wget https://github.com/opencv/opencv/archive/4.5.1.zip
unzip 4.5.1.zip
cd opencv-4.5.1
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_LIST=features2d,highgui,flann,calib3d -DWITH_OPENEXR=ON -DBUILD_EXAMPLES=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/opt/opencv451  ..
njobs=`lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l` # https://stackoverflow.com/a/23378780
make -j{njobs} # njobs - число потоков которое будет использоваться для компиляции (по числу ядер)
sudo make install
