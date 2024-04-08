cmake_minimum_required(VERSION 3.1)

add_subdirectory(libs)

project(Photogrammetry)

set(CMAKE_CXX_STANDARD 17)

# OpenMP позволит распараллеливать циклы на все ядра процессора простыми директивами
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP not found! You need OpenMP for speedup on multicore CPUs!")
endif()

find_package(OpenCV 4.5.1 REQUIRED)
#      Linux:
# скачайте Source code (opencv-4.5.1.zip) отсюда - https://github.com/opencv/opencv/releases распакуйте и скомпилируйте:
# wget https://github.com/opencv/opencv/archive/4.5.1.zip
# unzip 4.5.1.zip
# cd opencv-4.5.1
# mkdir build
# cd build
# cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_LIST=features2d,highgui,flann,calib3d -DWITH_OPENEXR=ON -DBUILD_EXAMPLES=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/opt/opencv451  ..
# njobs=`lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l` # https://stackoverflow.com/a/23378780
# make -j{njobs} # njobs - число потоков которое будет использоваться для компиляции (по числу ядер)
# sudo make install
# укажите для CMake путь к OpenCV:
# в CLion это делается через File->Settings->Build, Execution, Deployment->CMake->CMake options: -DOpenCV_DIR=/opt/opencv451/lib/cmake/opencv4
# там же сразу стоит добавить в дополнению к Debug сборке - сборку с включенными оптимизациями и отладочными символами - RelWithDebInfo
#
#     Windows: (но рекомендуется использовать Linux)
# Скачайте этот скрипт и следуйте инструкции - https://gist.github.com/PolarNick239/18036d942ae4b9aad9f1da756ce6c845#file-buildocv-sh-L10-L33

find_package(Eigen3 3.3 REQUIRED)
#      Linux:
# sudo apt install libeigen3-dev
#      Windows (чтобы удобно копировать в Git Bash консоль: выделите, нажмите Ctrl+C, в консоли нажмите Shift+Insert):
# 1) Скачать https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.zip
# 2) Распаковать и открыть Git Bash в папке распаковки (правый клик в проводнике -> Git Bash here)
# 3) myDirPath=$(pwd)
# 4) mkdir build_dir
# 5) cd build_dir
# 6) cmake -G "Visual Studio 16 2019" -DCMAKE_INSTALL_PREFIX="$myDirPath/install_dir" ..
# 7) cmake --build .  --target install
# 8) запомните папку куда была проведена установка Eigen3 - она имеет вид .../eigen-3.3.9/install_dir
# 9) Добавьте в File->Settings->CMake options: -DEigen3_DIR=<ПОПРАВЬТЕ ЭТОТ ПУТЬ>/eigen-3.3.9/install_dir/share/eigen3/cmake

find_package(Ceres 2.0.0 REQUIRED)
#      Если вы на Mac - http://ceres-solver.org/installation.html#macos
#      Linux:
# sudo apt install libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev
# wget https://github.com/ceres-solver/ceres-solver/archive/2.0.0.zip
# unzip 2.0.0.zip
# cd ceres-solver-2.0.0
# mkdir build
# cd build
# cmake -DCMAKE_INSTALL_PREFIX=/opt/ceres-solver200 -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF ..
# njobs=`lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l` # https://stackoverflow.com/a/23378780
# make -j{njobs} # njobs - число потоков которое будет использоваться для компиляции (по числу ядер)
# make test
# sudo make install
# указать для CMake путь к ceres-solver (в дополнение к OpenCV и Eigen3):
# File->Settings->Build, Execution, Deployment->CMake->CMake options: -DCeres_DIR=/opt/ceres-solver200/lib/cmake/Ceres
# (стоит добавить и к Debug сборке)
#
#      Windows (чтобы удобно копировать в Git Bash консоль: выделите, нажмите Ctrl+C, в консоли нажмите Shift+Insert):
#      (учитывайте что компиляция ceres-solver займет некоторое время, в теории можно ускорить это за счет многпоточности, но видимо для этого надо устанавливать ninja)
# 1) Скачать https://github.com/ceres-solver/ceres-solver/archive/2.0.0.zip
# 2) Распаковать и открыть Git Bash в папке распаковки (правый клик в проводнике -> Git Bash here)
# 3) myDirPath=$(pwd)
# 4) mkdir build_dir
# 5) cd build_dir
#     поправьте в команде путь к установке Eigen:
# 6)  cmake -G "Visual Studio 16 2019" -DMINIGLOG=ON -DEigen3_DIR=<ПОПРАВЬТЕ ЭТОТ ПУТЬ>/eigen-3.3.9/install_dir/share/eigen3/cmake -DCMAKE_INSTALL_PREFIX="$myDirPath/install_dir" ..
# 7)  cmake --build . --config debug
# 8)  cmake --build . --config RelWithDebInfo
# 9)  cmake --build . --target install --config debug
# 10) cmake --build . --target install --config RelWithDebInfo
#     указать для CMake путь к ceres-solver (в дополнение к OpenCV и Eigen3):
#     в CLion это делается через File->Settings->Build, Execution, Deployment->CMake->CMake options: -DCeres_DIR=<ПОПРАВЬТЕ ЭТОТ ПУТЬ>\ceres-solver-2.0.0\install_dir\CMake
#     (стоит добавить и к Debug сборке)

set(SOURCES
        src/phg/core/calibration.cpp
        src/phg/core/calibration.h
        src/phg/core/camera.cpp
        src/phg/core/camera.h
        src/phg/sift/sift.cpp
        src/phg/sift/sift.h
        src/phg/matching/descriptor_matcher.cpp
        src/phg/matching/descriptor_matcher.h
        src/phg/matching/bruteforce_matcher.cpp
        src/phg/matching/bruteforce_matcher.h
        src/phg/matching/bruteforce_matcher_gpu.cpp
        src/phg/matching/bruteforce_matcher_gpu.h
        src/phg/matching/gms_matcher.cpp
        src/phg/matching/gms_matcher.h
        src/phg/matching/gms_matcher_impl.h
        src/phg/matching/flann_matcher.cpp
        src/phg/matching/flann_matcher.h
        src/phg/matching/flann_factory.h
        src/phg/mvs/depth_maps/pm_depth_maps.cpp
        src/phg/mvs/depth_maps/pm_depth_maps.h
        src/phg/mvs/depth_maps/pm_fast_random.cpp
        src/phg/mvs/depth_maps/pm_fast_random.h
        src/phg/mvs/depth_maps/pm_geometry.cpp
        src/phg/mvs/depth_maps/pm_geometry.h
        src/phg/sfm/defines.h
        src/phg/sfm/ematrix.cpp
        src/phg/sfm/ematrix.h
        src/phg/sfm/fmatrix.cpp
        src/phg/sfm/fmatrix.h
        src/phg/sfm/homography.cpp
        src/phg/sfm/homography.h
        src/phg/sfm/panorama_stitcher.cpp
        src/phg/sfm/panorama_stitcher.h
        src/phg/sfm/resection.cpp
        src/phg/sfm/resection.h
        src/phg/sfm/sfm_utils.cpp
        src/phg/sfm/sfm_utils.h
        src/phg/sfm/triangulation.cpp
        src/phg/sfm/triangulation.h
        src/phg/utils/cameras_bundler_export.cpp
        src/phg/utils/cameras_bundler_export.h
        src/phg/utils/cameras_bundler_import.cpp
        src/phg/utils/cameras_bundler_import.h
        src/phg/utils/point_cloud_export.cpp
        src/phg/utils/point_cloud_export.h
        )

# convertIntoHeader CMake-функция объявлена в libs/gpu/CMakeLists.txt:71
# Она считывает все байты из файла bruteforce_matcher.cl (т.е. весь исходный код кернела) и преобразует их в массив байтов в файле bruteforce_matcher_cl.h bruteforce_matcher_kernel
convertIntoHeader(src/phg/matching/cl/bruteforce_matcher.cl src/phg/matching/cl/bruteforce_matcher_cl.h bruteforce_matcher_kernel)
# Обратите внимание что это происходит на этапе компиляции, кроме того необходимо чтобы файл bruteforce_matcher_cl.h был перечислен среди исходников для компиляции при вызове add_executable
set(SOURCES ${SOURCES} src/phg/matching/cl/bruteforce_matcher_cl.h)

set(LIBRARIES
        libutils
        ${OpenCV_LIBS}
        ${OpenMP_CXX_LIBRARIES}
        Eigen3::Eigen
        Ceres::ceres
        )

add_library(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)

set(TEST_SOURCES
        tests/utils/test_utils.cpp
        tests/utils/test_utils.h
        )

set(TEST_LIBRARIES
        gtest
        gtest_main
        )

add_executable(test_sift            tests/test_sift.cpp             ${TEST_SOURCES})
add_executable(test_matching        tests/test_matching.cpp         ${TEST_SOURCES})
add_executable(test_sfm             tests/test_sfm.cpp              ${TEST_SOURCES})
add_executable(test_ceres_solver    tests/test_ceres_solver.cpp     ${TEST_SOURCES})
add_executable(test_sfm_ba          tests/test_sfm_ba.cpp           ${TEST_SOURCES})
add_executable(test_depth_maps_pm   tests/test_depth_maps_pm.cpp    ${TEST_SOURCES})

target_link_libraries(test_sift          ${PROJECT_NAME} ${LIBRARIES} ${TEST_LIBRARIES})
target_link_libraries(test_matching      ${PROJECT_NAME} ${LIBRARIES} ${TEST_LIBRARIES})
target_link_libraries(test_sfm           ${PROJECT_NAME} ${LIBRARIES} ${TEST_LIBRARIES})
target_link_libraries(test_ceres_solver  ${PROJECT_NAME} ${LIBRARIES} ${TEST_LIBRARIES})
target_link_libraries(test_sfm_ba        ${PROJECT_NAME} ${LIBRARIES} ${TEST_LIBRARIES})
target_link_libraries(test_depth_maps_pm ${PROJECT_NAME} ${LIBRARIES} ${TEST_LIBRARIES})
