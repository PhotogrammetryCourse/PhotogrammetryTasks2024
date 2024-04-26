#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>

#include <libutils/timer.h>
#include <libutils/rasserts.h>
#include <libutils/string_utils.h>

#include <phg/utils/mesh_export.h>
#include <phg/utils/point_cloud_export.h>
#include <phg/utils/cameras_bundler_import.h>
#include <phg/mvs/depth_maps/pm_depth_maps.h>
#include <phg/mvs/depth_maps/pm_geometry.h>
#include <phg/mvs/model_min_cut/min_cut_model_builder.h>

#include "utils/test_utils.h"

//________________________________________________________________________________
// Datasets:

// Скачайте и распакуйте архивы с картами глубины так чтобы в DATASET_DIR были папки depthmaps_downscaleN с .exr float32-картинками - картами глубины
// - saharov32  (downscales:     x4, x2, x1) - https://disk.yandex.com/d/2fWAdzpM4ibYBg
// - herzjesu25 (downscales: x8, x4, x2, x1) - https://disk.yandex.com/d/n3MyKUjvuVPF6Q

#define DATASET_DIR                  "saharov32"
#define DATASET_DOWNSCALE            4

//#define DATASET_DIR                  "herzjesu25"
//#define DATASET_DOWNSCALE            8
//________________________________________________________________________________

#define CAMERAS_LIMIT                5


void checkFloat32ImageReadWrite()
{
    // проверяем что OpenCV успешно пишет и читает float32 exr-файлы (OpenEXR)
    std::string test_image_path = "test_image_32f.exr";
    int rows = 10;
    int cols = 20;
    cv::Mat img32f(rows, cols, CV_32FC1);
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            img32f.at<float>(j, i) = (j * cols + i) * 32.125f;
        }
    }
    cv::imwrite(test_image_path, img32f);

    cv::Mat copy = cv::imread(test_image_path, cv::IMREAD_UNCHANGED);
    if (copy.empty()) {
        throw std::runtime_error("Can't read float32 image: " + to_string(test_image_path));
    }
    rassert(copy.type() == CV_32FC1, 2381294810217);
    rassert(copy.cols == img32f.cols, 2371827410218);
    rassert(copy.rows == img32f.rows, 2371827410219);

    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            rassert(img32f.at<float>(j, i) == copy.at<float>(j, i), 2381924819223);
        }
    }
}

std::vector<cv::Mat> loadDepthMaps(const std::string &datasetDir, int downscale, const Dataset &dataset)
{
    checkFloat32ImageReadWrite();

    timer t;

    std::vector<cv::Mat> depth_maps(dataset.ncameras);

    for (int ci = 0; ci < dataset.cameras_labels.size(); ++ci) {
        std::string camera_label = dataset.cameras_labels[ci];
        // удаляем расширение картинки (.png, .jpg)
        std::string camera_label_without_extension = camera_label.substr(0, camera_label.find_last_of("."));

        std::string depth_map_path = std::string("data/src/datasets/") + DATASET_DIR + "/depthmaps_downscale" + to_string(downscale) + "/" + camera_label_without_extension + ".exr";
        depth_maps[ci] = cv::imread(depth_map_path, cv::IMREAD_UNCHANGED);

        if (depth_maps[ci].empty()) {
            throw std::runtime_error("Can't read depth map: " + to_string(depth_map_path)); // может быть вы забыли скачать и распаковать в правильную папку карты глубины? см. подробнее выше - "Datasets:"
        }

        rassert(depth_maps[ci].type() == CV_32FC1, 2381294810206);
        rassert(depth_maps[ci].cols == dataset.cameras_imgs[ci].cols, 2371827410207); // т.к. картинки мы тоже уменьшили в downscale раз
        rassert(depth_maps[ci].rows == dataset.cameras_imgs[ci].rows, 2371827410208);
    }

    std::cout << DATASET_DIR << " dataset: " << dataset.ncameras << " depth maps (x" << downscale << " downscale)" << " loaded in " << t.elapsed() << " s" << std::endl;

    return depth_maps;
}

TEST (test_mesh_min_cut, DepthMapsToPointClouds) {
    // TODO 1001: запустите test_mesh_min_cut/DepthMapsToPointClouds и убедитесь что облака точек построенные по картам глубины выглядят правдоподобно
    // Этот тест просто проверяет что логика считывания карт глубины и их интерпретация - работают
    // Для этого вам надо скачать карты глубины по ссылкам выше (см. "Datasets")
    // Затем запустить этот тест, в частности он преобразует карты глубины в облака точек, взгляните на них в папке: data/debug/test_mesh_min_cut/DepthMapsToPointClouds
    Dataset dataset = loadDataset(DATASET_DIR, DATASET_DOWNSCALE);

    std::vector<cv::Mat> depth_maps = loadDepthMaps(DATASET_DIR, DATASET_DOWNSCALE, dataset);

    for (size_t ci = 0; ci < dataset.ncameras; ++ci) {
        const cv::Mat &depth_map = depth_maps[ci];
        vector3d camera_center = phg::invP(dataset.cameras_P[ci]) * phg::homogenize(vector3d(0.0, 0.0, 0.0));
        vector3d camera_dir = cv::normalize((phg::invP(dataset.cameras_P[ci]) * phg::homogenize(vector3d(0.0, 0.0, 1.0))) - camera_center);

        std::vector<vector3d> points;
        std::vector<double> radiuses;
        std::vector<cv::Vec3b> colors;
        std::vector<vector3d> normals;

        phg::buildPoints(depth_maps[ci], dataset.cameras_imgs[ci], dataset.cameras_P[ci], dataset.calibration,
                         points, radiuses, normals, colors);

        points.push_back(camera_center);
        colors.push_back(cv::Vec3b(0, 0, 255));
        normals.push_back(camera_dir);

        std::string depth_map_points_path = std::string("data/debug/") + getTestSuiteName() + "/" + getTestName() + "/" + to_string(ci) + "_depth_map.ply";
        phg::exportPointCloud(points, depth_map_points_path, colors, normals);
        std::cout << "Dense cloud built from depth map exported to " << depth_map_points_path << std::endl;
    }
}

TEST (test_mesh_min_cut, FromSingleDepthMap) {
    // Этот тест просто строит модель на базе одиночной карты глубины
    // Результаты см. в папке data/debug/test_mesh_min_cut/FromSingleDepthMap
    Dataset dataset = loadDataset(DATASET_DIR, DATASET_DOWNSCALE);

    std::vector<cv::Mat> depth_maps = loadDepthMaps(DATASET_DIR, DATASET_DOWNSCALE, dataset);

    for (size_t ci = 0; ci < dataset.ncameras; ++ci) {
        const cv::Mat &depth_map = depth_maps[ci];
        vector3d camera_center = phg::invP(dataset.cameras_P[ci]) * phg::homogenize(vector3d(0.0, 0.0, 0.0));

        std::vector<vector3d> points;
        std::vector<double> radiuses;
        std::vector<cv::Vec3b> colors;
        std::vector<vector3d> normals;

        phg::buildPoints(depth_maps[ci], dataset.cameras_imgs[ci], dataset.cameras_P[ci], dataset.calibration,
                         points, radiuses, normals, colors);

        std::string depth_map_points_path = std::string("data/debug/") + getTestSuiteName() + "/" + getTestName() + "/" + to_string(ci) + "_depth_map.ply";
        phg::exportPointCloud(points, depth_map_points_path, colors, normals);
        std::cout << "Dense cloud built from depth map exported to " << depth_map_points_path << std::endl;

        MinCutModelBuilder model_builder;
        model_builder.appendToTriangulation(ci, camera_center, points, radiuses, colors, normals);
        model_builder.printAppendStats();

        std::vector<cv::Vec3i> mesh_faces;
        std::vector<vector3d> mesh_vertices;
        std::vector<cv::Vec3b> mesh_vertices_color;
        model_builder.buildMesh(mesh_faces, mesh_vertices, mesh_vertices_color);

        std::string model_path = std::string("data/debug/") + getTestSuiteName() + "/" + getTestName() + "/" + to_string(ci) + "_model.ply";
        phg::exportMesh(model_path, mesh_vertices, mesh_faces, mesh_vertices_color);
        std::cout << "Model exported to " << model_path << std::endl;
    }
}

TEST (test_mesh_min_cut, FromAllDepthMaps) {
    // TODO 1002: запустите test_mesh_min_cut/FromAllDepthMaps и убедитесь что полигональные модели построенные по картам глубины выглядят правдоподобно
    // Этот тест строит модель на базе всех карт глубин (с учетом опционального ограничения CAMERAS_LIMIT)
    // Результаты см. в папке data/debug/test_mesh_min_cut/FromAllDepthMaps
    Dataset dataset = loadDataset(DATASET_DIR, DATASET_DOWNSCALE);

    std::vector<cv::Mat> depth_maps = loadDepthMaps(DATASET_DIR, DATASET_DOWNSCALE, dataset);

#ifdef CAMERAS_LIMIT
    const size_t cameras_limit = std::min((size_t) CAMERAS_LIMIT, dataset.ncameras);
#else
    const size_t cameras_limit = dataset.ncameras;
#endif

    MinCutModelBuilder model_builder;
    for (size_t ci = 0; ci < cameras_limit; ++ci) {
        const cv::Mat &depth_map = depth_maps[ci];
        vector3d camera_center = phg::invP(dataset.cameras_P[ci]) * phg::homogenize(vector3d(0.0, 0.0, 0.0));

        std::vector<vector3d> points;
        std::vector<double> radiuses;
        std::vector<cv::Vec3b> colors;
        std::vector<vector3d> normals;

        phg::buildPoints(depth_maps[ci], dataset.cameras_imgs[ci], dataset.cameras_P[ci], dataset.calibration,
                         points, radiuses, normals, colors);

        std::string depth_map_points_path = std::string("data/debug/") + getTestSuiteName() + "/" + getTestName() + "/" + to_string(ci) + "_depth_map.ply";
        phg::exportPointCloud(points, depth_map_points_path, colors, normals);
        std::cout << "Dense cloud built from depth map exported to " << depth_map_points_path << std::endl;

        model_builder.appendToTriangulation(ci, camera_center, points, radiuses, colors, normals);
        std::cout << "Camera #" << (ci + 1) << "/" << cameras_limit << ": ";
        model_builder.printAppendStats();
    }

    std::vector<cv::Vec3i> mesh_faces;
    std::vector<vector3d> mesh_vertices;
    std::vector<cv::Vec3b> mesh_vertices_color;
    model_builder.buildMesh(mesh_faces, mesh_vertices, mesh_vertices_color);

    std::string model_path = std::string("data/debug/") + getTestSuiteName() + "/" + getTestName() + "/full_model" + to_string(cameras_limit) + ".ply";
    phg::exportMesh(model_path, mesh_vertices, mesh_faces, mesh_vertices_color);
    std::cout << "Model exported to " << model_path << std::endl;
}
