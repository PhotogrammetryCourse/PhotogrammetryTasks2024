#include "test_utils.h"

#include <fstream>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <libutils/timer.h>
#include <libutils/rasserts.h>

#include <phg/sfm/ematrix.h>
#include <phg/mvs/depth_maps/pm_geometry.h>
#include <phg/mvs/depth_maps/pm_depth_maps.h>
#include <phg/utils/point_cloud_export.h>
#include <phg/utils/cameras_bundler_import.h>


cv::Mat concatenateImagesLeftRight(const cv::Mat &img0, const cv::Mat &img1) {
    // это способ гарантировать себе что предположение которое явно в этой функции есть (совпадение типов картинок)
    // однажды не нарушится (по мере изменения кода) и не приведет к непредсказуемым последствиям
    // в отличие от assert() у таких rassert есть три преимущества:
    // 1) они попадают в т.ч. в релизную сборку
    // 2) есть (псевдо)уникальный идентификатор по которому легко найти где это произошло
    //    (в отличие от просто __LINE__, т.к. даже если исходный файл угадать и легко, то нумерация строк может меняться от коммита к коммиту,
    //     а падение могло случится у пользователя на старорй версии)
    // 3) есть общая удобная точка остановки на которую легко поставить breakpoint - rasserts.cpp/debugPoint()
    rassert(img0.type() == img1.type(), 125121612363131);
    rassert(img0.channels() == img1.channels(), 136161251414);

    size_t width = img0.cols + img1.cols;
    size_t height = std::max(img0.rows, img1.rows);

    cv::Mat res(height, width, img0.type());
    img0.copyTo(res(cv::Rect(0, 0, img0.cols, img0.rows)));
    img1.copyTo(res(cv::Rect(img0.cols, 0, img1.cols, img1.rows)));

    return res;
}


std::string getTestName() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}


std::string getTestSuiteName() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
}

void drawMatches(const cv::Mat &img1,
                 const cv::Mat &img2,
                 const std::vector<cv::KeyPoint> &keypoints1,
                 const std::vector<cv::KeyPoint> &keypoints2,
                 const std::vector<cv::DMatch> &matches,
                 const std::string &path)
{
    cv::Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, matches, img_matches, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::imwrite(path, img_matches);
}

void generateTiePointsCloud(const std::vector<vector3d> &tie_points,
                            const std::vector<phg::Track> &tracks,
                            const std::vector<std::vector<cv::KeyPoint>> &keypoints,
                            const std::vector<cv::Mat> &imgs,
                            const std::vector<char> &aligned,
                            const std::vector<matrix34d> &cameras,
                            int ncameras,
                            std::vector<vector3d> &tie_points_and_cameras,
                            std::vector<cv::Vec3b> &tie_points_colors)
{
    rassert(tie_points.size() == tracks.size(), 24152151251241);

    tie_points_and_cameras.clear();
    tie_points_colors.clear();

    for (int i = 0; i < (int) tie_points.size(); ++i) {
        const phg::Track &track = tracks[i];
        if (track.disabled)
            continue;

        int img = track.img_kpt_pairs.front().first;
        int kpt = track.img_kpt_pairs.front().second;
        cv::Vec2f px = keypoints[img][kpt].pt;
        tie_points_and_cameras.push_back(tie_points[i]);
        tie_points_colors.push_back(imgs[img].at<cv::Vec3b>(px[1], px[0]));
    }

    for (int i_camera = 0; i_camera < ncameras; ++i_camera) {
        if (!aligned[i_camera]) {
            throw std::runtime_error("camera " + std::to_string(i_camera) + " is not aligned");
        }

        matrix3d R;
        vector3d O;
        phg::decomposeUndistortedPMatrix(R, O, cameras[i_camera]);

        tie_points_and_cameras.push_back(O);
        tie_points_colors.push_back(cv::Vec3b(0, 0, 255));
        tie_points_and_cameras.push_back(O + R.t() * cv::Vec3d(0, 0, 1));
        tie_points_colors.push_back(cv::Vec3b(255, 0, 0));
    }
}

Dataset Dataset::subset(size_t from, size_t to) const
{
    Dataset res = *this;

    rassert(from < to, 23481294812944);
    res.ncameras = to - from;
    rassert(ncameras >= 2, 123419481245);

    res.cameras_imgs = std::vector<cv::Mat>(cameras_imgs.begin() + from, cameras_imgs.begin() + to);
    res.cameras_imgs_grey = std::vector<cv::Mat>(cameras_imgs_grey.begin() + from, cameras_imgs_grey.begin() + to);
    res.cameras_labels = std::vector<std::string>(cameras_labels.begin() + from, cameras_labels.begin() + to);
    res.cameras_P = std::vector<matrix34d>(cameras_P.begin() + from, cameras_P.begin() + to);
    res.cameras_keypoints = std::vector<std::vector<cv::KeyPoint>>(cameras_keypoints.begin() + from, cameras_keypoints.begin() + to);

    res.cameras_depth_min = std::vector<float>(cameras_depth_min.begin() + from, cameras_depth_min.begin() + to);
    res.cameras_depth_max = std::vector<float>(cameras_depth_max.begin() + from, cameras_depth_max.begin() + to);

    res.tracks.clear();
    res.tie_points.clear();

    return res;
}

Dataset loadDataset(const std::string &dataset_dir_name, int dataset_downscale)
{
    timer t;

    Dataset dataset;

    std::string images_list_filename = std::string("data/src/datasets/") + dataset_dir_name + "/ordered_filenames.txt";
    std::ifstream in(images_list_filename);
    if (!in) {
        throw std::runtime_error("Can't read file: " + to_string(images_list_filename)); // проверьте 'Working directory' в 'Edit Configurations...' в CLion (должна быть корневая папка проекта, чтобы относительные пути к датасетам сталик орректны)
    }
    in >> dataset.ncameras;

    std::cout << "loading " << dataset.ncameras << " images..." << std::endl;
    for (size_t ci = 0; ci < dataset.ncameras; ++ci) {
        std::string img_name;
        in >> img_name;
        std::string img_path = std::string("data/src/datasets/") + dataset_dir_name + "/" + img_name;
        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION); // чтобы если камера записала в exif-tag повернута она была или нет - мы получили сырую картинку, без поворота с учетом этой информации, ведь одну и ту же камеру могли повернуть по-разному (напр. saharov32)

        if (img.empty()) {
            throw std::runtime_error("Can't read image: " + to_string(img_path));
        }

        // выполняем опциональное уменьшение картинки
        int downscale = dataset_downscale;
        while (downscale > 1) {
            cv::pyrDown(img, img);
            rassert(downscale % 2 == 0, 1249219412940115);
            downscale /= 2;
        }

        if (ci == 0) {
            dataset.calibration.width_ = img.cols;
            dataset.calibration.height_ = img.rows;
            std::cout << "resolution: " << img.cols << "x" << img.rows << std::endl;
        } else {
            rassert(dataset.calibration.width_  == img.cols, 2931924190089);
            rassert(dataset.calibration.height_ == img.rows, 2931924190090);
        }

        cv::Mat grey;
        cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);

        dataset.cameras_imgs.push_back(img);
        dataset.cameras_imgs_grey.push_back(grey);
        dataset.cameras_labels.push_back(img_name);
    }

    phg::importCameras(std::string("data/src/datasets/") + dataset_dir_name + "/cameras.out",
                       dataset.cameras_P, dataset.calibration, dataset.tie_points, dataset.tracks, dataset.cameras_keypoints, dataset_downscale);

    dataset.cameras_depth_max.resize(dataset.ncameras);
    dataset.cameras_depth_min.resize(dataset.ncameras);
    #pragma omp parallel for schedule(dynamic, 1)
    for (ptrdiff_t ci = 0; ci < dataset.ncameras; ++ci) {
        double depth_min = std::numeric_limits<double>::max();
        double depth_max = 0.0;

        for (size_t ti = 0; ti < dataset.tracks.size(); ++ti) {
            ptrdiff_t kpt = -1;
            auto img_kpt_pairs = dataset.tracks[ti].img_kpt_pairs;
            for (size_t i = 0; i < img_kpt_pairs.size(); ++i) {
                if (img_kpt_pairs[i].first == ci) {
                    kpt = img_kpt_pairs[i].second;
                }
            }
            if (kpt == -1)
                continue; // эта ключевая точка не имеет отношения к текущей камере ci

            vector3d tie_point = dataset.tie_points[ti];

            vector3d px = phg::project(tie_point, dataset.calibration, dataset.cameras_P[ci]);

            // проверяем project->unproject на идемпотентность
            // при отладке удобно у отлаживаемого цикла закомментировать #pragma omp parallel for
            // еще можно наспамить много project-unproject вызвов строчка за строчкой, чтобы при отладке не перезапускать программу
            // а просто раз за разом просматривать как проходит исполнение этих функций до понимания что пошло не так
            vector3d point_test = phg::unproject(px, dataset.calibration, phg::invP(dataset.cameras_P[ci]));

            vector3d diff = tie_point - point_test;
            double norm2 = phg::norm2(diff);
            rassert(norm2 < 0.0001, 241782412410125);

            double depth = px[2];
            rassert(depth > 0.0, 238419481290132);
            depth_min = std::min(depth_min, depth);
            depth_max = std::max(depth_max, depth);
        }

        // имеет смысл расширить диапазон глубины, т.к. ключевые точки по которым он построен - лишь ориентир
        double depth_range = depth_max - depth_min;
        depth_min = std::max(depth_min - 0.25 * depth_range, depth_min / 2.0);
        depth_max =          depth_max + 0.25 * depth_range;

        rassert(depth_min > 0.0, 2314512515210146);
        rassert(depth_min < depth_max, 23198129410137);

        dataset.cameras_depth_max[ci] = depth_max;
        dataset.cameras_depth_min[ci] = depth_min;
    }

    std::cout << dataset_dir_name << " dataset loaded in " << t.elapsed() << " s" << std::endl;

    std::vector<vector3d> tie_points_and_cameras;
    std::vector<cv::Vec3b> points_colors;
    generateTiePointsCloud(dataset.tie_points, dataset.tracks, dataset.cameras_keypoints, dataset.cameras_imgs, std::vector<char>(dataset.ncameras, true), dataset.cameras_P, dataset.ncameras,
                           tie_points_and_cameras, points_colors);

    std::string tie_points_filename = std::string("data/debug/") + getTestSuiteName() + "/" + getTestName() + "/tie_points_and_cameras" + to_string(dataset.ncameras) + ".ply";
    phg::exportPointCloud(tie_points_and_cameras, tie_points_filename, points_colors);
    std::cout << "tie points cloud with cameras exported to: " << tie_points_filename << std::endl;

    return dataset;
}