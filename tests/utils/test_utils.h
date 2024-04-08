#pragma once

#include <opencv2/core.hpp>

#include <phg/sfm/defines.h>
#include <phg/sfm/sfm_utils.h>
#include <phg/core/calibration.h>


#ifndef M_PI
#define M_PI    3.14159265358979323846
#define M_PI_2  1.57079632679489661923  // pi/2
#define M_PI_4  0.785398163397448309616 // pi/4
#endif

cv::Mat concatenateImagesLeftRight(const cv::Mat &img0, const cv::Mat &img1);

std::string getTestName();

std::string getTestSuiteName();

void drawMatches(const cv::Mat &img1,
                 const cv::Mat &img2,
                 const std::vector<cv::KeyPoint> &keypoints1,
                 const std::vector<cv::KeyPoint> &keypoints2,
                 const std::vector<cv::DMatch> &matches,
                 const std::string &path);

void generateTiePointsCloud(const std::vector<vector3d> &tie_points,
                            const std::vector<phg::Track> &tracks,
                            const std::vector<std::vector<cv::KeyPoint>> &keypoints,
                            const std::vector<cv::Mat> &imgs,
                            const std::vector<char> &aligned,
                            const std::vector<matrix34d> &cameras,
                            int ncameras,
                            std::vector<vector3d> &tie_points_and_cameras,
                            std::vector<cv::Vec3b> &tie_points_colors);

class Dataset {
public:
    Dataset() : ncameras(0), calibration(0, 0)
    {}

    Dataset subset(size_t from, size_t to) const;

    size_t                                  ncameras;

    std::vector<cv::Mat>                    cameras_imgs;
    std::vector<cv::Mat>                    cameras_imgs_grey;
    std::vector<std::string>                cameras_labels;
    std::vector<matrix34d>                  cameras_P; // переводит из глобальной системы координат в локальную
    std::vector<std::vector<cv::KeyPoint>>  cameras_keypoints;

    std::vector<float>                      cameras_depth_min;
    std::vector<float>                      cameras_depth_max;

    phg::Calibration                        calibration;

    std::vector<phg::Track>                 tracks;
    std::vector<vector3d>                   tie_points;
};

Dataset loadDataset(const std::string &dataset_dir_name, int dataset_downscale);