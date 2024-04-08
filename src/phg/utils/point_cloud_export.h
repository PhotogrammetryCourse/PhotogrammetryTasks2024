#pragma once

#include <opencv2/core.hpp>

namespace phg {

    void exportPointCloud(const std::vector<cv::Vec3d> &point_cloud, const std::string &path, const std::vector<cv::Vec3b> &point_cloud_colors_bgr = {}, const std::vector<cv::Vec3d> &point_cloud_normal = {});

}
