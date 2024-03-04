#pragma once

#include <opencv2/core.hpp>

namespace phg {

    cv::Matx33d findFMatrix(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, double threshold_px = 3);
    cv::Matx33d findFMatrixCV(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, double threshold_px = 3);

    cv::Matx33d composeFMatrix(const cv::Matx34d &P0, const cv::Matx34d &P1);

}