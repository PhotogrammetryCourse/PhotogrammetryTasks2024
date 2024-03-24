#pragma once

#include <opencv2/core.hpp>
#include <phg/core/calibration.h>

namespace phg {

    cv::Matx34d findCameraMatrix(const Calibration &calib, const std::vector<cv::Vec3d> &X, const std::vector<cv::Vec2d> &x, bool verbose=true);

}
