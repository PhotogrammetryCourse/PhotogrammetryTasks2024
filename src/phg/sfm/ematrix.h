#pragma once

#include <opencv2/core.hpp>
#include <phg/core/calibration.h>

namespace phg {

    cv::Matx33d fmatrix2ematrix(const cv::Matx33d &F, const Calibration &calib0, const Calibration &calib1);

    void decomposeEMatrix(cv::Matx34d &P0, cv::Matx34d &P1, const cv::Matx33d &E, const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, const Calibration &calib0, const Calibration &calib1, bool verbose=true);

    void decomposeUndistortedPMatrix(cv::Matx33d &R, cv::Vec3d &O, const cv::Matx34d &P);
    cv::Matx34d composeCameraMatrixRO(const cv::Matx33d &R, const cv::Vec3d &O);

    cv::Matx33d composeEMatrixRT(const cv::Matx33d &R, const cv::Vec3d &T);
}