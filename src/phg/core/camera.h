#pragma once

#include <opencv2/core.hpp>

#include "calibration.h"

namespace phg {

struct Camera {

    cv::Vec3d center() const;

private:
    cv::Matx44d T; // transform: rotation + translation
    Calibration calib; // intrinsic parameters

};

}
