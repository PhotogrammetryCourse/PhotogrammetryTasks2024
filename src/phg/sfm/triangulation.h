#pragma once

#include <opencv2/core.hpp>


namespace phg {

    cv::Vec4d triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count);

}
