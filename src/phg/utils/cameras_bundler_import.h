#pragma once

#include <vector>
#include <phg/sfm/defines.h>
#include <phg/sfm/sfm_utils.h>
#include <phg/core/calibration.h>


namespace phg {

// See abound bundler .out v0.3 format in 'Output format and scene representation' section:
// https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6
void importCameras(const std::string &path,
                 std::vector<matrix34d> &cameras,
                 phg::Calibration &sensor_calibration,
                 std::vector<vector3d> &tie_points,
                 std::vector<phg::Track> &tracks,
                 std::vector<std::vector<cv::KeyPoint>> &keypoints,
                 int downscale=1,
                 std::vector<cv::Vec3b> *tie_points_colors=nullptr);

}
