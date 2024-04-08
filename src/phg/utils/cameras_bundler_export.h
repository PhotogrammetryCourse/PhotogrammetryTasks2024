#pragma once

#include <vector>
#include <phg/sfm/defines.h>
#include <phg/sfm/sfm_utils.h>
#include <phg/core/calibration.h>


namespace phg {

// See abound bundler .out v0.3 format in 'Output format and scene representation' section:
// https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6
void exportCameras(const std::string &path,
                 const std::vector<matrix34d> &cameras,
                 size_t ncameras,
                 const phg::Calibration &sensor_calibration,
                 const std::vector<vector3d> &tie_points,
                 const std::vector<phg::Track> &tracks,
                 const std::vector<std::vector<cv::KeyPoint>> &keypoints,
                 int downscale=1,
                 const std::vector<cv::Vec3b> *tie_points_colors=nullptr);

}
