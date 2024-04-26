#pragma once

#include <opencv2/core.hpp>

namespace phg {

    void exportMesh(const std::string &path,
                    const std::vector<cv::Vec3d> &mesh_vertices, const std::vector<cv::Vec3i> &mesh_faces, const std::vector<cv::Vec3b> &mesh_vertices_color = {});

}
