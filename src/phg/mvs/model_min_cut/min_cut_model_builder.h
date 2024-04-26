#pragma once

#include <vector>
#include <memory>
#include <unordered_map>

#include <phg/sfm/defines.h>


class MinCutModelBuilder {
public:
    MinCutModelBuilder();

    void appendToTriangulation(unsigned int camera_id, const vector3d &camera_center,
                               const std::vector<vector3d> &points, const std::vector<double> &radiuses,
                               const std::vector<cv::Vec3b> &colors, const std::vector<vector3d> &normals);
    void printAppendStats();

    void buildMesh(std::vector<cv::Vec3i> &mesh_faces, std::vector<vector3d> &mesh_vertices, std::vector<cv::Vec3b> &mesh_vertices_color);

protected:
    void insertBoundingBoxVertices(vector3d &bb_min, vector3d &bb_max);

    struct TriangulationProxy;
    std::shared_ptr<TriangulationProxy> proxy;

    size_t append_total_points;
    size_t append_inserted_points;
    double append_nn_search_total_t;
    double append_triangulation_insert_total_t;
    double append_total_t;

    std::unordered_map<unsigned int, vector3d> cameras_centers;
};
