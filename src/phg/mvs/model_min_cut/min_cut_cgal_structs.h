#pragma once

#include <phg/sfm/defines.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Delaunay_triangulation_3.h>


struct vertex_info_t {
    std::vector<unsigned int> camera_ids;
    cv::Vec3b                 color;
    size_t                    vertex_on_surface_id;
    double                    radius;
    int                       weight;

    vertex_info_t() : color(0, 0, 255), weight(1) // red color, BGR convention (OpenCV compatible)
    {}

    vertex_info_t(unsigned int camera_id, const cv::Vec3b &color, double radius);

    void merge(const vertex_info_t &that);
};

struct cell_info_t {
    size_t      cell_id;

    float       s_capacity;
    float       t_capacity;
    float       facets_capacities[4];
};

typedef CGAL::Exact_predicates_inexact_constructions_kernel                                      cgal_kernel_t;

typedef CGAL::Triangulation_vertex_base_with_info_3<vertex_info_t, cgal_kernel_t>                cgal_vertex_t;
typedef CGAL::Triangulation_cell_base_with_info_3  <cell_info_t,   cgal_kernel_t>                cgal_cell_t;

typedef CGAL::Triangulation_data_structure_3<cgal_vertex_t, cgal_cell_t, CGAL::Sequential_tag>   triangulation_data_t;
typedef CGAL::Delaunay_triangulation_3<cgal_kernel_t, triangulation_data_t, CGAL::Fast_location> triangulation_t;

typedef cgal_kernel_t::Point_3          cgal_point_t;
typedef cgal_kernel_t::Triangle_3       cgal_triangle_t;

typedef triangulation_t::Vertex_handle  vertex_handle_t;
typedef triangulation_t::Cell_handle    cell_handle_t;

typedef triangulation_t::Facet          cgal_facet_t;

vector3d from_cgal_point(cgal_point_t p);

cgal_point_t to_cgal_point(vector3d p);
