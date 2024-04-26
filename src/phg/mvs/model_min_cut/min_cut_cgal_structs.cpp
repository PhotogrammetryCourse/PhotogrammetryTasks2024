#include "min_cut_cgal_structs.h"

#include <libutils/rasserts.h>


vector3d from_cgal_point(cgal_point_t p)
{
    return vector3d(p.x(), p.y(), p.z());
}

cgal_point_t to_cgal_point(vector3d p)
{
    return cgal_point_t(p[0], p[1], p[2]);
}

vertex_info_t::vertex_info_t(unsigned int camera_id, const cv::Vec3b &color) : color(color)
{
    camera_ids.push_back(camera_id);
}

void vertex_info_t::merge(const vertex_info_t &that)
{
    for (int i = 1; i < camera_ids.size(); ++i) {
        rassert(camera_ids[i - 1] < camera_ids[i], 23781274121024);
    }
    for (int i = 1; i < that.camera_ids.size(); ++i) {
        rassert(that.camera_ids[i - 1] < that.camera_ids[i], 23781274121021);
    }

    for (int i = 0; i < that.camera_ids.size(); ++i) {
        unsigned int ci = that.camera_ids[i];
        if (std::find(camera_ids.begin(), camera_ids.end(), ci) == camera_ids.end()) {
            camera_ids.push_back(ci);
        }
    }

    std::sort(camera_ids.begin(), camera_ids.end());
    for (int i = 1; i < camera_ids.size(); ++i) {
        rassert(camera_ids[i - 1] < camera_ids[i], 23781274121024);
    }
}
