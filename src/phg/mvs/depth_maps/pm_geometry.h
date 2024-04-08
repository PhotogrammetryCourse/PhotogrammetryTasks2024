#pragma once

#include "pm_fast_random.h"

#include <phg/sfm/defines.h>
#include <opencv2/core/matx.hpp>


namespace phg {

    vector4d homogenize(const vector3d &p);
    
    double dot(const vector3d &a, const vector3d &b);
    
    float norm2(const vector3f &v);

    vector3f randomNormalObservedFromCamera(const matrix3d &RtoWorld, FastRandom &r);

    bool intersectWithPlane(const vector3d &plane_point, const vector3d &plane_normal, const vector3d &ray_org, const vector3d &ray_dir, vector3d &global_intersection);
    
}