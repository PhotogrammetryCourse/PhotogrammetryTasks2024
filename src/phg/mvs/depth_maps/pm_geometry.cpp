#include "pm_geometry.h"

#include <libutils/rasserts.h>


namespace phg {

    vector4d homogenize(const vector3d &p)
    {
        return vector4d(p[0], p[1], p[2], 1.0);
    }
    
    double dot(const vector3d &a, const vector3d &b)
    {
        return a.dot(b);
    }
    
    float norm2(const vector3f &v)
    {
        return dot(v, v);
    }

    vector3f randomNormalObservedFromCamera(const matrix3d &RtoWorld, FastRandom &r)
    {
        vector3d local_n = r.nextPointFromSphere();

        // если нормаль смотрит по напралению оси +Z в локальной системе координат камеры
        if (local_n[2] > 0.0f) {      // (т.е. если камера эту поверхность видит с изнанки)
            local_n[2] = -local_n[2]; // то зеркалим нормаль по этой оси, исправляя ситуацию 
        }

        // переводим нормаль в глобальную систему координат
        vector3f global_n = RtoWorld * local_n;

        // проверяем что это все еще единичный вектор
        float len2 = norm2(global_n);
        rassert(len2 > 0.99f && len2 < 1.01f, 28314924192016);

        return global_n;
    }

    bool intersectWithPlane(const vector3d &plane_point, const vector3d &plane_normal, const vector3d &ray_org, const vector3d &ray_dir, vector3d &global_intersection)
    {
        // see https://stackoverflow.com/a/23976134/1549330
        double pace = dot(ray_dir, plane_normal);
        if (fabs(pace) <= 0.0001) {
            return false;
        }
        global_intersection = ray_org + ray_dir * (dot((plane_point - ray_org), plane_normal) / pace);
        return true;
    }

}