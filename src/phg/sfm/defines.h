#pragma once

#include <opencv2/core.hpp>

typedef cv::Matx22d matrix2d;
typedef cv::Vec2d vector2d;
typedef cv::Matx33d matrix3d;
typedef cv::Matx34d matrix34d;
typedef cv::Vec3d vector3d;
typedef cv::Vec3f vector3f;
typedef cv::Matx44d matrix4d;
typedef cv::Vec4d vector4d;

inline matrix3d skew(const vector3d &m)
{
    matrix3d result;

    double x = m[0];
    double y = m[1];
    double z = m[2];

    // 0, -z, y
    result(0, 0) = 0;
    result(0, 1) = -z;
    result(0, 2) = y;

    // z, 0, -x
    result(1, 0) = z;
    result(1, 1) = 0;
    result(1, 2) = -x;

    // -y, x, 0
    result(2, 0) = -y;
    result(2, 1) = x;
    result(2, 2) = 0;

    return result;
}

inline matrix34d make34(const matrix3d &R, const vector3d &O)
{
    matrix34d result;
    for (int i = 0; i < 9; ++i) {
        result(i / 3, i % 3) = R(i / 3, i % 3);
    }
    for (int i = 0; i < 3; ++i) {
        result(i, 3) = O(i);
    }
    return result;
}

template <typename EIGEN_TYPE>
inline void copy(const matrix3d &Fcv, EIGEN_TYPE &F)
{
    F = EIGEN_TYPE(3, 3);

    F(0, 0) = Fcv(0, 0); F(0, 1) = Fcv(0, 1); F(0, 2) = Fcv(0, 2);
    F(1, 0) = Fcv(1, 0); F(1, 1) = Fcv(1, 1); F(1, 2) = Fcv(1, 2);
    F(2, 0) = Fcv(2, 0); F(2, 1) = Fcv(2, 1); F(2, 2) = Fcv(2, 2);
}

template <typename EIGEN_TYPE>
inline void copy(const EIGEN_TYPE &F, matrix3d &Fcv)
{
    Fcv(0, 0) = F(0, 0); Fcv(0, 1) = F(0, 1); Fcv(0, 2) = F(0, 2);
    Fcv(1, 0) = F(1, 0); Fcv(1, 1) = F(1, 1); Fcv(1, 2) = F(1, 2);
    Fcv(2, 0) = F(2, 0); Fcv(2, 1) = F(2, 1); Fcv(2, 2) = F(2, 2);
}
