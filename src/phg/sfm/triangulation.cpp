#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    using mat = Eigen::MatrixXd;
    using vec = Eigen::VectorXd;

    mat A(2 * count, 4);

    for (int i = 0; i < count; ++i) {

        const matrix34d &P = Ps[i];
        const vector3d &m = ms[i];

        double x = m[0];
        double y = m[1];
        double z = m[2];

        auto p0 = P.row(0);
        auto p1 = P.row(1);
        auto p2 = P.row(2);

        auto row0 = x * p2 - z * p0;
        auto row1 = y * p2 - z * p1;

        A.row(i * 2 + 0) << row0(0), row0(1), row0(2), row0(3);
        A.row(i * 2 + 1) << row1(0), row1(1), row1(2), row1(3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd null_space = svd.matrixV().col(3);

    vector4d result;
    for (int i = 0; i < 4; ++i) {
        result(i) = null_space(i);
    }

    return result;
}
