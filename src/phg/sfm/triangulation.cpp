#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида 
// x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
    double x0 = ms[0][0];
    double y0 = ms[0][1];

    double x1 = ms[1][0];
    double y1 = ms[1][1];

    auto p00 = Ps[0].row(0);
    auto p01 = Ps[0].row(1);
    auto p02 = Ps[0].row(2);
    auto p03 = Ps[0].row(3);

    auto p10 = Ps[0].row(0);
    auto p11 = Ps[0].row(1);
    auto p12 = Ps[0].row(2);
    auto p13 = Ps[0].row(3);

    Eigen::Matrix4d E;
    E(0, 0) = x0 * p03(0, 0) - p01(0, 0);
    E(0, 1) = x0 * p03(0, 1) - p01(0, 1);
    E(0, 2) = x0 * p03(0, 2) - p01(0, 2);
    E(0, 3) = x0 * p03(0, 3) - p01(0, 3);

    E(1, 0) = y0 * p03(0, 0) - p02(0, 0);
    E(1, 1) = y0 * p03(0, 1) - p02(0, 1);
    E(1, 2) = y0 * p03(0, 2) - p02(0, 2);
    E(1, 3) = y0 * p03(0, 3) - p02(0, 3);

    E(2, 0) = x1 * p13(0, 0) - p11(0, 0);
    E(2, 1) = x1 * p13(0, 1) - p11(0, 1);
    E(2, 2) = x1 * p13(0, 2) - p11(0, 2);
    E(2, 3) = x1 * p13(0, 3) - p11(0, 3);

    E(3, 0) = y1 * p13(0, 0) - p12(0, 0);
    E(3, 1) = y1 * p13(0, 1) - p12(0, 1);
    E(3, 2) = y1 * p13(0, 2) - p12(0, 2);
    E(3, 3) = y1 * p13(0, 3) - p12(0, 3);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto V = svd.matrixV().row(3);
    return cv::Vec4d(V(0, 0), V(0, 1), V(0, 2), V(0, 3));
}
