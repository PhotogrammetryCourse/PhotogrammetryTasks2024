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
    double x = ms[0][0];
    double y = ms[0][1];

    double x_ = ms[1][0];
    double y_ = ms[1][1];

    auto p1 = Ps[0].row(0);
    auto p2 = Ps[0].row(1);
    auto p3 = Ps[0].row(2);

    auto p_1 = Ps[1].row(0);
    auto p_2 = Ps[1].row(1);
    auto p_3 = Ps[1].row(2);

    Eigen::Matrix4d A;
    A(0, 0) = x * p3(0, 0) - p1(0, 0);
    A(0, 1) = x * p3(0, 1) - p1(0, 1);
    A(0, 2) = x * p3(0, 2) - p1(0, 2);
    A(0, 3) = x * p3(0, 3) - p1(0, 3);

    A(1, 0) = y * p3(0, 0) - p2(0, 0);
    A(1, 1) = y * p3(0, 1) - p2(0, 1);
    A(1, 2) = y * p3(0, 2) - p2(0, 2);
    A(1, 3) = y * p3(0, 3) - p2(0, 3);

    A(2, 0) = x_ * p_3(0, 0) - p_1(0, 0);
    A(2, 1) = x_ * p_3(0, 1) - p_1(0, 1);
    A(2, 2) = x_ * p_3(0, 2) - p_1(0, 2);
    A(2, 3) = x_ * p_3(0, 3) - p_1(0, 3);

    A(3, 0) = y_ * p_3(0, 0) - p_2(0, 0);
    A(3, 1) = y_ * p_3(0, 1) - p_2(0, 1);
    A(3, 2) = y_ * p_3(0, 2) - p_2(0, 2);
    A(3, 3) = y_ * p_3(0, 3) - p_2(0, 3);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto V = svd.matrixV();
    Eigen::Vector4d sol = V.col(3);
    return cv::Vec4d(sol(0), sol(1), sol(2), sol(3));
}
