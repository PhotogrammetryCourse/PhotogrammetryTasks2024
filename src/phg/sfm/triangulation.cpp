#include "triangulation.h"

#include "defines.h"

#include <iostream>
#include <iomanip>

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
    // throw std::runtime_error("not implemented yet");
    double x0 = ms[0][0];
    double y0 = ms[0][1];

    double x1 = ms[1][0];
    double y1 = ms[1][1];

    auto p0 = Ps[0].row(0);
    auto p1 = Ps[0].row(1);
    auto p2 = Ps[0].row(2);

    auto P0 = Ps[1].row(0);
    auto P1 = Ps[1].row(1);
    auto P2 = Ps[1].row(2);

    Eigen::Matrix4d A;
    Eigen::MatrixXd B0(2, 1), C0(1, 4), D0(2, 4), B1(2, 1), C1(1, 4), D1(2, 4);
    B0.col(0) << x0, y0;
    C0.row(0) << p2(0, 0), p2(0, 1), p2(0, 2), p2(0, 3);
    B1.col(0) << x1, y1;
    C1.row(0) << P2(0, 0), P2(0, 1), P2(0, 2), P2(0, 3);

    D0.row(0) << p0(0, 0), p0(0, 1), p0(0, 2), p0(0, 3);
    D0.row(1) << p1(0, 0), p1(0, 1), p1(0, 2), p1(0, 3);

    D1.row(0) << P0(0, 0), P0(0, 1), P0(0, 2), P0(0, 3);
    D1.row(1) << P1(0, 0), P1(0, 1), P1(0, 2), P1(0, 3);
    auto BCmD0 = B0 * C0 - D0, BCmD1 = B1 * C1 - D1;

    for (int i = 0; i < 8; ++i) {
        A(i / 4, i % 4) = BCmD0(i / 4, i % 4);
        A(2 + (i / 4), i % 4) = BCmD1(i / 4, i % 4);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector4d solution = svd.matrixV().col(3);
    return cv::Vec4d(solution(0), solution(1), solution(2), solution(3));
}
