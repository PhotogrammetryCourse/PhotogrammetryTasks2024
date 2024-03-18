#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>
#include <iostream>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    Eigen::MatrixX4d A;
    A.resize(2 * count, 4);
    for (int i = 0; i < count; ++i) {
        auto p1 = Ps[i].row(0);
        auto p2 = Ps[i].row(1);
        auto p3 = Ps[i].row(2);
        auto x = ms[i][0];
        auto y = ms[i][1];
        auto w = ms[i][2];
        auto row1 = p3 * x - p1 * w;
        auto row2 =  p3 * y - p2 * w;
        for (int x_iter = 0; x_iter < 4; ++x_iter) {
            A(2 * i, x_iter) = row1(0, x_iter);
            A(2 * i + 1, x_iter) = row2(0, x_iter);
        }
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    auto V = svd.matrixV();
    cv::Vec4d result;
    for (int i = 0; i < 4; ++i) {
        result[i] = V(i, 3);
    }
//    std::cout << "First matrix projection: " << Ps[0] * result << std::endl;
//    std::cout << "Second matrix projection: " << Ps[1] * result << std::endl;
    return result;
}
