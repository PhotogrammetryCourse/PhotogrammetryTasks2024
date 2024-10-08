#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count) {
    Eigen::MatrixX4d A(2 * count, 4);
    for (int i = 0; i < count; ++i) {
        auto x = ms[i][0] / ms[i][2];
        auto r2i = x * Ps[i].row(2) - Ps[i].row(0);
        A.row(2 * i) << r2i(0), r2i(1), r2i(2), r2i(3);
        auto y = ms[i][1] / ms[i][2];
        auto r2i1 = y * Ps[i].row(2) - Ps[i].row(1);
        A.row(2 * i + 1) << r2i1(0), r2i1(1), r2i1(2), r2i1(3);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::MatrixXd V = svd.matrixV();
    return cv::Vec4d(V(0, 3), V(1, 3), V(2, 3), V(3, 3));
}
