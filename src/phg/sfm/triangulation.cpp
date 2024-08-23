#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count) {
    // составление однородной системы + SVD
    // без подвохов

    Eigen::MatrixXd A(2 * count, 4);
    for (int i = 0; i < count; i++) {
        double x = ms[i][0] / ms[i][2];
        double y = ms[i][1] / ms[i][2];

        auto row0 = x * Ps[i].row(2) - Ps[i].row(0);
        A.row(2 * i) << row0(0), row0(1), row0(2), row0(3);

        auto row1 = y * Ps[i].row(2) - Ps[i].row(1);
        A.row(2 * i + 1) << row1(0), row1(1), row1(2), row1(3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd null_space = svd.matrixV().col(3);

    return cv::Vec4d(null_space[0], null_space[1], null_space[2], null_space[3]);
}
