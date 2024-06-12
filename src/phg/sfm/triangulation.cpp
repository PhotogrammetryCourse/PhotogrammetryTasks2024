#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов

    Eigen::MatrixXd A(2 * count, 4);
    for (int i = 0; i < count; i++)
    {
        cv::Matx t0 = ms[i][0] * Ps[i].row(2).t() - Ps[i].row(0).t();
        cv::Matx t1 = ms[i][1] * Ps[i].row(2).t() - Ps[i].row(1).t();
        A.row(2 * i) << t0(0, 0), t0(1, 0), t0(2, 0), t0(3, 0);
        A.row(2 * i + 1) << t1(0, 0), t1(1, 0), t1(2, 0), t1(3, 0);
    }
    Eigen::JacobiSVD svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    cv::Vec4d res;
    for (int i = 0; i < 4; i++)
    {
        res[i] = svd.matrixV().transpose()(3, i);
    }
    return res;
}
