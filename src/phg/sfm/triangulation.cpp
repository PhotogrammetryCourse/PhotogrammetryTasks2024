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

    for(int i = 0; i < count; ++i)
    {
        cv::Matx34d P = Ps[i];
        cv::Vec3d m = ms[i];
        A.row(2 * i) << m[0] * P(2, 0) - P(0, 0),
                m[0] * P(2, 1) - P(0, 1),
                m[0] * P(2, 2) - P(0, 2),
                m[0] * P(2, 3) - P(0, 3);

        A.row(2 * i + 1) << m[1] * P(2, 0) - P(1, 0),
                m[1] * P(2, 1) - P(1, 1),
                m[1] * P(2, 2) - P(1, 2),
                m[1] * P(2, 3) - P(1, 3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    cv::Vec4d X(svda.matrixV().transpose()(3, 0),
                svda.matrixV().transpose()(3, 1),
                svda.matrixV().transpose()(3, 2),
                svda.matrixV().transpose()(3, 3));
    X /= X[3];
    return X;
}
