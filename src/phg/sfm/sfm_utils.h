#pragma once

#include <cstdint>
#include <vector>
#include <opencv2/core.hpp>

// template <size_t dims> вышел из чата

// Нужно создать матрицу преобразования, которая сдвинет переданное множество точек так, что центр масс перейдет в ноль, а Root Mean Square расстояние до него станет sqrt(2)
// (см. Hartley & Zisserman p.107 Why is normalization essential?)
#define makeNormalizer(dims, dimsm1) ([](const std::vector<cv::Vec##dimsm1##d>& m) -> cv::Matx##dims##dims##d { \
    cv::Matx##dims##dims##d transform = cv::Matx##dims##dims##d::all(0);                                        \
    for (int i = 0; i < dimsm1; ++i) {                                                                          \
        double mi = std::accumulate(                                                                            \
            m.begin(), m.end(), 0.0,                                                                            \
            [i](double z, const auto& v) { return z + v[i]; }) / m.size();                                      \
        double sdi = std::sqrt(                                                                                 \
            std::accumulate(                                                                                    \
                m.begin(), m.end(), 0.0,                                                                        \
                [i, mi](double z, const auto& v) { return z + std::pow(v[i] - mi, 2); }) / m.size());           \
                                                                                                                \
        transform(i, i) = std::sqrt(2) / sdi;                                                                   \
        transform(i, dimsm1) = -mi * transform(i, i);                                                           \
    }                                                                                                           \
    transform(dimsm1, dimsm1) = 1;                                                                              \
    return transform;                                                                                           \
    })

#define makePointTransformer(dims, dimsm1) ([]                                             \
    (const cv::Vec##dimsm1##d& pt, const cv::Matx##dims##dims##d& T) -> cv::Vec##dims##d { \
    cv::Vec##dims##d tmp;                                                                  \
    for (int i = 0; i < dimsm1; ++i) {                                                     \
        tmp[i] = pt[i];                                                                    \
    }                                                                                      \
    tmp[dimsm1] = 1;                                                                       \
    tmp = T * tmp;                                                                         \
    if (tmp[dimsm1] == 0) {                                                                \
        throw std::runtime_error("infinite point");                                        \
    }                                                                                      \
    return tmp / tmp[dimsm1];                                                              \
    })

namespace phg {

    void randomSample(std::vector<int> &dst, int max_id, int sample_size, uint64_t *state);
    bool epipolarTest(const cv::Vec2d& pt0, const cv::Vec2d& pt1, const cv::Matx33d& F, double t);

}
