#include "fmatrix.h"
#include "sfm_utils.h"
#include "defines.h"

#include <iostream>
#include <Eigen/SVD>
#include <opencv2/calib3d.hpp>

namespace {

    void infoF(const cv::Matx33d &Fcv)
    {
        Eigen::MatrixXd F;
        copy(Fcv, F);

        Eigen::JacobiSVD<Eigen::MatrixXd> svdf(F, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::MatrixXd U = svdf.matrixU();
        Eigen::VectorXd s = svdf.singularValues();
        Eigen::MatrixXd V = svdf.matrixV();

        std::cout << "F info:\nF:\n" << F << "\nU:\n" << U << "\ns:\n" << s << "\nV:\n" << V << std::endl;
    }

    // (см. Hartley & Zisserman p.279)
    cv::Matx33d estimateFMatrixDLT(const cv::Vec2d *m0, const cv::Vec2d *m1, int count)
    {
//        throw std::runtime_error("not implemented yet");
        int a_rows = count;
        int a_cols = 9;

        Eigen::MatrixXd A;
        A.resize(a_rows, a_cols);

        for (int i_pair = 0; i_pair < count; ++i_pair) {

            double x = m0[i_pair][0];
            double y = m0[i_pair][1];

            double xp = m1[i_pair][0];
            double yp = m1[i_pair][1];

            double coeffs[] = {
                    xp * x,
                    xp * y,
                    xp,
                    yp * x,
                    yp * y,
                    yp,
                    x,
                    y,
                    1
            };
            for (int i = 0; i < 9; ++i) {
                A(i_pair, i) = coeffs[i];
            }
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd V = svda.matrixV(); // 9x9
        Eigen::VectorXd null_space;
        null_space.resize(9, 1);
        for (int i = 0; i < 9; ++i) {
            null_space(i, 0) = V(i, 8);
        }
//        auto check = A * null_space;
//        auto check2 = V.transpose() * null_space;
//        std::cout << "A * nullspace = " << check << std::endl;
//        std::cout << "check2 = " << check2 << std::endl;

        Eigen::Matrix3d F(3, 3);
        F.row(0) << null_space[0], null_space[1], null_space[2];
        F.row(1) << null_space[3], null_space[4], null_space[5];
        F.row(2) << null_space[6], null_space[7], null_space[8];
//        std::cout  << F << std::endl;
        // Поправить F так, чтобы соблюдалось свойство фундаментальной матрицы (последнее сингулярное значение = 0)
        Eigen::JacobiSVD<Eigen::MatrixXd> svdf(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto singulars = svdf.singularValues();
        Eigen::Matrix3d corrected_S = Eigen::Matrix3d::Zero(3, 3);
        corrected_S(0, 0) = singulars[0];
        corrected_S(1, 1) = singulars[1];
        Eigen::Matrix3d fixed = svdf.matrixU() * corrected_S * svdf.matrixV().transpose();
        cv::Matx33d Fcv;
        copy(fixed, Fcv);

        return Fcv;
    }

    // Нужно создать матрицу преобразования, которая сдвинет переданное множество точек так, что центр масс перейдет в ноль, а Root Mean Square расстояние до него станет sqrt(2)
    // (см. Hartley & Zisserman p.107 Why is normalization essential?)
    cv::Matx33d getNormalizeTransform(const std::vector<cv::Vec2d> &m)
    {
        cv::Vec2d sum;
        sum[0] = 0.0;
        sum[1] = 0.0;
        for (auto const& p: m) {
            sum += p;
        }
        cv::Vec2d const avg  = sum / ((double) m.size());
//        std::cout << "avg = " << avg / (double) m.size() << std::endl;
        auto root_sum = 0.0;
        for (auto const& p: m) {
            const cv::Vec<double, 2> &difference = p - avg;
            root_sum += difference.dot(difference);
        }
//        std::cout << "Root mean = " << root_sum / (double) m.size() << std::endl;
        auto const scale_factor = std::sqrt(2.0) /  sqrt(root_sum / (double) m.size());
        cv::Matx33d F = {scale_factor, 0, -avg[0] * scale_factor, // x y w
                            0, scale_factor, -avg[1] * scale_factor,
                            0, 0, 1.0};
        return F;
    }

    cv::Vec2d transformPoint(const cv::Vec2d &pt, const cv::Matx33d &T)
    {
        cv::Vec3d tmp = T * cv::Vec3d(pt[0], pt[1], 1.0);

        if (tmp[2] == 0) {
            throw std::runtime_error("infinite point");
        }

        return cv::Vec2d(tmp[0] / tmp[2], tmp[1] / tmp[2]);
    }

    cv::Matx33d estimateFMatrixRANSAC(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, double threshold_px)
    {
        if (m0.size() != m1.size()) {
            throw std::runtime_error("estimateFMatrixRANSAC: m0.size() != m1.size()");
        }

        const int n_matches = m0.size();

        cv::Matx33d TN0 = getNormalizeTransform(m0);
        cv::Matx33d TN1 = getNormalizeTransform(m1);

        std::vector<cv::Vec2d> m0_t(n_matches);
        std::vector<cv::Vec2d> m1_t(n_matches);
        for (int i = 0; i < n_matches; ++i) {
            m0_t[i] = transformPoint(m0[i], TN0);
            m1_t[i] = transformPoint(m1[i], TN1);
        }

        {
//             Проверьте лог: при повторной нормализации должно найтись почти единичное преобразование

//            auto t1 = getNormalizeTransform(m0_t);
//            auto t2 = getNormalizeTransform(m1_t);
//            std::cout << "Loggin t1 and t2L\n";
//            infoF(t1);
//            infoF(t2);
        }
        // https://en.wikipedia.org/wiki/Random_sample_consensus#Parameters
        // будет отличаться от случая с гомографией
        const int n_trials = 100000;

        const int n_samples = std::min(n_matches, 9);
        uint64_t seed = 1;

        int best_support = 0;
        cv::Matx33d best_F;

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            phg::randomSample(sample, n_matches, n_samples, &seed);

            cv::Vec2d ms0[n_samples];
            cv::Vec2d ms1[n_samples];
            for (int i = 0; i < n_samples; ++i) {
                ms0[i] = m0_t[sample[i]];
                ms1[i] = m1_t[sample[i]];
            }

            cv::Matx33d F = estimateFMatrixDLT(ms0, ms1, n_samples);
//            cv::Matx33d F_old = F;

            // denormalize
            F = TN1.t() * F * TN0;

//            int fake_support = 0;
//            for (int i = 0; i < n_matches; ++i) {
//                if (phg::epipolarTest(ms0[i], ms1[i], F_old, threshold_px) && phg::epipolarTest(ms1[i], ms0[i], F_old, threshold_px))
//                {
//                    ++fake_support;
//                }
//            }
//            std::cout << "fake support = " << fake_support << std::endl;

            int support = 0;
            for (int i = 0; i < n_matches; ++i) {
                if (phg::epipolarTest(m0[i], m1[i], F, threshold_px)
//                && phg::epipolarTest(m1[i], m0[i], F, threshold_px)
                )
                {
                    ++support;
                }
            }

            if (support > best_support) {
                best_support = support;
                best_F = F;

                std::cout << "estimateFMatrixRANSAC : support: " << best_support << "/" << n_matches << std::endl;
//                infoF(F);

                if (best_support == n_matches) {
                    break;
                }
            }
        }

        std::cout << "estimateFMatrixRANSAC : best support: " << best_support << "/" << n_matches << std::endl;

        if (best_support == 0) {
            throw std::runtime_error("estimateFMatrixRANSAC : failed to estimate fundamental matrix");
        }

        return best_F;
    }

}

cv::Matx33d phg::findFMatrix(const std::vector <cv::Vec2d> &m0, const std::vector <cv::Vec2d> &m1, double threshold_px) {
    return estimateFMatrixRANSAC(m0, m1, threshold_px);
}

cv::Matx33d phg::findFMatrixCV(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, double threshold_px) {
    return cv::findFundamentalMat(m0, m1, cv::FM_RANSAC, threshold_px);
}

cv::Matx33d phg::composeFMatrix(const cv::Matx34d &P0, const cv::Matx34d &P1)
{
    // compute fundamental matrix from general cameras
    // Hartley & Zisserman (17.3 - p412)
    
    cv::Matx33d F;

#define det4(a, b, c, d) \
      ((a)(0) * (b)(1) - (a)(1) * (b)(0)) * ((c)(2) * (d)(3) - (c)(3) * (d)(2)) - \
      ((a)(0) * (b)(2) - (a)(2) * (b)(0)) * ((c)(1) * (d)(3) - (c)(3) * (d)(1)) + \
      ((a)(0) * (b)(3) - (a)(3) * (b)(0)) * ((c)(1) * (d)(2) - (c)(2) * (d)(1)) + \
      ((a)(1) * (b)(2) - (a)(2) * (b)(1)) * ((c)(0) * (d)(3) - (c)(3) * (d)(0)) - \
      ((a)(1) * (b)(3) - (a)(3) * (b)(1)) * ((c)(0) * (d)(2) - (c)(2) * (d)(0)) + \
      ((a)(2) * (b)(3) - (a)(3) * (b)(2)) * ((c)(0) * (d)(1) - (c)(1) * (d)(0))

    int i, j;
    for (j = 0; j < 3; j++)
        for (i = 0; i < 3; i++) {
            // here the sign is encoded in the order of lines ~ai
            const auto a1 = P0.row((i + 1) % 3);
            const auto a2 = P0.row((i + 2) % 3);
            const auto b1 = P1.row((j + 1) % 3);
            const auto b2 = P1.row((j + 2) % 3);

            F(j, i) = det4(a1, a2, b1, b2);
        }

#undef det4
    
    return F;
}
