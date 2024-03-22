#include "resection.h"

#include <Eigen/SVD>
#include <iostream>
#include "sfm_utils.h"
#include "defines.h"

namespace {

    cv::Matx33d getNormalizeTransform2(const std::vector<cv::Vec2d> &m)
    {
        cv::Vec2d mean = cv::Vec2d(0, 0);
        cv::Vec2d std = cv::Vec2d(0, 0);

        for (const auto &point : m) {
            mean[0] += point[0];
            mean[1] += point[1];
        }
        mean[0] /= m.size();
        mean[1] /= m.size();

        for (const auto &point : m) {
            std[0] += std::pow(point[0] - mean[0], 2);
            std[1] += std::pow(point[1] - mean[1], 2);
        }
        std[0] = std::sqrt(std[0] / m.size());
        std[1] = std::sqrt(std[1] / m.size());
        
        double scale = std::sqrt(2);
        cv::Matx33d normalize_transform;
        normalize_transform(0, 0) = scale / std[0];
        normalize_transform(0, 1) = 0;
        normalize_transform(0, 2) = -mean[0] * scale / std[0];

        normalize_transform(1, 0) = 0;
        normalize_transform(1, 1) = scale / std[1];
        normalize_transform(1, 2) = -mean[1] * scale / std[1];

        normalize_transform(2, 0) = 0;
        normalize_transform(2, 1) = 0;
        normalize_transform(2, 2) = 1;

        return normalize_transform;
    }

    cv::Vec3d transformPoint2(const cv::Vec2d &pt, const cv::Matx33d &T)
    {
        cv::Vec3d tmp = T * cv::Vec3d(pt[0], pt[1], 1.0);

        if (tmp[2] == 0) {
            throw std::runtime_error("infinite point");
        }

        return cv::Vec3d(tmp[0] / tmp[2], tmp[1] / tmp[2], 1.0);
    }

    cv::Matx44d getNormalizeTransform3(const std::vector<cv::Vec3d> &m)
    {
        cv::Vec3d mean = cv::Vec3d(0, 0, 0);
        cv::Vec3d std = cv::Vec3d(0, 0, 0);

        for (const auto &point : m) {
            mean[0] += point[0];
            mean[1] += point[1];
            mean[2] += point[2];
        }
        mean[0] /= m.size();
        mean[1] /= m.size();
        mean[2] /= m.size();

        for (const auto &point : m) {
            std[0] += std::pow(point[0] - mean[0], 2);
            std[1] += std::pow(point[1] - mean[1], 2);
            std[2] += std::pow(point[2] - mean[2], 2);
        }
        std[0] = std::sqrt(std[0] / m.size());
        std[1] = std::sqrt(std[1] / m.size());
        std[2] = std::sqrt(std[2] / m.size());
        
        double scale = std::sqrt(3);
        cv::Matx44d normalize_transform;
        normalize_transform(0, 0) = scale / std[0];
        normalize_transform(0, 1) = 0;
        normalize_transform(0, 2) = 0;
        normalize_transform(0, 3) = -mean[0] * scale / std[0];

        normalize_transform(1, 0) = 0;
        normalize_transform(1, 1) = scale / std[1];
        normalize_transform(1, 2) = 0;
        normalize_transform(1, 3) = -mean[1] * scale / std[1];

        normalize_transform(2, 0) = 0;
        normalize_transform(2, 1) = 0;
        normalize_transform(2, 2) = scale / std[2];
        normalize_transform(2, 3) = -mean[2] * scale / std[2];

        normalize_transform(3, 0) = 0;
        normalize_transform(3, 1) = 0;
        normalize_transform(3, 2) = 0;
        normalize_transform(3, 3) = 1;

        return normalize_transform;
    }

    cv::Vec3d transformPoint3(const cv::Vec3d &pt, const cv::Matx44d &T)
    {
        cv::Vec4d tmp = T * cv::Vec4d(pt[0], pt[1], pt[2], 1.0);

        if (tmp[3] == 0) {
            throw std::runtime_error("infinite point");
        }

        return cv::Vec3d(tmp[0] / tmp[3], tmp[1] / tmp[3], tmp[2] / tmp[3]);
    }

    // Сделать из первого минора 3х3 матрицу вращения, скомпенсировать масштаб у компоненты сдвига
    matrix34d canonicalizeP(const matrix34d &P)
    {
        matrix3d RR = P.get_minor<3, 3>(0, 0);
        vector3d tt;
        tt[0] = P(0, 3);
        tt[1] = P(1, 3);
        tt[2] = P(2, 3);

        if (cv::determinant(RR) < 0) {
            RR *= -1;
            tt *= -1;
        }

        double sc = 0;
        for (int i = 0; i < 9; i++) {
            sc += RR.val[i] * RR.val[i];
        }
        sc = std::sqrt(3 / sc);

        Eigen::MatrixXd RRe;
        copy(RR, RRe);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(RRe, Eigen::ComputeFullU | Eigen::ComputeFullV);
        RRe = svd.matrixU() * svd.matrixV().transpose();
        copy(RRe, RR);

        tt *= sc;

        matrix34d result;
        for (int i = 0; i < 9; ++i) {
            result(i / 3, i % 3) = RR(i / 3, i % 3);
        }
        result(0, 3) = tt(0);
        result(1, 3) = tt(1);
        result(2, 3) = tt(2);

        return result;
    }

    // (см. Hartley & Zisserman p.178)
    cv::Matx34d estimateCameraMatrixDLT(const cv::Vec3d *Xs, const cv::Vec3d *xs, int count)
    {
        using mat = Eigen::MatrixXd;
        using vec = Eigen::VectorXd;

        mat A(11, 12);

        for (int i = 0; i < count; ++i) {

            double x = xs[i][0];
            double y = xs[i][1];
            double w = xs[i][2];

            double X = Xs[i][0];
            double Y = Xs[i][1];
            double Z = Xs[i][2];
            double W = 1.0;

            A(2 * i, 0) = 0;
            A(2 * i, 1) = 0;
            A(2 * i, 2) = 0;
            A(2 * i, 3) = 0;

            A(2 * i, 4) = -w * X;
            A(2 * i, 5) = -w * Y;
            A(2 * i, 6) = -w * Z;
            A(2 * i, 7) = -w * W;

            A(2 * i, 8) = y * X;
            A(2 * i, 9) = y * Y;
            A(2 * i, 10) = y * Z;
            A(2 * i, 11) = y * W;

            if (i != count - 1) {
                A(2 * i + 1, 0) = w * X;
                A(2 * i + 1, 1) = w * Y;
                A(2 * i + 1, 2) = w * Z;
                A(2 * i + 1, 3) = w * W;

                A(2 * i + 1, 4) = 0;
                A(2 * i + 1, 5) = 0;
                A(2 * i + 1, 6) = 0;
                A(2 * i + 1, 7) = 0;

                A(2 * i + 1, 8) = -x * X;
                A(2 * i + 1, 9) = -x * Y;
                A(2 * i + 1, 10) = -x * Z;
                A(2 * i + 1, 11) = -x * W;
            }
        }

        // std::cout << "A:\n" << A << std::endl;

        Eigen::JacobiSVD svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd V = svd.matrixV();
        int cols_number = static_cast<int>(V.cols());
        Eigen::VectorXd solution = V.col(cols_number - 1);
        
        matrix34d result;
        result(0, 0) = solution[0]; result(0, 1) = solution[1]; result(0, 2) = solution[2]; result(0, 3) = solution[3];
        result(1, 0) = solution[4]; result(1, 1) = solution[5]; result(1, 2) = solution[6]; result(1, 3) = solution[7];
        result(2, 0) = solution[8]; result(2, 1) = solution[9]; result(2, 2) = solution[10]; result(2, 3) = solution[11];

        return canonicalizeP(result);
    }


    // По трехмерным точкам и их проекциям на изображении определяем положение камеры
    cv::Matx34d estimateCameraMatrixRANSAC(const phg::Calibration &calib, const std::vector<cv::Vec3d> &X, const std::vector<cv::Vec2d> &x)
    {
        // throw std::runtime_error("not implemented yet");
        if (X.size() != x.size()) {
            throw std::runtime_error("estimateCameraMatrixRANSAC: X.size() != x.size()");
        }

        const int n_points = X.size();

        // https://en.wikipedia.org/wiki/Random_sample_consensus#Parameters
        // будет отличаться от случая с гомографией
        const float inliers_ratio = 0.25;
        const float success_probability = 0.999;
        const int n_samples = 6;
        const int n_trials = static_cast<int>(ceil(log(1 - success_probability) / log(1 - pow(inliers_ratio, n_samples))));

        const double threshold_px = 3;

        uint64_t seed = 1;

        int best_support = 0;
        cv::Matx34d best_P;

        // std::vector<cv::Vec2d> x_uncalibrated;
        // for (cv::Vec2d x_el : x) {
        //     cv::Vec3d tmp = calib.K().inv() * cv::Vec3d(x_el[0], x_el[1], 1);
        //     x_uncalibrated.push_back(cv::Vec2d(tmp[0] / tmp[2], tmp[1] / tmp[2]));
        // }

        // cv::Matx44d U = getNormalizeTransform3(X);
        // cv::Matx33d T = getNormalizeTransform2(x_uncalibrated);

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            phg::randomSample(sample, n_points, n_samples, &seed);

            cv::Vec3d ms0[n_samples];
            cv::Vec3d ms1[n_samples];
            for (int i = 0; i < n_samples; ++i) {
                // ms0[i] = transformPoint3(X[sample[i]], U);
                ms0[i] = X[sample[i]];
                // ms1[i] = transformPoint2(x_uncalibrated[sample[i]], T);
                ms1[i] = calib.K().inv() * cv::Vec3d(x[sample[i]][0], x[sample[i]][1], 1);
            }

            cv::Matx34d P_estimated = estimateCameraMatrixDLT(ms0, ms1, n_samples);
            // cv::Matx34d P = T.inv() * P_estimated * U;
            cv::Matx34d P = P_estimated;

            int support = 0;
            for (int i = 0; i < n_points; ++i) {
                // TODO спроецировать 3Д точку в пиксель с использованием P и calib
                cv::Vec3d pX = calib.K() * P * cv::Vec4d(X[i][0], X[i][1], X[i][2], 1);
                cv::Vec2d px = cv::Vec2d(pX[0] / pX[2], pX[1] / pX[2]);

                if (cv::norm(px - x[i]) < threshold_px) {
                    ++support;
                }
            }

            if (support > best_support) {
                best_support = support;
                best_P = P;

                std::cout << "estimateCameraMatrixRANSAC : support: " << best_support << "/" << n_points << std::endl;

                if (best_support == n_points) {
                    break;
                }
            }
        }

        std::cout << "estimateCameraMatrixRANSAC : best support: " << best_support << "/" << n_points << std::endl;

        if (best_support == 0) {
            throw std::runtime_error("estimateCameraMatrixRANSAC : failed to estimate camera matrix");
        }

        return best_P;
    }


}

cv::Matx34d phg::findCameraMatrix(const Calibration &calib, const std::vector <cv::Vec3d> &X, const std::vector <cv::Vec2d> &x) {
    return estimateCameraMatrixRANSAC(calib, X, x);
}
