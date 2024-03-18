#include "resection.h"

#include <Eigen/SVD>
#include <iostream>
#include "sfm_utils.h"
#include "defines.h"

namespace {

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

        tt *= sc;
        RR *= sc;

        Eigen::MatrixXd RRe;
        copy(RR, RRe);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(RRe, Eigen::ComputeFullU | Eigen::ComputeFullV);
        RRe = svd.matrixU() * svd.matrixV().transpose();
        copy(RRe, RR);



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

        std::vector<std::vector<double>> Av;
        for (int i = 0; i < count; ++i) {

            double x = xs[i][0];
            double y = xs[i][1];
            double w = xs[i][2];

            double X = Xs[i][0];
            double Y = Xs[i][1];
            double Z = Xs[i][2];
            double W = 1.0;
            Av.push_back({
                                 0.0, 0.0, 0.0, 0.0, -w * X, -w * Y, -w * Z, -w * W, y * X, y * Y, y * Z, y * W
                         });
            Av.push_back({
                                 w * X, w * Y, w * Z, w * W, 0.0, 0.0, 0.0, 0.0, -x * X, -x * Y, -x * Z, -x * W
                         });
        }
        mat A;
        A.resize(Av.size(), Av[0].size());
        for (int y = 0; y < Av.size(); ++y) {
            for (int x = 0; x < 12; ++x) {
                A(y, x) = Av[y][x];
            }
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto V = svd.matrixV();
        matrix34d result;
        Eigen::Matrix<double, 12, 1> check;
        for (int i = 0; i < 12; ++i) {
            result(i / 4, i % 4) = V(i, 11);
            check(i) = V(i, 11);
        }
        for (int i = 0; i < count; ++i) {
            cv::Vec4d Xi;
            Xi(3) = 1.0;
            for (int j = 0; j < 3; ++j)
                Xi(j) = Xs[i](j);
            auto px_homogen = result * Xi;
            cv::Vec2d px = cv::Matx23d::eye() * (px_homogen / px_homogen(2));
//            std::cout << i << " norm = " << cv::norm(px - xs[i]) << std::endl;
        }

//        return canonicalizeP(result);
        return result;
    }


    // По трехмерным точкам и их проекциям на изображении определяем положение камеры
    cv::Matx34d estimateCameraMatrixRANSAC(const phg::Calibration &calib, const std::vector<cv::Vec3d> &X, const std::vector<cv::Vec2d> &x)
    {
//        throw std::runtime_error("not implemented yet");
        if (X.size() != x.size()) {
            throw std::runtime_error("estimateCameraMatrixRANSAC: X.size() != x.size()");
        }

        const int n_points = X.size();

        // https://en.wikipedia.org/wiki/Random_sample_consensus#Parameters
        // будет отличаться от случая с гомографией
        const int n_trials = 1000;

        const double threshold_px = 3;

        const int n_samples = 10;
        uint64_t seed = 1;

        int best_support = 0;
        cv::Matx34d best_P;

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            phg::randomSample(sample, n_points, n_samples, &seed);

            cv::Vec3d ms0[n_samples];
            cv::Vec3d ms1[n_samples];
            for (int i = 0; i < n_samples; ++i) {
                ms0[i] = X[sample[i]];
                ms1[i] = calib.unproject(x[sample[i]]);
            }

            cv::Matx34d P = estimateCameraMatrixDLT(ms0, ms1, n_samples);
//            P = calib.K() * canonicalizeP(calib.K().inv() * P);
            int support = 0;
            for (int i = 0; i < n_points; ++i) {
                cv::Vec4d Xi;
                Xi(3) = 1.0;
                for (int j = 0; j < 3; ++j)
                    Xi(j) = X[i](j);
                auto px_homogen = calib.K() * P * Xi;
                cv::Vec2d px = cv::Matx23d::eye() * (px_homogen / px_homogen(2));
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

//        return best_P;
        return canonicalizeP(best_P);
    }


}

cv::Matx34d phg::findCameraMatrix(const Calibration &calib, const std::vector <cv::Vec3d> &X, const std::vector <cv::Vec2d> &x) {
    return estimateCameraMatrixRANSAC(calib, X, x);
}
