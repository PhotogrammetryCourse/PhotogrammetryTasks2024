#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <libutils/timer.h>
#include <phg/matching/gms_matcher.h>
#include <phg/sfm/fmatrix.h>
#include <phg/sfm/ematrix.h>
#include <phg/sfm/sfm_utils.h>
#include <phg/sfm/defines.h>
#include <Eigen/SVD>
#include <phg/sfm/triangulation.h>
#include <phg/sfm/resection.h>
#include <phg/utils/point_cloud_export.h>

#include "utils/test_utils.h"


#define ENABLE_MY_SFM 1

namespace {

    void filterMatchesF(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> keypoints_query,
                        const std::vector<cv::KeyPoint> keypoints_train, const cv::Matx33d &F, std::vector<cv::DMatch> &result, double threshold_px)
    {
        result.clear();

        for (const cv::DMatch &match : matches) {
            cv::Vec2f pt1 = keypoints_query[match.queryIdx].pt;
            cv::Vec2f pt2 = keypoints_train[match.trainIdx].pt;

            if (phg::epipolarTest(pt1, pt2, F, threshold_px)) {
                result.push_back(match);
            }
        }
    }

    // Fundamental matrix has to be of rank 2. See Hartley & Zisserman, p.243
    bool checkFmatrixSpectralProperty(const matrix3d &Fcv)
    {
        Eigen::MatrixXd F;
        copy(Fcv, F);

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd s = svd.singularValues();

        std::cout << "checkFmatrixSpectralProperty: s: " << s.transpose() << std::endl;

        double thresh = 1e10;
        return s[0] > thresh * s[2] && s[1] > thresh * s[2];
    }

    // Essential matrix has to be of rank 2, and two non-zero singular values have to be equal. See Hartley & Zisserman, p.257
    bool checkEmatrixSpectralProperty(const matrix3d &Fcv)
    {
        Eigen::MatrixXd F;
        copy(Fcv, F);

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd s = svd.singularValues();

        std::cout << "checkEmatrixSpectralProperty: s: " << s.transpose() << std::endl;

        double thresh = 1e10;

        bool rank2 = s[0] > thresh * s[2] && s[1] > thresh * s[2];
        bool equal = (s[0] < (1.0 + thresh) * s[1]) && (s[1] < (1.0 + thresh) * s[0]);

        return rank2 && equal;
    }

    template <typename MAT>
    double matRMS(const MAT &a, const MAT &b)
    {
        MAT d = (a - b);
        d = d.mul(d);
        double rms = std::sqrt(cv::sum(d)[0] / (a.cols * a.rows));
        return rms;
    }

    vector3d relativeOrientationAngles(const matrix3d &R0, const vector3d &O0, const matrix3d &R1, const vector3d &O1)
    {
        vector3d a = R0 * vector3d{0, 0, 1};
        vector3d b = O0 - O1;
        vector3d c = R1 * vector3d{0, 0, 1};

        double norma = cv::norm(a);
        double normb = cv::norm(b);
        double normc = cv::norm(c);

        if (norma == 0 || normb == 0 || normc == 0) {
            throw std::runtime_error("norma == 0 || normb == 0 || normc == 0");
        }

        a /= norma;
        b /= normb;
        c /= normc;

        vector3d cos_vals;

        cos_vals[0] = a.dot(c);
        cos_vals[1] = a.dot(b);
        cos_vals[2] = b.dot(c);

        return cos_vals;
    }

}

#define TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps) \
EXPECT_FALSE(phg::epipolarTest(pt0, pt1, F, std::max(0.0, t - eps))); \
EXPECT_TRUE(phg::epipolarTest(pt0, pt1, F, t + eps));

TEST (SFM, EpipolarDist) {

#if !ENABLE_MY_SFM
    return;
#endif

    const vector2d pt0 = {0, 0};
    const double eps = 1e-5;

    {
        // line: y = 0
        const double l[3] = {0, 1, 0};
        const matrix3d F = {0, 0, l[0], 0, 0, l[1], 0, 0, l[2]};

        vector2d pt1;
        double t;

        pt1 = {0, 0};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps)

        pt1 = {1000, 0};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps)

        pt1 = {0, 1000};
        t = 1000;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps)
    }

    {
        // line: y = x
        const double l[3] = {1, -1, 0};
        const matrix3d F = {0, 0, l[0], 0, 0, l[1], 0, 0, l[2]};

        vector2d pt1;
        double t;

        pt1 = {0, 0};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps)

        pt1 = {1, 1};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps)

        pt1 = {-1, -1};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps)

        pt1 = {-1, 1};
        t = std::sqrt(2);
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps)

        pt1 = {10, 0};
        t = 10 / std::sqrt(2);
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps)
    }

    {
        // line: y = x + 1
        const double l[3] = {1, -1, 1};
        const matrix3d F = {0, 0, l[0], 0, 0, l[1], 0, 0, l[2]};

        vector2d pt1;
        double t;

        pt1 = {0, 1};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps)

        pt1 = {1, 2};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps)

        pt1 = {-1, 0};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps)

        pt1 = {-1, 2};
        t = std::sqrt(2);
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps)

        pt1 = {10, 1};
        t = 10 / std::sqrt(2);
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps)
    }
}

TEST (SFM, FmatrixSimple) {

#if !ENABLE_MY_SFM
    return;
#endif

    std::vector<cv::Vec2d> pts0, pts1;
    std::srand(1);
    for (int i = 0; i < 8; ++i) {
        pts0.push_back({(double) (std::rand() % 100), (double) (std::rand() % 100)});
        pts1.push_back({(double) (std::rand() % 100), (double) (std::rand() % 100)});
    }

    matrix3d F = phg::findFMatrix(pts0, pts1);
    matrix3d Fcv = phg::findFMatrixCV(pts0, pts1);

    EXPECT_TRUE(checkFmatrixSpectralProperty(F));
    EXPECT_TRUE(checkFmatrixSpectralProperty(Fcv));
}

TEST (SFM, EmatrixSimple) {

#if !ENABLE_MY_SFM
    return;
#endif

    phg::Calibration calib(360, 240);
    std::cout << "EmatrixSimple: calib: \n" << calib.K() << std::endl;

    std::vector<cv::Vec2d> pts0, pts1;
    std::srand(1);
    for (int i = 0; i < 8; ++i) {
        pts0.push_back({(double) (std::rand() % calib.width()), (double) (std::rand() % calib.height())});
        pts1.push_back({(double) (std::rand() % calib.width()), (double) (std::rand() % calib.height())});
    }

    matrix3d F = phg::findFMatrix(pts0, pts1, 10);
    matrix3d E = phg::fmatrix2ematrix(F, calib, calib);

    EXPECT_TRUE(checkEmatrixSpectralProperty(E));
}

TEST (SFM, EmatrixDecomposeSimple) {

#if !ENABLE_MY_SFM
    return;
#endif

    phg::Calibration calib(360, 240);
    std::cout << "EmatrixSimple: calib: \n" << calib.K() << std::endl;

    std::vector<cv::Vec2d> pts0, pts1;
    std::srand(1);
    for (int i = 0; i < 8; ++i) {
        pts0.push_back({(double) (std::rand() % calib.width()), (double) (std::rand() % calib.height())});
        pts1.push_back({(double) (std::rand() % calib.width()), (double) (std::rand() % calib.height())});
    }

    matrix3d F = phg::findFMatrix(pts0, pts1, 10);
    matrix3d E = phg::fmatrix2ematrix(F, calib, calib);

    matrix34d P0, P1;
    phg::decomposeEMatrix(P0, P1, E, pts0, pts1, calib, calib);

    matrix3d R;
    R = P1.get_minor<3, 3>(0, 0);
    vector3d T;
    T(0) = P1(0, 3);
    T(1) = P1(1, 3);
    T(2) = P1(2, 3);

    matrix3d E1 = phg::composeEMatrixRT(R, T);
    matrix3d E2 = phg::composeFMatrix(P0, P1);

    EXPECT_NE(E(2, 2), 0);
    EXPECT_NE(E1(2, 2), 0);
    EXPECT_NE(E2(2, 2), 0);

    E /= E(2, 2);
    E1 /= E1(2, 2);
    E2 /= E2(2, 2);

    double rms1 = matRMS(E, E1);
    double rms2 = matRMS(E, E2);
    double rms3 = matRMS(E1, E2);

    std::cout << "E: \n" << E << std::endl;
    std::cout << "E1: \n" << E1 << std::endl;
    std::cout << "E2: \n" << E2 << std::endl;
    std::cout << "RMS1: " << rms1 << std::endl;
    std::cout << "RMS2: " << rms2 << std::endl;
    std::cout << "RMS3: " << rms3 << std::endl;

    double eps = 1e-10;
    EXPECT_LT(rms1, eps);
    EXPECT_LT(rms2, eps);
    EXPECT_LT(rms3, eps);
}

TEST (SFM, TriangulationSimple) {

#if !ENABLE_MY_SFM
    return;
#endif

    vector4d X = {0, 0, 2, 1};

    matrix34d P0 = matrix34d::eye();
    vector3d x0 = {0, 0, 1};

    // P1
    vector3d O = {2, 0, 0};
    double alpha = M_PI_4;
    double s = std::sin(alpha);
    double c = std::cos(alpha);
    matrix3d R = { c, 0, s,
                   0, 1, 0,
                  -s, 0, c};
    vector3d T = -R * O;
    matrix34d P1 = {
             R(0, 0), R(0, 1), R(0, 2), T[0],
             R(1, 0), R(1, 1), R(1, 2), T[1],
             R(2, 0), R(2, 1), R(2, 2), T[2]
    };

    // x1
    vector3d x1 = {0, 0, 1};

    std::cout << "P1:\n" << P1 << std::endl;
    std::cout << "x2:\n" << P0 * X << std::endl;
    std::cout << "x3:\n" << P1 * X << std::endl;

    matrix34d Ps[2] = {P0, P1};
    vector3d xs[2] = {x0, x1};

    vector4d X1 = phg::triangulatePoint(Ps, xs, 2);
    std::cout << "X1:\n" << X1 << std::endl;

    EXPECT_NE(X1[3], 0);
    X1 /= X1[3];

    vector4d d = X - X1;
    std::cout << "|X - X1| = " << cv::norm(d) << std::endl;

    double eps = 1e-10;
    EXPECT_LT(cv::norm(d), eps);
}

TEST (SFM, FmatrixMatchFiltering) {

#if !ENABLE_MY_SFM
    return;
#endif

    using namespace cv;

    cv::Mat img1 = cv::imread("data/src/test_sfm/saharov/IMG_3023.JPG");
    cv::Mat img2 = cv::imread("data/src/test_sfm/saharov/IMG_3024.JPG");

    std::cout << "detecting points..." << std::endl;
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute( img1, cv::noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, cv::noArray(), keypoints2, descriptors2 );

    std::cout << "matching points..." << std::endl;
    std::vector<std::vector<DMatch>> knn_matches;

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    std::vector<DMatch> good_matches(knn_matches.size());
    for (int i = 0; i < (int) knn_matches.size(); ++i) {
        good_matches[i] = knn_matches[i][0];
    }

    std::cout << "filtering matches GMS..." << std::endl;
    std::vector<DMatch> good_matches_gms;
    phg::filterMatchesGMS(good_matches, keypoints1, keypoints2, img1.size(), img2.size(), good_matches_gms);

    std::cout << "filtering matches F..." << std::endl;
    std::vector<DMatch> good_matches_gms_plus_f;
    std::vector<DMatch> good_matches_f;
    double threshold_px = 3;
    {
        std::vector<cv::Vec2d> points1, points2;
        for (const cv::DMatch &match : good_matches) {
            cv::Vec2f pt1 = keypoints1[match.queryIdx].pt;
            cv::Vec2f pt2 = keypoints2[match.trainIdx].pt;
            points1.push_back(pt1);
            points2.push_back(pt2);
        }
        matrix3d F = phg::findFMatrix(points1, points2, threshold_px);
        filterMatchesF(good_matches, keypoints1, keypoints2, F, good_matches_f, threshold_px);
    }
    {
        std::vector<cv::Vec2d> points1, points2;
        for (const cv::DMatch &match : good_matches_gms) {
            cv::Vec2f pt1 = keypoints1[match.queryIdx].pt;
            cv::Vec2f pt2 = keypoints2[match.trainIdx].pt;
            points1.push_back(pt1);
            points2.push_back(pt2);
        }
        matrix3d F = phg::findFMatrix(points1, points2, threshold_px);
        filterMatchesF(good_matches_gms, keypoints1, keypoints2, F, good_matches_gms_plus_f, threshold_px);
    }

    drawMatches(img1, img2, keypoints1, keypoints2, good_matches_gms, "data/debug/test_sfm/matches_GMS.jpg");
    drawMatches(img1, img2, keypoints1, keypoints2, good_matches_f, "data/debug/test_sfm/matches_F.jpg");
    drawMatches(img1, img2, keypoints1, keypoints2, good_matches_gms_plus_f, "data/debug/test_sfm/matches_GMS_plus_F.jpg");

    std::cout << "n matches gms: " << good_matches_gms.size() << std::endl;
    std::cout << "n matches F: " << good_matches_f.size() << std::endl;
    std::cout << "n matches gms + F: " << good_matches_gms_plus_f.size() << std::endl;

    EXPECT_GT(good_matches_gms_plus_f.size(), 0.5 * good_matches_gms.size());
    EXPECT_GT(good_matches_f.size(), 0.5 * good_matches_gms.size());

    EXPECT_GT(good_matches_f.size(), 0.5 * good_matches_gms_plus_f.size());
    EXPECT_GT(good_matches_gms_plus_f.size(), 0.5 * good_matches_f.size());
}

TEST (SFM, RelativePosition2View) {

#if !ENABLE_MY_SFM
    return;
#endif

    using namespace cv;

    const cv::Mat img1 = cv::imread("data/src/test_sfm/saharov/IMG_3023.JPG");
    const cv::Mat img2 = cv::imread("data/src/test_sfm/saharov/IMG_3024.JPG");

    const phg::Calibration calib0(img1.cols, img1.rows);
    const phg::Calibration calib1(img2.cols, img2.rows);

    std::cout << "detecting points..." << std::endl;
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute( img1, cv::noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, cv::noArray(), keypoints2, descriptors2 );

    std::cout << "matching points..." << std::endl;
    std::vector<std::vector<DMatch>> knn_matches;

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    std::vector<DMatch> good_matches(knn_matches.size());
    for (int i = 0; i < (int) knn_matches.size(); ++i) {
        good_matches[i] = knn_matches[i][0];
    }

    std::cout << "filtering matches GMS..." << std::endl;
    std::vector<DMatch> good_matches_gms;
    phg::filterMatchesGMS(good_matches, keypoints1, keypoints2, img1.size(), img2.size(), good_matches_gms);

    std::vector<cv::Vec2d> points1, points2;
    for (const cv::DMatch &match : good_matches_gms) {
        cv::Vec2f pt1 = keypoints1[match.queryIdx].pt;
        cv::Vec2f pt2 = keypoints2[match.trainIdx].pt;
        points1.push_back(pt1);
        points2.push_back(pt2);
    }

    matrix3d F = phg::findFMatrix(points1, points2);
    matrix3d E = phg::fmatrix2ematrix(F, calib0, calib1);

    matrix34d P0, P1;
    phg::decomposeEMatrix(P0, P1, E, points1, points2, calib0, calib1);

    matrix3d R0, R1;
    vector3d O0, O1;
    phg::decomposeUndistortedPMatrix(R0, O0, P0);
    phg::decomposeUndistortedPMatrix(R1, O1, P1);

    std::cout << "Camera positions: " << std::endl;
    std::cout << "R0:\n" << R0 << std::endl;
    std::cout << "O0: " << O0.t() << std::endl;
    std::cout << "R1:\n" << R1 << std::endl;
    std::cout << "O1: " << O1.t() << std::endl;

    {
        vector3d relative_cos_vals = relativeOrientationAngles(R0, O0, R1, O1);
        std::cout << "relative_cos_vals: " << relative_cos_vals << std::endl;
        vector3d relative_cos_vals_expected = {0.961669, -0.1386, -0.404852};
        EXPECT_LT(cv::norm(relative_cos_vals - relative_cos_vals_expected), 0.05);
    }

    std::cout << "exporting point cloud..." << std::endl;
    std::vector<vector3d> point_cloud;
    std::vector<cv::Vec3b> point_cloud_colors;

    matrix34d Ps[2] = {P0, P1};
    for (int i = 0; i < (int) good_matches_gms.size(); ++i) {
        vector3d ms[2] = {calib0.unproject(points1[i]), calib1.unproject(points2[i])};
        vector4d X = phg::triangulatePoint(Ps, ms, 2);

        if (X(3) == 0) {
            std::cerr << "infinite point" << std::endl;
            continue;
        }

        point_cloud.push_back(vector3d{X(0) / X(3), X(1) / X(3), X(2) / X(3)});
        point_cloud_colors.push_back(img1.at<cv::Vec3b>(points1[i][1], points1[i][0]));
    }

    point_cloud.push_back(O0);
    point_cloud_colors.push_back(cv::Vec3b{0, 0, 255});
    point_cloud.push_back(O0 + R0.t() * cv::Vec3d(0, 0, 1));
    point_cloud_colors.push_back(cv::Vec3b(255, 0, 0));

    point_cloud.push_back(O1);
    point_cloud_colors.push_back(cv::Vec3b{0, 0, 255});
    point_cloud.push_back(O1 + R1.t() * cv::Vec3d(0, 0, 1));
    point_cloud_colors.push_back(cv::Vec3b(255, 0, 0));

    std::cout << "exporting " << point_cloud.size() << " points..." << std::endl;
    phg::exportPointCloud(point_cloud, "data/debug/test_sfm/point_cloud_2_cameras.ply", point_cloud_colors);
}

TEST (SFM, Resection) {

#if !ENABLE_MY_SFM
    return;
#endif

    using namespace cv;

    const cv::Mat img1 = cv::imread("data/src/test_sfm/saharov/IMG_3023.JPG");
    const cv::Mat img2 = cv::imread("data/src/test_sfm/saharov/IMG_3024.JPG");

    const phg::Calibration calib0(img1.cols, img1.rows);
    const phg::Calibration calib1(img2.cols, img2.rows);

    std::cout << "detecting points..." << std::endl;
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute( img1, cv::noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, cv::noArray(), keypoints2, descriptors2 );

    std::cout << "matching points..." << std::endl;
    std::vector<std::vector<DMatch>> knn_matches;

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    std::vector<DMatch> good_matches(knn_matches.size());
    for (int i = 0; i < (int) knn_matches.size(); ++i) {
        good_matches[i] = knn_matches[i][0];
    }

    std::cout << "filtering matches GMS..." << std::endl;
    std::vector<DMatch> good_matches_gms;
    phg::filterMatchesGMS(good_matches, keypoints1, keypoints2, img1.size(), img2.size(), good_matches_gms);

    std::vector<cv::Vec2d> points1, points2;
    for (const cv::DMatch &match : good_matches_gms) {
        cv::Vec2f pt1 = keypoints1[match.queryIdx].pt;
        cv::Vec2f pt2 = keypoints2[match.trainIdx].pt;
        points1.push_back(pt1);
        points2.push_back(pt2);
    }

    matrix3d F = phg::findFMatrix(points1, points2);
    matrix3d E = phg::fmatrix2ematrix(F, calib0, calib1);

    matrix34d P0, P1;
    phg::decomposeEMatrix(P0, P1, E, points1, points2, calib0, calib1);

    matrix3d R0, R1;
    vector3d O0, O1;
    phg::decomposeUndistortedPMatrix(R0, O0, P0);
    phg::decomposeUndistortedPMatrix(R1, O1, P1);

    std::cout << "Camera positions: " << std::endl;
    std::cout << "R0:\n" << R0 << std::endl;
    std::cout << "O0: " << O0.t() << std::endl;
    std::cout << "R1:\n" << R1 << std::endl;
    std::cout << "O1: " << O1.t() << std::endl;

    std::vector<cv::Vec3d> Xs;
    std::vector<cv::Vec2d> x0s;
    std::vector<cv::Vec2d> x1s;

    matrix34d Ps[2] = {P0, P1};
    for (int i = 0; i < (int) good_matches_gms.size(); ++i) {
        vector3d ms[2] = {calib0.unproject(points1[i]), calib1.unproject(points2[i])};
        vector4d X = phg::triangulatePoint(Ps, ms, 2);

        if (X(3) == 0) {
            std::cerr << "infinite point" << std::endl;
            continue;
        }

        Xs.push_back(vector3d{X(0) / X(3), X(1) / X(3), X(2) / X(3)});
        x0s.push_back(points1[i]);
        x1s.push_back(points2[i]);
    }

    matrix34d P0res = phg::findCameraMatrix(calib0, Xs, x0s);
    matrix34d P1res = phg::findCameraMatrix(calib1, Xs, x1s);

    double rms0 = matRMS(P0res, P0);
    double rms1 = matRMS(P1res, P1);
    double rms2 = matRMS(P0, P1);

    EXPECT_LT(rms0, 0.005);
    EXPECT_LT(rms1, 0.005);
    EXPECT_LT(rms0, 0.05 * rms2);
    EXPECT_LT(rms1, 0.05 * rms2);
}

namespace {

    // one track corresponds to one 3d point
    struct Track {
        std::vector<std::pair<int, int>> img_kpt_pairs;
    };

}

TEST (SFM, ReconstructNViews) {

#if !ENABLE_MY_SFM
    return;
#endif

    using namespace cv;

    std::vector<cv::Mat> imgs;
    imgs.push_back(cv::imread("data/src/test_sfm/saharov/IMG_3023.JPG"));
    imgs.push_back(cv::imread("data/src/test_sfm/saharov/IMG_3024.JPG"));
    imgs.push_back(cv::imread("data/src/test_sfm/saharov/IMG_3025.JPG"));

    std::vector<vector3d> expected_orientations;
    expected_orientations.push_back({0.961669, -0.1386, -0.404852});
    expected_orientations.push_back({0.972914, -0.535828, -0.715859});

    std::vector<phg::Calibration> calibs;
    for (const auto &img : imgs) {
        calibs.push_back(phg::Calibration(img.cols, img.rows));
    }

    const int n_imgs = imgs.size();

    std::cout << "detecting points..." << std::endl;
    std::vector<std::vector<cv::KeyPoint>> keypoints(n_imgs);
    std::vector<std::vector<int>> track_ids(n_imgs);
    std::vector<cv::Mat> descriptors(n_imgs);
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
    for (int i = 0; i < (int) imgs.size(); ++i) {
        detector->detectAndCompute(imgs[i], cv::noArray(), keypoints[i], descriptors[i]);
        track_ids[i].resize(keypoints[i].size(), -1);
    }

    std::cout << "matching points..." << std::endl;
    using Matches = std::vector<cv::DMatch>;
    std::vector<std::vector<Matches>> matches(n_imgs);
    for (int i = 0; i < n_imgs; ++i) {
        matches[i].resize(n_imgs);
        for (int j = 0; j < n_imgs; ++j) {
            if (i == j) {
                continue;
            }

            std::vector<std::vector<DMatch>> knn_matches;
            std::cout << "flann matching..." << std::endl;
            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
            matcher->knnMatch( descriptors[i], descriptors[j], knn_matches, 2 );
            std::vector<DMatch> good_matches(knn_matches.size());
            for (int k = 0; k < (int) knn_matches.size(); ++k) {
                good_matches[k] = knn_matches[k][0];
            }

            std::cout << "filtering matches GMS..." << std::endl;
            std::vector<DMatch> good_matches_gms;
            phg::filterMatchesGMS(good_matches, keypoints[i], keypoints[j], imgs[i].size(), imgs[j].size(), good_matches_gms);

            matches[i][j] = good_matches_gms;
        }
    }

    std::vector<Track> tracks;
    std::vector<vector3d> tie_points;
    std::vector<matrix34d> cameras(n_imgs);
    std::vector<char> aligned(n_imgs);

    // align first two cameras
    {
        // matches from first to second image in specified sequence
        const Matches &good_matches_gms = matches[0][1];
        const std::vector<cv::KeyPoint> &keypoints0 = keypoints[0];
        const std::vector<cv::KeyPoint> &keypoints1 = keypoints[1];
        const phg::Calibration &calib0 = calibs[0];
        const phg::Calibration &calib1 = calibs[1];

        std::vector<cv::Vec2d> points0, points1;
        for (const cv::DMatch &match : good_matches_gms) {
            cv::Vec2f pt1 = keypoints0[match.queryIdx].pt;
            cv::Vec2f pt2 = keypoints1[match.trainIdx].pt;
            points0.push_back(pt1);
            points1.push_back(pt2);
        }

        matrix3d F = phg::findFMatrix(points0, points1);
        matrix3d E = phg::fmatrix2ematrix(F, calib0, calib1);

        matrix34d P0, P1;
        phg::decomposeEMatrix(P0, P1, E, points0, points1, calib0, calib1);

        cameras[0] = P0;
        cameras[1] = P1;
        aligned[0] = true;
        aligned[1] = true;

        matrix34d Ps[2] = {P0, P1};
        for (int i = 0; i < (int) good_matches_gms.size(); ++i) {
            vector3d ms[2] = {calib0.unproject(points0[i]), calib1.unproject(points1[i])};
            vector4d X = phg::triangulatePoint(Ps, ms, 2);

            if (X(3) == 0) {
                std::cerr << "infinite point" << std::endl;
                continue;
            }

            vector3d X3d{X(0) / X(3), X(1) / X(3), X(2) / X(3)};

            tie_points.push_back(X3d);

            Track track;
            track.img_kpt_pairs.push_back({0, good_matches_gms[i].queryIdx});
            track.img_kpt_pairs.push_back({1, good_matches_gms[i].trainIdx});
            track_ids[0][good_matches_gms[i].queryIdx] = tracks.size();
            track_ids[1][good_matches_gms[i].trainIdx] = tracks.size();
            tracks.push_back(track);
        }
    }

    // append remaining cameras one by one
    for (int i_camera = 2; i_camera < n_imgs; ++i_camera) {

        const std::vector<cv::KeyPoint> &keypoints0 = keypoints[i_camera];
        const phg::Calibration &calib0 = calibs[i_camera];

        std::vector<vector3d> Xs;
        std::vector<vector2d> xs;
        for (int i_camera_prev = 0; i_camera_prev < i_camera; ++i_camera_prev) {
            const Matches &good_matches_gms = matches[i_camera][i_camera_prev];
            for (const cv::DMatch &match : good_matches_gms) {
                int track_id = track_ids[i_camera_prev][match.trainIdx];
                if (track_id != -1) {
                    Xs.push_back(tie_points[track_id]);
                    cv::Vec2f pt = keypoints0[match.queryIdx].pt;
                    xs.push_back(pt);
                }
            }
        }

        matrix34d P = phg::findCameraMatrix(calib0, Xs, xs);

        cameras[i_camera] = P;
        aligned[i_camera] = true;

        for (int i_camera_prev = 0; i_camera_prev < i_camera; ++i_camera_prev) {
            const std::vector<cv::KeyPoint> &keypoints1 = keypoints[i_camera_prev];
            const phg::Calibration &calib1 = calibs[i_camera_prev];
            const Matches &good_matches_gms = matches[i_camera][i_camera_prev];
            for (const cv::DMatch &match : good_matches_gms) {
                int track_id = track_ids[i_camera_prev][match.trainIdx];
                if (track_id == -1) {
                    matrix34d Ps[2] = {P, cameras[i_camera_prev]};
                    cv::Vec2f pts[2] = {keypoints0[match.queryIdx].pt, keypoints1[match.trainIdx].pt};
                    vector3d ms[2] = {calib0.unproject(pts[0]), calib1.unproject(pts[1])};
                    vector4d X = phg::triangulatePoint(Ps, ms, 2);

                    if (X(3) == 0) {
                        std::cerr << "infinite point" << std::endl;
                        continue;
                    }

                    tie_points.push_back({X(0) / X(3), X(1) / X(3), X(2) / X(3)});

                    Track track;
                    track.img_kpt_pairs.push_back({i_camera, match.queryIdx});
                    track.img_kpt_pairs.push_back({i_camera_prev, match.trainIdx});
                    track_ids[i_camera][match.queryIdx] = tracks.size();
                    track_ids[i_camera_prev][match.trainIdx] = tracks.size();
                    tracks.push_back(track);
                } else {
                    Track &track = tracks[track_id];
                    track.img_kpt_pairs.push_back({i_camera, match.queryIdx});
                    track_ids[i_camera][match.queryIdx] = track_id;
                }
            }
        }
    }

    if (tie_points.size() != tracks.size()) {
        throw std::runtime_error("tie_points.size() != tracks.size()");
    }

    std::vector<cv::Vec3b> tie_points_colors;
    for (int i = 0; i < (int) tie_points.size(); ++i) {
        const Track &track = tracks[i];
        int img = track.img_kpt_pairs.front().first;
        int kpt = track.img_kpt_pairs.front().second;
        cv::Vec2f px = keypoints[img][kpt].pt;
        tie_points_colors.push_back(imgs[img].at<cv::Vec3b>(px[1], px[0]));
    }

    for (int i_camera = 0; i_camera < n_imgs; ++i_camera) {
        if (!aligned[i_camera]) {
            throw std::runtime_error("camera " + std::to_string(i_camera) + " is not aligned");
        }

        matrix3d R;
        vector3d O;
        phg::decomposeUndistortedPMatrix(R, O, cameras[i_camera]);

        tie_points.push_back(O);
        tie_points_colors.push_back(cv::Vec3b(0, 0, 255));
        tie_points.push_back(O + R.t() * cv::Vec3d(0, 0, 1));
        tie_points_colors.push_back(cv::Vec3b(255, 0, 0));
    }

    for (int i = 1; i < n_imgs; ++i) {
        matrix3d R0, R1;
        vector3d O0, O1;
        phg::decomposeUndistortedPMatrix(R0, O0, cameras[i - 1]);
        phg::decomposeUndistortedPMatrix(R1, O1, cameras[i]);

        vector3d relative_cos_vals = relativeOrientationAngles(R0, O0, R1, O1);
        std::cout << "relative_cos_vals: " << relative_cos_vals << std::endl;
        vector3d relative_cos_vals_expected = expected_orientations[i - 1];
        EXPECT_LT(cv::norm(relative_cos_vals - relative_cos_vals_expected), 0.05);
    }

    std::cout << "exporting " << tie_points.size() << " points..." << std::endl;
    phg::exportPointCloud(tie_points, "data/debug/test_sfm/point_cloud_N_cameras.ply", tie_points_colors);

}
