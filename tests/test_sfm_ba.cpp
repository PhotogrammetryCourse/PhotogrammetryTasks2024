#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <fstream>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/rasserts.h>
#include <phg/matching/gms_matcher.h>
#include <phg/sfm/fmatrix.h>
#include <phg/sfm/ematrix.h>
#include <phg/sfm/sfm_utils.h>
#include <phg/sfm/defines.h>
#include <phg/sfm/triangulation.h>
#include <phg/sfm/resection.h>
#include <phg/utils/point_cloud_export.h>

#include <ceres/rotation.h>
#include <ceres/ceres.h>

// TODO включите Bundle Adjustment (но из любопытства посмотрите как ведет себя реконструкция без BA например для saharov32 без BA)
#define ENABLE_BA                             0

// TODO когда заработает при малом количестве фотографий - увеличьте это ограничение до 100 чтобы попробовать обработать все фотографии (если же успешно будут отрабаывать только N фотографий - отправьте PR выставив здесь это N)
#define NIMGS_LIMIT                           10 // сколько фотографий обрабатывать (можно выставить меньше чтобы ускорить экспериментирование, или в случае если весь датасет не выравнивается)
#define INTRINSICS_CALIBRATION_MIN_IMGS       5 // начиная со скольки камер начинать оптимизировать внутренние параметры камеры (фокальную длину и т.п.) - из соображений что "пока камер мало - наблюдений может быть недостаточно чтобы не сойтись к ложной внутренней модели камеры"

#define ENABLE_INSTRINSICS_K1_K2              1 // TODO учитывать ли радиальную дисторсию - коэффициенты k1, k2 попробуйте с ним и и без saharov32, заметна ли разница?
#define INTRINSIC_K1_K2_MIN_IMGS              7 // начиная со скольки камер начинать оптимизировать k1, k2

// TODO попробуйте повыключать эти фильтрации выбросов, насколько изменился результат?
#define ENABLE_OUTLIERS_FILTRATION_3_SIGMA    1
#define ENABLE_OUTLIERS_FILTRATION_COLINEAR   1
#define ENABLE_OUTLIERS_FILTRATION_NEGATIVE_Z 1

//________________________________________________________________________________
// Datasets:

// достаточно чтобы у вас работало на этом датасете, тестирование на Travis CI тоже ведется на нем
#define DATASET_DIR                  "saharov32"
#define DATASET_DOWNSCALE            1 // картинки уже уменьшены в 4 раза (оригинальные вы можете скачать по ссылке из saharov32/LINK.txt)
#define DATASET_F                    (1585.5 / DATASET_DOWNSCALE)

// но если любопытно - для экспериментов предлагаются еще дополнительные датасеты
// скачайте их фотографии в папку data/src/datasets/DATASETNAME/ по ссылке из файла LINK.txt в папке датасета:

// saharov32 и herzjesu25 - приятные датасеты, вероятно их оба получится выравнять целиком
//#define DATASET_DIR                  "herzjesu25"
//#define DATASET_DOWNSCALE            2 // для ускорения SIFT
//#define DATASET_F                    (2761.5 / DATASET_DOWNSCALE) // see herzjesu25/K.txt
// TODO почему фокальная длина меняется от того что мы уменьшаем картинку? почему именно в такой пропорции? может надо домножать? или делить на downscale^2 ?

// но temple47 - не вышло, я не разобрался в чем с ним проблема, может быть слишком мало точек, может критерии фильтрации выкидышей для него слишком строги
//#define DATASET_DIR                  "temple47"
//#define DATASET_DOWNSCALE            1
//#define DATASET_F                    (1520.4 / DATASET_DOWNSCALE) // see temple47/README.txt about K-matrix (i.e. focal length = K11 from templeR_par.txt)

// Специальный датасет прямо с Марса!
/*
#define DATASET_DIR                  "perseverance25"
#define DATASET_DOWNSCALE            1
#define DATASET_F                    (4720.4 / DATASET_DOWNSCALE)
// на этом датасете фотографии длиннофокусные, поэтому многие лучи почти колинеарны, поэтому этот фильтр подавляет все точки и третья камера не подвыравнивается
#undef  ENABLE_OUTLIERS_FILTRATION_COLINEAR
#define ENABLE_OUTLIERS_FILTRATION_COLINEAR 0
 */
// и в целом все плохо... у меня не получилось выравнять этот датасет нашим простым прототипом
//________________________________________________________________________________


namespace {

    vector3d relativeOrientationAngles(const matrix3d &R0, const vector3d &O0, const matrix3d &R1, const vector3d &O1) {
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

    // one track corresponds to one 3d point
    class Track {
    public:
        Track()
        {
            disabled = false;
        }

        bool disabled;
        std::vector<std::pair<int, int>> img_kpt_pairs;
    };

}

void generateTiePointsCloud(const std::vector<vector3d> &tie_points,
                            const std::vector<Track> &tracks,
                            const std::vector<std::vector<cv::KeyPoint>> &keypoints,
                            const std::vector<cv::Mat> &imgs,
                            const std::vector<char> &aligned,
                            const std::vector<matrix34d> &cameras,
                            int ncameras,
                            std::vector<vector3d> &tie_points_and_cameras,
                            std::vector<cv::Vec3b> &tie_points_colors);

void runBA(std::vector<vector3d> &tie_points,
           std::vector<Track> &tracks,
           std::vector<std::vector<cv::KeyPoint>> &keypoints,
           std::vector<matrix34d> &cameras,
           int ncameras,
           phg::Calibration &calib,
           bool verbose=false);

TEST (SFM, ReconstructNViews) {
    using namespace cv;

    // Чтобы было проще - картинки упорядочены заранее в файле data/src/datasets/DATASETNAME/ordered_filenames.txt
    std::vector<cv::Mat> imgs;
    std::vector<std::string> imgs_labels;
    {
        std::ifstream in(std::string("data/src/datasets/") + DATASET_DIR + "/ordered_filenames.txt");
        size_t nimages = 0;
        in >> nimages;
        std::cout << nimages << " images" << std::endl;
        for (int i = 0; i < nimages; ++i) {
            std::string img_name;
            in >> img_name;
            std::string img_path = std::string("data/src/datasets/") + DATASET_DIR + "/" + img_name;
            cv::Mat img = cv::imread(img_path);

            if (img.empty()) {
                throw std::runtime_error("Can't read image: " + to_string(img_path));
            }

            // выполняем уменьшение картинки если оригинальные картинки в этом датасете - слишком большие для используемой реализации SIFT
            int downscale = DATASET_DOWNSCALE;
            while (downscale > 1) {
                cv::pyrDown(img, img);
                rassert(downscale % 2 == 0, 1249219412940115);
                downscale /= 2;
            }

            imgs.push_back(img);
            imgs_labels.push_back(img_name);
        }
    }

    phg::Calibration calib(imgs[0].cols, imgs[0].rows);
    calib.f_ = DATASET_F;

    // сверяем что все картинки одинакового размера (мы ведь предполагаем что их снимала одна и та же камера с одними и те же интринсиками)
    for (const auto &img : imgs) {
        rassert(img.cols == imgs[0].cols && img.rows == imgs[0].rows, 34125412512512);
    }

    const size_t n_imgs = std::min(imgs.size(), (size_t) NIMGS_LIMIT);

    std::cout << "detecting points..." << std::endl;
    std::vector<std::vector<cv::KeyPoint>> keypoints(n_imgs);
    std::vector<std::vector<int>> track_ids(n_imgs);
    std::vector<cv::Mat> descriptors(n_imgs);
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
    for (int i = 0; i < (int) n_imgs; ++i) {
        detector->detectAndCompute(imgs[i], cv::noArray(), keypoints[i], descriptors[i]);
        track_ids[i].resize(keypoints[i].size(), -1);
    }

    std::cout << "matching points..." << std::endl;
    using Matches = std::vector<cv::DMatch>;
    std::vector<std::vector<Matches>> matches(n_imgs);
    size_t ndone = 0;
    #pragma omp parallel for
    for (int i = 0; i < n_imgs; ++i) {
        matches[i].resize(n_imgs);
        for (int j = 0; j < n_imgs; ++j) {
            if (i == j) {
                continue;
            }

            // Flann matching
            std::vector<std::vector<DMatch>> knn_matches;
            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
            matcher->knnMatch( descriptors[i], descriptors[j], knn_matches, 2 );
            std::vector<DMatch> good_matches(knn_matches.size());
            for (int k = 0; k < (int) knn_matches.size(); ++k) {
                good_matches[k] = knn_matches[k][0];
            }

            // Filtering matches GMS
            std::vector<DMatch> good_matches_gms;
            int inliers = phg::filterMatchesGMS(good_matches, keypoints[i], keypoints[j], imgs[i].size(), imgs[j].size(), good_matches_gms, false);
            #pragma omp critical
            {
                ++ndone;
                if (inliers > 0) {
                    std::cout << to_percent(ndone, n_imgs * (n_imgs - 1)) + "% - Cameras " << i << "-" << j << " (" << imgs_labels[i] << "-" << imgs_labels[j] << "): " << inliers << " matches" << std::endl;
                }
            }

            matches[i][j] = good_matches_gms;
        }
    }

    std::vector<Track> tracks;
    std::vector<vector3d> tie_points;
    std::vector<matrix34d> cameras(n_imgs);
    std::vector<char> aligned(n_imgs);

    // align first two cameras
    {
        std::cout << "Initial alignment from cameras #0 and #1 (" << imgs_labels[0] << ", " << imgs_labels[1] << ")" << std::endl;
        // matches from first to second image in specified sequence
        const Matches &good_matches_gms = matches[0][1];
        const std::vector<cv::KeyPoint> &keypoints0 = keypoints[0];
        const std::vector<cv::KeyPoint> &keypoints1 = keypoints[1];
        const phg::Calibration &calib0 = calib;
        const phg::Calibration &calib1 = calib;

        std::vector<cv::Vec2d> points0, points1;
        for (const cv::DMatch &match : good_matches_gms) {
            cv::Vec2f pt1 = keypoints0[match.queryIdx].pt;
            cv::Vec2f pt2 = keypoints1[match.trainIdx].pt;
            points0.push_back(pt1);
            points1.push_back(pt2);
        }

        matrix3d F = phg::findFMatrix(points0, points1, 3, false);
        matrix3d E = phg::fmatrix2ematrix(F, calib0, calib1);

        matrix34d P0, P1;
        phg::decomposeEMatrix(P0, P1, E, points0, points1, calib0, calib1, false);

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

        int ncameras = 2;

        std::vector<vector3d> tie_points_and_cameras;
        std::vector<cv::Vec3b> tie_points_colors;
        generateTiePointsCloud(tie_points, tracks, keypoints, imgs, aligned, cameras, ncameras, tie_points_and_cameras, tie_points_colors);
        phg::exportPointCloud(tie_points_and_cameras, std::string("data/debug/test_sfm_ba/") + DATASET_DIR + "/point_cloud_" + to_string(ncameras) + "_cameras.ply", tie_points_colors);

#if ENABLE_BA
        runBA(tie_points, tracks, keypoints, cameras, ncameras, calib);
#endif
        generateTiePointsCloud(tie_points, tracks, keypoints, imgs, aligned, cameras, ncameras, tie_points_and_cameras, tie_points_colors);
        phg::exportPointCloud(tie_points_and_cameras, std::string("data/debug/test_sfm_ba/") + DATASET_DIR + "/point_cloud_" + to_string(ncameras) + "_cameras_ba.ply", tie_points_colors);
    }

    // append remaining cameras one by one
    for (int i_camera = 2; i_camera < n_imgs; ++i_camera) {

        const std::vector<cv::KeyPoint> &keypoints0 = keypoints[i_camera];
        const phg::Calibration &calib0 = calib;

        std::vector<vector3d> Xs;
        std::vector<vector2d> xs;
        for (int i_camera_prev = 0; i_camera_prev < i_camera; ++i_camera_prev) {
            const Matches &good_matches_gms = matches[i_camera][i_camera_prev];
            for (const cv::DMatch &match : good_matches_gms) {
                int track_id = track_ids[i_camera_prev][match.trainIdx];
                if (track_id != -1) {
                    if (tracks[track_id].disabled)
                        continue; // пропускаем выключенные точки (признанные выбросами)
                    Xs.push_back(tie_points[track_id]);
                    cv::Vec2f pt = keypoints0[match.queryIdx].pt;
                    xs.push_back(pt);
                }
            }
        }

        std::cout << "Append camera #" << i_camera << " (" << imgs_labels[i_camera] << ") to alignment via " << Xs.size() << " common points" << std::endl;
        rassert(Xs.size() > 0, 2318254129859128305);
        matrix34d P = phg::findCameraMatrix(calib0, Xs, xs, false);

        cameras[i_camera] = P;
        aligned[i_camera] = true;

        for (int i_camera_prev = 0; i_camera_prev < i_camera; ++i_camera_prev) {
            const std::vector<cv::KeyPoint> &keypoints1 = keypoints[i_camera_prev];
            const phg::Calibration &calib1 = calib;
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
                    if (tracks[track_id].disabled)
                        continue; // пропускаем выключенные точки (признанные выбросами)
                    Track &track = tracks[track_id];
                    track.img_kpt_pairs.push_back({i_camera, match.queryIdx});
                    track_ids[i_camera][match.queryIdx] = track_id;
                }
            }
        }

        int ncameras = i_camera + 1;

        std::vector<vector3d> tie_points_and_cameras;
        std::vector<cv::Vec3b> tie_points_colors;
        generateTiePointsCloud(tie_points, tracks, keypoints, imgs, aligned, cameras, ncameras, tie_points_and_cameras, tie_points_colors);
        phg::exportPointCloud(tie_points_and_cameras, std::string("data/debug/test_sfm_ba/") + DATASET_DIR + "/point_cloud_" + to_string(ncameras) + "_cameras.ply", tie_points_colors);

        // Запуск Bundle Adjustment
#if ENABLE_BA
        runBA(tie_points, tracks, keypoints, cameras, ncameras, calib);
#endif

        generateTiePointsCloud(tie_points, tracks, keypoints, imgs, aligned, cameras, ncameras, tie_points_and_cameras, tie_points_colors);
        phg::exportPointCloud(tie_points_and_cameras, std::string("data/debug/test_sfm_ba/") + DATASET_DIR + "/point_cloud_" + to_string(ncameras) + "_cameras_ba.ply", tie_points_colors);
    }
}

class ReprojectionError {
public:
    ReprojectionError(double x, double y) : observed_x(x), observed_y(y)
    {}

    template <typename T>
    bool operator()(const T* camera_extrinsics, // положение камеры:   [6] = {translation[3], rotation[3]} (разное для всех кадров, т.к. каждая фотография со своего ракурса)
                    const T* camera_intrinsics, // внутренние калибровочные параметры камеры: [5] = {k1, k2, f, cx, cy} (одни и те же для всех кадров, т.к. снято на одну и ту же камеру)
                    const T* point_global,      // 3D точка: [3]  = {x, y, z}
                    T* residuals) const {       // невязка:  [2]  = {dx, dy}
        // TODO реализуйте функцию проекции, все нужно делать в типе T чтобы ceres-solver мог под него подставить как Jet (очень рекомендую посмотреть Jet.h - как класная статья из википедии!), так и double

        // translation[3] - сдвиг в локальную систему координат камеры

        // rotation[3] - angle-axis rotation, поворачиваем точку point->p (чтобы перейти в локальную систему координат камеры)
        // подробнее см. https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
        // (P.S. у камеры всмысле вращения три степени свободы)

        // Проецируем точку на фокальную плоскость матрицы (т.е. плоскость Z=фокальная длина)

#if ENABLE_INSTRINSICS_K1_K2
        // k1, k2 - коэффициенты радиального искажения (radial distortion)
#endif

        // Домножаем на f, тем самым переводя в пиксели

        // Из координат когда точка (0, 0) - центр оптической оси
        // Переходим в координаты когда точка (0, 0) - левый верхний угол картинки
        // cx, cy - координаты центра оптической оси (обычно это центр картинки, но часто он чуть смещен)

        // Теперь по спроецированным координатам не забудьте посчитать невязку репроекции

        return true;
        // TODO сверьте эту функцию с вашей реализацией проекции в src/phg/core/calibration.cpp (они должны совпадать)
    }
protected:
    double observed_x;
    double observed_y;
};

void printCamera(double* camera_intrinsics)
{
    std::cout << "camera: k1=" << camera_intrinsics[0] << ", k2=" << camera_intrinsics[1] << ", "
              << "f=" << camera_intrinsics[2] << ", "
              << "cx=" << camera_intrinsics[3] << ", cy=" << camera_intrinsics[4] << std::endl;
}

void runBA(std::vector<vector3d> &tie_points,
           std::vector<Track> &tracks,
           std::vector<std::vector<cv::KeyPoint>> &keypoints,
           std::vector<matrix34d> &cameras,
           int ncameras,
           phg::Calibration &calib,
           bool verbose)
{
    // Формулируем задачу
    ceres::Problem problem;

    ASSERT_NEAR(calib.f_ , DATASET_F, 0.2 * DATASET_F);
    ASSERT_NEAR(calib.cx_, 0.0, 0.3 * calib.width());
    ASSERT_NEAR(calib.cy_, 0.0, 0.3 * calib.height());

    // внутренние калибровочные параметры камеры: [5] = {k1, k2, f, cx, cy}
    // TODO: преобразуйте calib в блок параметров камеры (ее внутренних характеристик) для оптимизации в BA
    double camera_intrinsics[5];
    std::cout << "Before BA ";
    printCamera(camera_intrinsics);

    const int CAMERA_EXTRINSICS_NPARAMS = 6;

    // внешние калибровочные параметры камеры для каждого кадра: [6] = {translation[3], rotation[3]}
    std::vector<double> cameras_extrinsics(CAMERA_EXTRINSICS_NPARAMS * ncameras, 0.0);
    for (size_t camera_id = 0; camera_id < ncameras; ++camera_id) {
        matrix3d R;
        vector3d O;
        // Декомпозируем на матрицу поворота (локальная система камеры -> глобальная система координат мира)
        //                + координаты камеры в мире (т.е. ее точка отсчета)
        phg::decomposeUndistortedPMatrix(R, O, cameras[camera_id]);

        double* camera_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;
        double* translation = camera_extrinsics + 0;
        double* rotation_angle_axis = camera_extrinsics + 3;

        // AngleAxisToRotationMatrix оперирует R матрицей у которой в памяти подряд идут колонки а не строчки:
        // > Conversions between 3x3 rotation matrix (in >>>column major order<<<) and
        // > axis-angle rotation representations. Templated for use with autodifferentiation.
        // поэтому нужно транспонировать:
        matrix3d Rt = R.t();
        ceres::RotationMatrixToAngleAxis(&(Rt(0, 0)), rotation_angle_axis);

        for (int d = 0; d < 3; ++d) {
            translation[d] = O[d];
        }
    }

    // остались только блоки параметров для 3D точек, но их аллоцировать не обязательно, т.к. мы можем их оптимизировать напрямую в tie_points массиве

    // TODO по хорошему, должна быть среднеквадратичным отклонением от наблюдаемой ошибки а не константой. Можно оставить так для простоты, можно поправить и сделать правильно
    const double sigma = 2.0; // измеряется в пикселях

    double inliers_mse = 0.0;
    size_t inliers = 0;
    size_t nprojections = 0;
    std::vector<double> cameras_inliers_mse(ncameras, 0.0);
    std::vector<size_t> cameras_inliers(ncameras, 0);
    std::vector<size_t> cameras_nprojections(ncameras, 0);

    std::vector<ceres::CostFunction*> reprojection_residuals;
    std::vector<ceres::CostFunction*> reprojection_residuals_for_deletion;

    // Создаем невязки для всех проекций 3D точек в камеры (т.е. для всех наблюдений этих ключевых точек)
    for (size_t i = 0; i < tie_points.size(); ++i) {
        const Track &track = tracks[i];
        for (size_t ci = 0; ci < track.img_kpt_pairs.size(); ++ci) {
            int camera_id = track.img_kpt_pairs[ci].first;
            int keypoint_id = track.img_kpt_pairs[ci].second;
            cv::Vec2f px = keypoints[camera_id][keypoint_id].pt;

            ceres::CostFunction* keypoint_reprojection_residual = new ceres::AutoDiffCostFunction<ReprojectionError,
                    2, // количество невязок (размер искомого residual массива переданного в функтор, т.е. размерность искомой невязки, у нас это dx, dy (ошибка проекции по обеим осям)
                    6, 5, 3> // число параметров в каждом блоке параметров, у нас три блок параметров (внешние параметры камеры[6], внутренние параметры камеры[5] и 3D точка)
                    (new ReprojectionError(px[0], px[1]));
            reprojection_residuals.push_back(keypoint_reprojection_residual);

            double* camera_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;

            // блоки параметров для 3D точек аллоцировать не обязательно, т.к. мы можем их оптимизировать напрямую в tie_points массиве
            double* point3d_params = &(tie_points[i][0]);

            {
                const double* params[3];
                double residual[2] = {-1.0};
                params[0] = camera_extrinsics;
                params[1] = camera_intrinsics;
                params[2] = point3d_params;
                keypoint_reprojection_residual->Evaluate(params, residual, NULL);
                double error2 = residual[0] * residual[0] + residual[1] * residual[1];
                if (error2 < 3.0 * sigma) {
                    inliers_mse += error2;
                    ++inliers;
                    cameras_inliers_mse[camera_id] += error2;
                    ++cameras_inliers[camera_id];
                }
                ++nprojections;
                ++cameras_nprojections[camera_id];
            }

            if (!track.disabled) {
                problem.AddResidualBlock(keypoint_reprojection_residual, new ceres::HuberLoss(3.0 * sigma),
                                         camera_extrinsics,
                                         camera_intrinsics,
                                         point3d_params);
            } else {
                reprojection_residuals_for_deletion.push_back(keypoint_reprojection_residual); // если мы не передали невязку в ceres-solver, то за его lifetime ответственны все еще мы
            }
        }
    }
    std::cout << "Before BA projections: " << to_percent(inliers, nprojections) << "% inliers with MSE=" << (inliers_mse / inliers) << std::endl;
    for (size_t camera_id = 0; camera_id < ncameras; ++camera_id) {
        size_t ninls = cameras_inliers[camera_id];
        size_t nproj = cameras_nprojections[camera_id];
        std::cout << "    Camera #" << camera_id << " projections: " << to_percent(ninls, nproj) << "% inliers "
        << "(" << ninls << "/" << nproj << ") with MSE=" << (cameras_inliers_mse[camera_id] / ninls) << std::endl;
    }

    if (ncameras < INTRINSICS_CALIBRATION_MIN_IMGS) {
        // Полностью фиксируем внутренние калибровочные параметры камеры,
        // т.к. пока что наблюдений мало, и мы можем сойтись к какому-то неправильному решению которое потом не выйдет спасти
        // иначе говоря пока наблюдений мало - лучше уменьшить число степеней свободы, чтобы не испортить решение фатально
        problem.SetParameterBlockConstant(camera_intrinsics);
    } else {
        if (ncameras < INTRINSIC_K1_K2_MIN_IMGS) {
            problem.SetParameterization(camera_intrinsics, new ceres::SubsetParameterization(5, {0, 1}));
        }
    }

    {
        // Полностью фиксируем положение первой камеры (чтобы не уползло облако точек)
        size_t camera_id = 0;
        double* camera0_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;
        problem.SetParameterBlockConstant(camera0_extrinsics);
    }
    {
        // Фиксируем координаты второй камеры, т.е. translation[3] (чтобы фиксировать масштаб)
        size_t camera_id = 1;
        double* camera1_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;
        problem.SetParameterization(camera1_extrinsics, new ceres::SubsetParameterization(6, {0, 1, 2}));
    }
//http://ceres-solver.org/nnls_solving.html
    if (ENABLE_BA) {
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = verbose;
        ceres::Solver::Summary summary;
        Solve(options, &problem, &summary);

        if (verbose) {
            std::cout << summary.BriefReport() << std::endl;
        }
    }

    std::cout << "After BA ";
    printCamera(camera_intrinsics);
    // TODO преобразуйте параметры камеры в обратную сторону, чтобы последующая резекция учла актуальное представление о пространстве:
    // calib.* = camera_intrinsics[*];

    ASSERT_NEAR(calib.f_ , DATASET_F, 0.2 * DATASET_F);
    ASSERT_NEAR(calib.cx_, 0.0, 0.3 * calib.width());
    ASSERT_NEAR(calib.cy_, 0.0, 0.3 * calib.height());

    for (size_t camera_id = 0; camera_id < ncameras; ++camera_id) {
        matrix3d R;
        vector3d O;

        phg::decomposeUndistortedPMatrix(R, O, cameras[camera_id]);
        std::cout << "Camera #" << camera_id << " center: " << O << " -> ";

        double* camera_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;
        double* translation = camera_extrinsics + 0;
        double* rotation_angle_axis = camera_extrinsics + 3;

        matrix3d Rt;
        ceres::AngleAxisToRotationMatrix(rotation_angle_axis, &(Rt(0, 0)));
        // AngleAxisToRotationMatrix оперирует R матрицей у которой в памяти подряд идут колонки а не строчки:
        // > Conversions between 3x3 rotation matrix (in >>>column major order<<<) and
        // > axis-angle rotation representations. Templated for use with autodifferentiation.
        // поэтому нужно транспонировать:
        R = Rt.t();

        for (int d = 0; d < 3; ++d) {
            O[d] = translation[d];
        }

        std::cout << O << std::endl;
        cameras[camera_id] = phg::composeCameraMatrixRO(R, O);
    }

    inliers_mse = 0.0;
    inliers = 0;
    nprojections = 0;
    cameras_inliers_mse = std::vector<double>(ncameras, 0.0);
    cameras_inliers = std::vector<size_t>(ncameras, 0);
    cameras_nprojections = std::vector<size_t>(ncameras, 0);

    size_t n_old_outliers = 0;
    size_t n_new_outliers = 0;

    size_t next_loss_k = 0;
    for (size_t i = 0; i < tie_points.size(); ++i) {
        Track &track = tracks[i];
        bool should_be_disabled = false;

        vector3d track_point = tie_points[i];

        for (size_t ci = 0; ci < track.img_kpt_pairs.size(); ++ci) {
            int camera_id = track.img_kpt_pairs[ci].first;

            ceres::CostFunction* keypoint_reprojection_residual = reprojection_residuals[next_loss_k++];

            double* camera_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;

            double* point3d_params = &(tie_points[i][0]);

            matrix3d R; vector3d camera_origin;
            phg::decomposeUndistortedPMatrix(R, camera_origin, cameras[camera_id]);

            if (ENABLE_OUTLIERS_FILTRATION_NEGATIVE_Z && ENABLE_BA) {
                vector3d track_in_camera = R * (track_point - camera_origin);
                double z = track_in_camera[2];
                if (z < 0.0) {
                    // за спиной камеры
                    if (ENABLE_OUTLIERS_FILTRATION_NEGATIVE_Z && ENABLE_BA)
                        should_be_disabled = true;
                }
            }

            if (ENABLE_OUTLIERS_FILTRATION_COLINEAR && ENABLE_BA) {
                // TODO выполните проверку случая когда два луча почти параллельны, чтобы не было странных точек улетающих на бесконечность (например чтобы угол был хотя бы 2.5 градуса)
                // should_be_disabled = true;
            }

            {
                const double* params[3];
                double residual[2] = {-1.0};
                params[0] = camera_extrinsics;
                params[1] = camera_intrinsics;
                params[2] = point3d_params;
                keypoint_reprojection_residual->Evaluate(params, residual, NULL);
                double error2 = residual[0] * residual[0] + residual[1] * residual[1];
                if (error2 < 3.0 * sigma) {
                    inliers_mse += error2;
                    ++inliers;
                    cameras_inliers_mse[camera_id] += error2;
                    ++cameras_inliers[camera_id];
                } else {
                    if (ENABLE_OUTLIERS_FILTRATION_3_SIGMA && ENABLE_BA)
                        should_be_disabled = true;
                }
                ++nprojections;
                ++cameras_nprojections[camera_id];
            }
        }

        if (should_be_disabled && !track.disabled) {
            track.disabled = true;
            ++n_new_outliers;
        } else if (track.disabled) {
            ++n_old_outliers;
        }
    }
    std::cout << "After BA tie poits: " << to_percent(n_old_outliers, tie_points.size()) << "% old + " << to_percent(n_new_outliers, tie_points.size()) << "% new = " << to_percent(n_old_outliers + n_new_outliers, tie_points.size()) << "% total outliers" << std::endl;
    std::cout << "After BA projections: " << to_percent(inliers, nprojections) << "% inliers with MSE=" << (inliers_mse / inliers) << std::endl;
    for (size_t camera_id = 0; camera_id < ncameras; ++camera_id) {
        size_t ninls = cameras_inliers[camera_id];
        size_t nproj = cameras_nprojections[camera_id];
        double mse = (cameras_inliers_mse[camera_id] / ninls);
        std::cout << "    Camera #" << camera_id << " projections: " << to_percent(ninls, nproj) << "% inliers "
                  << "(" << ninls << "/" << nproj << ") with MSE=" << mse << std::endl;
        ASSERT_GT(ninls, 0.15 * nproj);
    }

    for (auto ptr : reprojection_residuals_for_deletion) {
        delete ptr; // т.к. мы не отдали указатель в ceres-solver - мы ответственны за его lifetime - надо самим деаллоцировать
    }
}

void generateTiePointsCloud(const std::vector<vector3d> &tie_points,
                            const std::vector<Track> &tracks,
                            const std::vector<std::vector<cv::KeyPoint>> &keypoints,
                            const std::vector<cv::Mat> &imgs,
                            const std::vector<char> &aligned,
                            const std::vector<matrix34d> &cameras,
                            int ncameras,
                            std::vector<vector3d> &tie_points_and_cameras,
                            std::vector<cv::Vec3b> &tie_points_colors)
{
    rassert(tie_points.size() == tracks.size(), 24152151251241);

    tie_points_and_cameras.clear();
    tie_points_colors.clear();

    for (int i = 0; i < (int) tie_points.size(); ++i) {
        const Track &track = tracks[i];
        if (track.disabled)
            continue;

        int img = track.img_kpt_pairs.front().first;
        int kpt = track.img_kpt_pairs.front().second;
        cv::Vec2f px = keypoints[img][kpt].pt;
        tie_points_and_cameras.push_back(tie_points[i]);
        tie_points_colors.push_back(imgs[img].at<cv::Vec3b>(px[1], px[0]));
    }

    for (int i_camera = 0; i_camera < ncameras; ++i_camera) {
        if (!aligned[i_camera]) {
            throw std::runtime_error("camera " + std::to_string(i_camera) + " is not aligned");
        }

        matrix3d R;
        vector3d O;
        phg::decomposeUndistortedPMatrix(R, O, cameras[i_camera]);

        tie_points_and_cameras.push_back(O);
        tie_points_colors.push_back(cv::Vec3b(0, 0, 255));
        tie_points_and_cameras.push_back(O + R.t() * cv::Vec3d(0, 0, 1));
        tie_points_colors.push_back(cv::Vec3b(255, 0, 0));
    }
}
