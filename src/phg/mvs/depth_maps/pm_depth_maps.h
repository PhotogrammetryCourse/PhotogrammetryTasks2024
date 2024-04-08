#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include <libutils/rasserts.h>

#include <phg/sfm/defines.h>
#include <phg/sfm/ematrix.h>
#include <phg/core/calibration.h>


namespace phg {

    matrix3d extractR(const matrix34d &P);
    matrix34d invP(const matrix34d &P);

    vector3d project  (const vector3d &global_point, const phg::Calibration &calibration, const matrix34d &PtoLocal);
    vector3d unproject(const vector3d &pixel,        const phg::Calibration &calibration, const matrix34d &PtoWorld);

    class PMDepthMapsBuilder {
    public:
        PMDepthMapsBuilder(
                const size_t                      &ncameras,
                const std::vector<cv::Mat>        &cameras_imgs,
                const std::vector<cv::Mat>        &cameras_imgs_grey,
                const std::vector<std::string>    &cameras_labels,
                const std::vector<matrix34d>      &cameras_P,
                const phg::Calibration            &calibration)
                : ncameras(ncameras),
                  cameras_imgs(cameras_imgs), cameras_imgs_grey(cameras_imgs_grey),
                  cameras_labels(cameras_labels), cameras_PtoLocal(cameras_P),
                  calibration(calibration)
        {
            cameras_PtoWorld.resize(ncameras);
            cameras_RtoWorld.resize(ncameras);

            rassert(cameras_imgs.size() == ncameras, 2815841251015);
            rassert(cameras_imgs_grey.size() == ncameras, 2815841251014);
            rassert(cameras_labels.size() == ncameras, 2815841251016);
            rassert(cameras_P.size() == ncameras, 2815841251017);
            for (size_t ci = 0; ci < ncameras; ++ci) {
                rassert(cameras_imgs[ci].cols == calibration.width(), 23812989251031);
                rassert(cameras_imgs[ci].rows == calibration.height(), 23812989251032);

                rassert(cameras_imgs_grey[ci].cols == calibration.width(), 23812989251033);
                rassert(cameras_imgs_grey[ci].rows == calibration.height(), 23812989251034);
                rassert(cameras_imgs_grey[ci].channels() == 1, 23812989251035);

                cameras_PtoWorld[ci] = invP(cameras_PtoLocal[ci]);
                cameras_RtoWorld[ci] = extractR(cameras_PtoWorld[ci]);
            }
        }

        void buildDepthMap(
                unsigned int camera_key,
                cv::Mat &depth_map, cv::Mat &normal_map, cv::Mat &cost_map,
                float depth_min, float depth_max);

        matrix34d getCameraPtoWorld(unsigned int camera_key) { rassert(camera_key < ncameras, 23172890061); return cameras_PtoWorld[camera_key]; }

        // эта функция берет только те "хорошие" (с учетом значения cost_map) точки карты глубины
        // и преобразует их в результат - облако точек с цветами и нормалями (теми что были определены в финальных гипотезах пикселей)
        static void buildGoodPoints(const cv::Mat &depth_map, const cv::Mat &normal_map, const cv::Mat &cost_map, const cv::Mat &img,
                                    const phg::Calibration &calibration, const matrix34d &PtoWorld,
                                    std::vector<cv::Vec3d> &points, std::vector<cv::Vec3b> &colors, std::vector<cv::Vec3d> &normals);

    protected:

        void refinement();

        void propagation();

        float estimateCost(ptrdiff_t i, ptrdiff_t j, double d, const vector3d &global_normal, size_t neighb_cam);
        float avgCost(std::vector<float> &costs);
        void  tryToPropagateDonor(ptrdiff_t ni, ptrdiff_t nj, int chessboard_pattern_step,
                std::vector<float> &hypos_depth, std::vector<vector3f> &hypos_normal, std::vector<float> &hypos_cost);

        void printCurrentStats();
        void debugCurrentPoints(const std::string &label);

        const size_t                        &ncameras;
        const std::vector<cv::Mat>          &cameras_imgs;
        const std::vector<cv::Mat>          &cameras_imgs_grey;
        const std::vector<std::string>      &cameras_labels;

        const std::vector<matrix34d>        &cameras_PtoLocal; // матрица переводящая глобальную систему координат мира в систему координат i-ой камеры (смотрящей по оси +Z)
        std::vector<matrix34d>               cameras_PtoWorld; // матрица переводящая локальную систему координат i-ой камеры (смотрящей по оси +Z) в глобальную систему координат мира
        std::vector<matrix3d>                cameras_RtoWorld; // матрица поворота из локальной системы координат i-ой камеры (смотрящей по оси +Z) в нлобальную систему координат мира

        const phg::Calibration              &calibration;
        
        unsigned int                        ref_cam; // индекс референсной камеры (для которой мы строим карту глубины)
        float                               ref_depth_min, ref_depth_max; // минимальная и максимальная глубины фрустума (на самом деле минимальная и максимальная допустимая координата по оси Z в локальной системе референсной камеры)

        size_t                              width, height; // ширина и высота референсной камеры и ее карт: глубины, нормалей, качества гипотез
        cv::Mat                             depth_map;
        cv::Mat                             normal_map;
        cv::Mat                             cost_map;
        
        int                                 iter; // номер итерации

        
    };
    
}