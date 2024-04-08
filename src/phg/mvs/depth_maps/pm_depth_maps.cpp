#include <libutils/fast_random.h>
#include <libutils/timer.h>
#include <phg/utils/point_cloud_export.h>
#include "pm_depth_maps.h"

#include "pm_fast_random.h"
#include "pm_geometry.h"
#include "pm_depth_maps_defines.h"


namespace phg {
    
    matrix3d extractR(const matrix34d &P)
    {
        matrix3d RtoLocal;
        vector3d O;
        phg::decomposeUndistortedPMatrix(RtoLocal, O, P);
        return RtoLocal;
    }

    matrix34d invP(const matrix34d &P)
    {
        vector3d p(2.124, 5361.4, 78.6);
        
        vector3d p01 = P * homogenize(p);
        
        matrix3d RtoLocal;
        vector3d O;
        phg::decomposeUndistortedPMatrix(RtoLocal, O, P);
        matrix3d RtoWorld = RtoLocal.inv();
        matrix34d Pinv = make34(RtoWorld, O);

        vector3d p10 = Pinv * homogenize(p01);
        rassert(norm2(p10 - p) < 0.00001, 231231241233);

        return Pinv;
    }

    vector3d project(const vector3d &global_point, const phg::Calibration &calibration, const matrix34d &PtoLocal)
    {
        vector3d local_point = PtoLocal * homogenize(global_point);
        double depth = local_point[2];

        vector3f pixel_with_depth = calibration.project(local_point);
        pixel_with_depth[2] = depth; // на самом деле это не глубина, это координата по оси +Z (вдоль которой смотрит камера в ее локальной системе координат)

        return pixel_with_depth;
    }

    // TODO 101 реализуйте unproject (вам поможет тест на идемпотентность project -> unproject в test_depth_maps_pm)
    vector3d unproject(const vector3d &pixel, const phg::Calibration &calibration, const matrix34d &PtoWorld)
    {
        double depth = pixel[2]; // на самом деле это не глубина, это координата по оси +Z (вдоль которой смотрит камера в ее локальной системе координат)

        vector3d local_point; // TODO 102 пустите луч pixel из calibration а затем возьмите ан нем точку у которой по оси +Z координата=depth

        vector3d global_point; // TODO 103 переведите точку из локальной системы в глобальную

        return global_point;
    }
    
    void PMDepthMapsBuilder::buildDepthMap(
            unsigned int camera_key,
            cv::Mat &depth_map_res, cv::Mat &normal_map_res, cv::Mat &cost_map_res,
            float depth_min, float depth_max)
    {
        rassert(camera_key < ncameras, 238192841294108);
        rassert(ncameras >= 2, 21849182491209);

        ref_cam = camera_key;
        ref_depth_min = depth_min;
        ref_depth_max = depth_max;

        width = calibration.width();
        height = calibration.height(); 

        // в этих трех картинках мы будем хранить для каждого пикселя лучшую на данный момент найденную гипотезу
        depth_map  = cv::Mat::zeros(height, width, CV_32FC1); // глубина (точнее координата по оси Z в локальной системе камеры) на которой находится текущая гипотеза (если гипотезы нет - то глубина=0)
        normal_map = cv::Mat::zeros(height, width, CV_32FC3); // нормаль к плоскости поверхности текущей гипотезы (единичный вектор в глобальной системе координат)
        cost_map   = cv::Mat::zeros(height, width, CV_32FC1); // оценка качества этой гипотезы (от 0.0 до 1.0, чем меньше - тем лучше гипотеза)

        iter = 0;

        // в первую очередь нам надо заполнить случайными гипотезами, этим займется refinement
        refinement();

        for (iter = 1; iter <= NITERATIONS; ++iter) {
            propagation();
            refinement();
        }

        depth_map_res  = depth_map;
        normal_map_res = normal_map;
        cost_map_res   = cost_map;
    }

    void PMDepthMapsBuilder::refinement()
    {
        timer t;
        verbose_cout << "Iteration #" << iter << "/" << NITERATIONS << ": refinement..." << std::endl;

        #pragma omp parallel for schedule(dynamic, 1)
        for (ptrdiff_t j = 0; j < height; ++j) {
            for (ptrdiff_t i = 0; i < width; ++i) {
                // хотим полного детерминизма, поэтому seed для рандома порождаем из номера итерации + из номера нашего пикселя,
                // тем самым получаем полный детерминизм и результат не зависит от числа ядер процессора и в теории может воспроизводиться даже на видеокарте 
                FastRandom r(iter, j * width + i);

                // хотим попробовать улучшить текущие гипотезы рассмотрев взаимные комбинации следующих гипотез:
                float d0, dp, dr;
                vector3f n0, np, nr;

                {
                    // 1) текущей гипотезы (то что уже смогли найти)
                    d0 = depth_map.at<float>(j, i);
                    n0 = normal_map.at<vector3f>(j, i);

                    // 2) случайной пертурбации текущей гипотезы (мутация и уточнение того что уже смогли найти)
                    dp = r.nextf(d0 * 0.5f, d0 * 1.5); // TODO 104: сделайте так чтобы отклонение было тем меньше, чем номер итерации ближе к NITERATIONS, улучшило ли это результат?
                    np = cv::normalize(n0 + randomNormalObservedFromCamera(cameras_RtoWorld[ref_cam], r) * 0.5); // TODO 105: сделайте так чтобы отклонение было тем меньше, чем номер итерации ближе к NITERATIONS, улучшило ли это результат?

                    dp = std::max(ref_depth_min, std::min(ref_depth_max, dp));

                    // 3) новой случайной гипотезы из фрустума поиска (новые идеи, вечный поиск во всем пространстве)
                    // TODO 106: создайте случайную гипотезу dr+nr, вам поможет:
                    //  - r.nextf(...)
                    //  - ref_depth_min, ref_depth_max
                    //  - randomNormalObservedFromCamera - поможет создать нормаль которая гарантированно смотрит на нас 
                }

                float    best_depth  = d0;
                vector3f best_normal = n0;
                float    best_cost   = cost_map.at<float>(j, i);
                if (d0 == NO_DEPTH) {
                    best_cost = NO_COST;
                }

                float depths[3] = {d0, dr, dp};
                vector3f normals[3] = {n0, nr, np};

                // перебираем все комбинации этих гипотез, т.е. 3х3=9 вариантов
                for (size_t hi = 0; hi < 3*3; ++hi) {
                    // эту комбинацию-гипотезу мы сейчас рассматриваем как очередного кандидата
                    float    d = depths [hi / 3];
                    vector3f n = normals[hi % 3];

                    // оцениваем cost для каждого соседа
                    std::vector<float> costs; 
                    for (size_t ni = 0; ni < ncameras; ++ni) {
                        if (ni == ref_cam) continue;

                        float costi = estimateCost(i, j, d, n, ni);
                        costs.push_back(costi);
                    }

                    // объединяем cost-ы всех соседей в одну общую оценку качества текущей гипотезы (условно "усредняем")
                    float total_cost = avgCost(costs);

                    // WTA (winner takes all)
                    if (total_cost < best_cost) {
                        best_depth  = d;
                        best_normal = n;
                        best_cost   = total_cost; // TODO 206: добавьте подсчет статистики, какая комбинация гипотез чаще всего побеждает? есть ли комбинации на которых мы можем сэкономить? а какие гипотезы при refinement рассматривает например Colmap?
                    }
                }

                depth_map.at<float>(j, i)     = best_depth;
                normal_map.at<vector3f>(j, i) = best_normal;
                cost_map.at<float>(j, i)      = best_cost;
            }
        }

        verbose_cout << "refinement done in " << t.elapsed() << " s: ";
#ifdef VERBOSE_LOGGING
        printCurrentStats();
#endif
#ifdef DEBUG_DIR
        debugCurrentPoints(to_string(ref_cam) + "_" + to_string(iter) + "_refinement");
#endif
    }

    void PMDepthMapsBuilder::tryToPropagateDonor(ptrdiff_t ni, ptrdiff_t nj, int chessboard_pattern_step,
            std::vector<float> &hypos_depth, std::vector<vector3f> &hypos_normal, std::vector<float> &hypos_cost)
    {
        // rassert-ы или любой другой способ явной фиксации инвариантов со встроенной их проверкой в runtime -
        // это очень приятный способ ускорить отладку и гарантировать что ожидания в голове сойдутся с реальностью в коде,
        // а если разойдутся - то узнать об этом в самом первом сломавшемся предположении
        // (в данном случае мы явно проверяем что нигде не промахнулись и все соседи - другого шахматного цвета)
        // пусть лучше эта проверка упадет, мы сразу это заметим и отладим, чем бага будет тихо портить результаты
        // а мы это может быть даже не заметим
        rassert((ni + nj) % 2 != chessboard_pattern_step, 2391249129510120);

        if (ni < 0 || ni >= width || nj < 0 || nj >= height)
            return;
        
        float d = depth_map.at<float>(nj, ni);
        if (d == NO_DEPTH)
            return;

        vector3f n = normal_map.at<vector3f>(nj, ni);
        float cost = cost_map.at<float>(nj, ni);

        hypos_depth.push_back(d);
        hypos_normal.push_back(n);
        hypos_cost.push_back(cost);
    }

    void PMDepthMapsBuilder::propagation()
    {
        timer t;
        verbose_cout << "Iteration #" << iter << "/" << NITERATIONS << ": propagation..." << std::endl;

        for (int chessboard_pattern_step = 0; chessboard_pattern_step < 2; ++chessboard_pattern_step) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (ptrdiff_t j = 0; j < height; ++j) {
                for (ptrdiff_t i = (j + chessboard_pattern_step) % 2; i < width; i += 2) {
                    std::vector<float>    hypos_depth;
                    std::vector<vector3f> hypos_normal;
                    std::vector<float>    hypos_cost;

                    /* 4 прямых соседа A, 8 соседей B через диагональ, 4 соседа C вдалеке (условный рисунок для PROPAGATION_STEP=5):
                     * (удобно подсвечивать через Ctrl+F)
                     *         center
                     *           |
                     *           v
                     * o o o o o C o o o o o
                     * o o o o o o o o o o o
                     * o o o o o o o v o o o
                     * o o o o B o B o v o o
                     * o o o B o A o B o o o
                     * C o o o A . A o o o C  <- center
                     * o o o B o A o B o o o
                     * o o o o B o B o v o o
                     * o o o o o o o v o o o
                     * o o o o o o o o o o o
                     * o o o o o C o o o o o
                     */
                    tryToPropagateDonor(i - 1, j + 0, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                    tryToPropagateDonor(i + 0, j - 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                    tryToPropagateDonor(i + 1, j + 0, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                    tryToPropagateDonor(i + 0, j + 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);

                    tryToPropagateDonor(i - 2, j - 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                    tryToPropagateDonor(i - 1, j - 2, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                    tryToPropagateDonor(i + 1, j - 2, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                    tryToPropagateDonor(i + 2, j - 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                    tryToPropagateDonor(i + 2, j + 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                    tryToPropagateDonor(i + 1, j + 2, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                    tryToPropagateDonor(i - 1, j + 2, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                    tryToPropagateDonor(i - 2, j + 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);

                    // в таких случаях очень приятно использовать множественный курсор (чтобы скопировав четыре строки выше, затем просто колесиком мышки сделать четыре каретки для того чтобы дважды вставить *PROPAGATION_STEP):
                    tryToPropagateDonor(i - 1*PROPAGATION_STEP, j + 0*PROPAGATION_STEP, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                    tryToPropagateDonor(i + 0*PROPAGATION_STEP, j - 1*PROPAGATION_STEP, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                    tryToPropagateDonor(i + 1*PROPAGATION_STEP, j + 0*PROPAGATION_STEP, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                    tryToPropagateDonor(i + 0*PROPAGATION_STEP, j + 1*PROPAGATION_STEP, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                    
                    // TODO 201 переделайте чтобы было как в ACMH:
                    // TODO 202 - паттерн донорства
                    // TODO 203 - логика про "берем 8 лучших по их личной оценке - по их личному cost" и только их примеряем уже на себя для рассчета cost в нашей точке
                    // TODO 301 - сделайте вместо наивного переноса depth+normal в наш пиксель - логику про "пересекли луч из нашего пикселя с плоскостью которую задает донор-сосед" и оценку cost в нашей точке тогда можно провести для более релевантной точки-пересечения 

                    float    best_depth  = depth_map.at<float>(j, i);
                    vector3f best_normal = normal_map.at<vector3f>(j, i);
                    float    best_cost   = cost_map.at<float>(j, i);
                    if (best_depth == NO_DEPTH) {
                        best_cost = NO_COST;
                    }
    
                    for (size_t hi = 0; hi < hypos_depth.size(); ++hi) {
                        // эту гипотезу мы сейчас рассматриваем как очередного кандидата
                        float    d = hypos_depth[hi];
                        vector3f n = hypos_normal[hi];
    
                        // оцениваем cost для каждого соседа
                        std::vector<float> costs; 
                        for (size_t ni = 0; ni < ncameras; ++ni) {
                            if (ni == ref_cam) continue;
    
                            float costi = estimateCost(i, j, d, n, ni);
                            if (costi == NO_COST) continue;

                            costs.push_back(costi);
                        }
    
                        // объединяем cost-ы всех соседей в одну общую оценку качества текущей гипотезы (условно "усредняем")
                        float total_cost = avgCost(costs);
    
                        // WTA (winner takes all)
                        if (total_cost < best_cost) {
                            best_depth  = d;
                            best_normal = n;
                            best_cost   = total_cost;
                        }
                    }
    
                    depth_map.at<float>(j, i)     = best_depth;
                    normal_map.at<vector3f>(j, i) = best_normal;
                    cost_map.at<float>(j, i)      = best_cost;
                }
            }
        }

        verbose_cout << "propagation done in " << t.elapsed() << " s: ";
#ifdef VERBOSE_LOGGING
        printCurrentStats();
#endif
#ifdef DEBUG_DIR
        debugCurrentPoints(to_string(ref_cam) + "_" + to_string(iter) + "_propagation");
#endif
    }

    float PMDepthMapsBuilder::estimateCost(ptrdiff_t i, ptrdiff_t j, double d, const vector3d &global_normal, size_t neighb_cam)
    {
        vector3d pixel(i + 0.5, j + 0.5, d);
        vector3d global_point = unproject(pixel, calibration, cameras_PtoWorld[ref_cam]);

        if (!(i - COST_PATCH_RADIUS >= 0 && i + COST_PATCH_RADIUS < width))
            return NO_COST;
        if (!(j - COST_PATCH_RADIUS >= 0 && j + COST_PATCH_RADIUS < height))
            return NO_COST;
        
        std::vector<float> patch0, patch1;

        for (ptrdiff_t dj = -COST_PATCH_RADIUS; dj <= COST_PATCH_RADIUS; ++dj) {
            for (ptrdiff_t di = -COST_PATCH_RADIUS; di <= COST_PATCH_RADIUS; ++di) {
                ptrdiff_t ni = i + di;
                ptrdiff_t nj = j + dj;

                patch0.push_back(cameras_imgs_grey[ref_cam].at<unsigned char>(nj, ni) / 255.0f);

                vector3d point_on_ray  = unproject(vector3d(ni + 0.5, nj + 0.5, 1.0), calibration, cameras_PtoWorld[ref_cam]);
                vector3d camera_center = unproject(vector3d(ni + 0.5, nj + 0.5, 0.0), calibration, cameras_PtoWorld[ref_cam]); // TODO 204: это немного неестественный способ, можно поправить его на более явный вариант, например хранить центр камер в поле cameras_O

                vector3d ray_dir = cv::normalize(point_on_ray - camera_center);
                vector3d ray_org = camera_center;

                vector3d global_intersection;
                if (!intersectWithPlane(global_point, global_normal, ray_org, ray_dir, global_intersection))
                    return NO_COST; // луч не пересек плоскость (например наблюдаем ее под близким к прямому углу)

                rassert(neighb_cam != ref_cam, 2334195412410286);
                vector3d neighb_proj = project(global_intersection, calibration, cameras_PtoLocal[neighb_cam]);
                if (neighb_proj[2] < 0.0)
                    return NO_COST;

                double x = neighb_proj[0];
                double y = neighb_proj[1];

                // TODO 205: замените этот наивный вариант nearest neighbor сэмплирования текстуры на билинейную интерполяцию (учтите что центр пикселя - .5 после запятой)
                ptrdiff_t u = x;
                ptrdiff_t v = y;

                // TODO 108: добавьте проверку "попали ли мы в камеру номер neighb_cam?" если не попали - возвращаем NO_COST

                float intensity = cameras_imgs_grey[neighb_cam].at<unsigned char>(v, u) / 255.0f;
                patch1.push_back(intensity);
            }
        }

        // TODO 109: реализуйте ZNCC https://en.wikipedia.org/wiki/Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
        // или слайд #25 в лекции 5 про SGM и Cost-функции - https://my.compscicenter.ru/attachments/classes/slides_w2n8WNLY/photogrammetry_lecture_090321.pdf
        rassert(patch0.size() == patch1.size(), 12489185129326);
        size_t n = patch0.size();
        float mean0 = 0.0f;
        float mean1 = 0.0f;
        // ...
        for (size_t k = 0; k < n; ++k) {
            float a = patch0[k];
            float b = patch1[k];
            mean0 += a;
            mean1 += b;
            // ...
        }
        mean0 /= n;
        mean1 /= n;
        // ...
        float zncc = 0.0f;

        // ZNCC в диапазоне [-1; 1], 1: идеальное совпадение, -1: ничего общего
        rassert(zncc == zncc, 23141241210380); // проверяем что не nan
        zncc = std::max(-1.0f, std::min(1.0f, zncc));
        rassert(zncc >= -1.0 && zncc <= 1.0, 141251251541357);

        // переводим в cost от [0; 1] (NO_COST=1)
        // чем ближе cost к нулю - тем лучше сопоставление
        float cost = (1.0f - zncc) / 2.0f;
        rassert(cost >= 0.0f && cost <= NO_COST, 23123912049102361);
        
        return cost;
    }

    float PMDepthMapsBuilder::avgCost(std::vector<float> &costs)
    {
        if (costs.size() == 0)
            return NO_COST;

        std::sort(costs.begin(), costs.end());

        float best_cost = costs[0];

        float cost_sum = best_cost;
        float cost_w = 1.0f;

        // TODO 110 реализуйте какое-то "усреднение cost-ов по всем соседям", с ограничением что участвуют только COSTS_BEST_K_LIMIT лучших
        // TODO 111 добавьте к этому усреднению еще одно ограничение: если cost больше чем best_cost*COSTS_K_RATIO - то такой cost подозрительно плохой и мы его не хотим учитывать (вероятно occlusion)
        // TODO 112 а что если в пикселе occlusion, но best_cost - большой и поэтому отсечение по best_cost*COSTS_K_RATIO не срабатывает? можно ли это отсечение как-то выправить для такого случая?
        // TODO 207 а что если добавить какой-нибудь бонус в случае если больше чем Х камер засчиталось? улучшается/ухудшается ли от этого что-то на herzjezu25? а при большем числе фотографий

        float avg_cost = cost_sum / cost_w;
        return avg_cost;
    }

    void PMDepthMapsBuilder::printCurrentStats()
    {
        double costs_sum = 0.0;
        double costs_n = 0.0;
        double good_costs_sum = 0.0;
        double good_costs_n = 0.0;
        #pragma omp parallel for schedule(dynamic, 1) reduction(+:costs_sum, costs_n, good_costs_sum, good_costs_n)
        for (ptrdiff_t j = 0; j < height; ++j) {
            for (ptrdiff_t i = 0; i < width; ++i) {
                float d = depth_map.at<float>(j, i);
                if (d == NO_DEPTH)
                    continue;

                float cost = cost_map.at<float>(j, i);
                if (cost == NO_COST)
                    continue;

                costs_sum += cost;
                costs_n += 1.0;

                if (cost < GOOD_COST) {
                    good_costs_sum += cost;
                    good_costs_n += 1.0;
                }
            }
        }
        double ntotal = width * height;
        verbose_cout << to_percent(costs_n, ntotal)      << "% pixels with "      << (costs_sum / costs_n)           << " avg cost, ";
        verbose_cout << to_percent(good_costs_n, ntotal) << "% pixels with good " << (good_costs_sum / good_costs_n) << " avg cost";
        verbose_cout << std::endl;
    }

    void PMDepthMapsBuilder::debugCurrentPoints(const std::string &label)
    {
        std::vector<cv::Vec3d> point_cloud_all;
        std::vector<cv::Vec3b> point_cloud_all_bgr;
        std::vector<cv::Vec3d> point_cloud_all_normal;
        
        std::vector<cv::Vec3d> point_cloud_good;
        std::vector<cv::Vec3b> point_cloud_good_bgr;
        std::vector<cv::Vec3b> point_cloud_good_cost;
        std::vector<cv::Vec3d> point_cloud_good_normal;

        for (ptrdiff_t j = 0; j < height; ++j) {
            for (ptrdiff_t i = 0; i < width; ++i) {
                float depth = depth_map.at<float>(j, i);
                float cost = cost_map.at<float>(j, i);
                vector3d normal = normal_map.at<vector3f>(j, i);

                if (depth == NO_DEPTH || cost == NO_COST)
                    continue;

                cv::Vec3d p = unproject(vector3d(i + 0.5, j + 0.5, depth), calibration, cameras_PtoWorld[ref_cam]);
                cv::Vec3b bgr = cameras_imgs[ref_cam].at<cv::Vec3b>(j, i);
                point_cloud_all.push_back(p);
                point_cloud_all_bgr.push_back(bgr);
                point_cloud_all_normal.push_back(normal);

                if (cost > GOOD_COST)
                    continue;

                cv::Vec3b cost_bgr;
                for (int c = 0; c < 3; ++c) {
                    cost_bgr[c] = (unsigned char) (255.0f * (1.0f - cost / GOOD_COST));
                }
                point_cloud_good.push_back(p);
                point_cloud_good_bgr.push_back(bgr);
                point_cloud_good_cost.push_back(cost_bgr);
                point_cloud_good_normal.push_back(normal);
            }
        }

        exportPointCloud(point_cloud_all, DEBUG_DIR + label + "_all_rgb.ply", point_cloud_all_bgr, point_cloud_all_normal);
        exportPointCloud(point_cloud_good, DEBUG_DIR + label + "_good_rgb.ply", point_cloud_good_bgr, point_cloud_good_normal);
        exportPointCloud(point_cloud_good, DEBUG_DIR + label + "_good_costs.ply", point_cloud_good_cost, point_cloud_good_normal);
    }

    void PMDepthMapsBuilder::buildGoodPoints(const cv::Mat &depth_map, const cv::Mat &normal_map, const cv::Mat &cost_map, const cv::Mat &img,
                                             const phg::Calibration &calibration, const matrix34d &PtoWorld,
                                             std::vector<cv::Vec3d> &points, std::vector<cv::Vec3b> &colors, std::vector<cv::Vec3d> &normals)
    {
        ptrdiff_t width = calibration.width();
        ptrdiff_t height = calibration.height();

        for (ptrdiff_t j = 0; j < height; ++j) {
            for (ptrdiff_t i = 0; i < width; ++i) {
                float depth = depth_map.at<float>(j, i);
                float cost = cost_map.at<float>(j, i);
                vector3d normal = normal_map.at<vector3f>(j, i);

                if (depth == NO_DEPTH || cost == NO_COST)
                    continue;

                cv::Vec3d p = unproject(vector3d(i + 0.5, j + 0.5, depth), calibration, PtoWorld);
                cv::Vec3b bgr = img.at<cv::Vec3b>(j, i);

                if (cost > GOOD_COST)
                    continue;

                points.push_back(p);
                colors.push_back(bgr);
                normals.push_back(normal);
            }
        }
    }

}
