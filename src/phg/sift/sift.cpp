#include "sift.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <libutils/rasserts.h>
#include <csignal>
#include <fstream>

template <typename Num>
auto range(Num x) {
    return std::ranges::iota_view{(Num)0, x};
}

using i32 = int32_t;

// Ссылки:
// [lowe04] - Distinctive Image Features from Scale-Invariant Keypoints, David G. Lowe, 2004
//
// Примеры реализаций (стоит обращаться только если совсем не понятны какие-то места):
// 1) https://github.com/robwhess/opensift/blob/master/src/sift.c
// 2) https://gist.github.com/lxc-xx/7088609 (адаптация кода с первой ссылки)
// 3) https://github.com/opencv/opencv/blob/1834eed8098aa2c595f4d1099eeaa0992ce8b321/modules/features2d/src/sift.dispatch.cpp (адаптация кода с первой ссылки)
// 4) https://github.com/opencv/opencv/blob/1834eed8098aa2c595f4d1099eeaa0992ce8b321/modules/features2d/src/sift.simd.hpp (адаптация кода с первой ссылки)

#define DEBUG_ENABLE     1
#define DEBUG_PATH       std::string("data/debug/test_sift/debug/")

#define NOCTAVES                    3                    // число октав
#define OCTAVE_NLAYERS              3                    // в [lowe04] это число промежуточных степеней размытия картинки в рамках одной октавы обозначается - s, т.е. s слоев в каждой октаве
#define OCTAVE_GAUSSIAN_IMAGES      (OCTAVE_NLAYERS + 3)
#define OCTAVE_DOG_IMAGES           (OCTAVE_NLAYERS + 2)
#define INITIAL_IMG_SIGMA           0.75                 // предполагаемая степень размытия изначальной картинки
#define INPUT_IMG_PRE_BLUR_SIGMA    1.0                  // сглаживание изначальной картинки

#define SUBPIXEL_FITTING_ENABLE      0    // такие тумблеры включающие/выключающие очередное улучшение алгоритма позволяют оценить какой вклад эта фича вносит в качество результата если в рамках уже готового алгоритма попробовать ее включить/выключить

#define ORIENTATION_NHISTS           8   // число корзин при определении ориентации ключевой точки через гистограммы
#define ORIENTATION_WINDOW_R         3    // минимальный радиус окна в рамках которого будет выбрана ориентиация (в пикселях), R=3 => 5x5 окно
#define ORIENTATION_VOTES_PEAK_RATIO 0.80 // 0.8 => если гистограмма какого-то направления получила >= 80% от максимального чиссла голосов - она тоже победила

#define DESCRIPTOR_SIZE            2 // 4x4 гистограммы декскриптора
#define DESCRIPTOR_NBINS           8 // 8 корзин-направлений в каждой гистограмме дескриптора (4х4 гистограммы, каждая по 8 корзин, итого 4x4x8=128 значений в дескрипторе)
#define DESCRIPTOR_SAMPLES_N       1 // 4x4 замера для каждой гистограммы дескриптора (всего гистограмм 4х4) итого 16х16 замеров
#define DESCRIPTOR_SAMPLE_WINDOW_R 1.0 // минимальный радиус окна в рамках которого строится гистограмма из 8 корзин-направлений (т.е. для каждого из 16 элементов дескриптора), R=1 => 1x1 окно


void phg::SIFT::detectAndCompute(const cv::Mat &originalImg, std::vector<cv::KeyPoint> &kps, cv::Mat &desc) {
    // используйте дебаг в файлы как можно больше, это очень удобно и потраченное время окупается крайне сильно,
    // ведь пролистывать через окошки показывающие картинки долго, и по ним нельзя проматывать назад, а по файлам - можно
    // вы можете запустить алгоритм, сгенерировать десятки картинок со всеми промежуточными визуализациями и после запуска
    // посмотреть на те этапы к которым у вас вопросы или про которые у вас опасения
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "00_input.png", originalImg);

    cv::Mat img = originalImg.clone();
    // для удобства используем черно-белую картинку и работаем с вещественными числами (это еще и может улучшить точность)
    if (originalImg.type() == CV_8UC1) { // greyscale image
        img.convertTo(img, CV_32FC1, 1.0);
    } else if (originalImg.type() == CV_8UC3) { // BGR image
        img.convertTo(img, CV_32FC3, 1.0);
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    } else {
        rassert(false, 14291409120);
    }
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "01_grey.png", img);
    cv::GaussianBlur(img, img, cv::Size(0, 0), INPUT_IMG_PRE_BLUR_SIGMA, INPUT_IMG_PRE_BLUR_SIGMA);
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "02_grey_blurred.png", img);

    // Scale-space extrema detection
    std::vector<cv::Mat> gaussianPyramid;
    std::vector<cv::Mat> DoGPyramid;
    buildPyramids(img, gaussianPyramid, DoGPyramid);

    findLocalExtremasAndDescribe(gaussianPyramid, DoGPyramid, kps, desc);
}

void phg::SIFT::buildPyramids(const cv::Mat &imgOrg, std::vector<cv::Mat> &gaussianPyramid, std::vector<cv::Mat> &DoGPyramid) {
    gaussianPyramid.resize(NOCTAVES * OCTAVE_GAUSSIAN_IMAGES);

    const double k = pow(2.0, 1.0 / OCTAVE_NLAYERS); // [lowe04] k = 2^{1/s} а у нас s=OCTAVE_NLAYERS

    // строим пирамиду гауссовых размытий картинки
    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        if (octave == 0) {
            int firstOcteveLayer = 0;
            gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + firstOcteveLayer] = imgOrg.clone();
        } else {
            int firstOctaveLayer = 0;
            size_t prevOctave = octave - 1;
            // берем картинку с предыдущей октавы и уменьшаем ее в два раза без какого бы то ни было дополнительного размытия (сигмы должны совпадать)
             cv::Mat img = gaussianPyramid[prevOctave *  OCTAVE_GAUSSIAN_IMAGES].clone();
            // тут есть очень важный момент, мы должны указать fx=0.5, fy=0.5 иначе при нечетном размере картинка будет не идеально 2 пикселя в один схлопываться - а слегка смещаться
             cv::resize(img, img, cv::Size( img.cols / 2 , img.rows / 2 ), 0.5, 0.5, cv::INTER_NEAREST);
             gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + firstOctaveLayer] = img;
        }

//        #pragma omp parallel for // TODO: если выполните TODO про "размытие из изначального слоя октавы" ниже - раскоментируйте это распараллеливание, ведь теперь слои считаются независимо (из самого первого), проверьте что результат на картинках не изменился
        for (ptrdiff_t layer = 1; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            size_t prevLayer = layer - 1;

            // если есть два последовательных гауссовых размытия с sigma1 и sigma2, то результат будет с sigma12=sqrt(sigma1^2 + sigma2^2) => sigma2=sqrt(sigma12^2-sigma1^2)
            double sigmaPrev = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, prevLayer); // sigma1  - сигма до которой дошла картинка на предыдущем слое
            double sigmaCur  = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);     // sigma12 - сигма до которой мы хотим дойти на текущем слое
            double sigma = sqrt(sigmaCur*sigmaCur - sigmaPrev*sigmaPrev);                // sigma2  - сигма которую надо добавить чтобы довести sigma1 до sigma12
            // посмотрите внимательно на формулу выше и решите как по мнению этой формулы соотносится сигма у первого А-слоя i-ой октавы
            // и сигма у одного из последних слоев Б предыдущей (i-1)-ой октавы из которого этот слой А был получен?
            // а как чисто идейно должны бы соотноситься сигмы размытия у двух картинок если картинка А была получена из картинки Б простым уменьшением в 2 раза?

            // TODO: переделайте это добавочное размытие с варианта "размываем предыдущий слой" на вариант "размываем самый первый слой октавы до степени размытия сигмы нашего текущего слоя"
            // проверьте - картинки отладочного вывода выглядят один-в-один до/после? (посмотрите на них туда-сюда быстро мигая)

            cv::Mat prevImageLayer = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + prevLayer].clone();
            cv::Size automaticKernelSize = cv::Size(0, 0);

            cv::GaussianBlur(prevImageLayer, prevImageLayer, automaticKernelSize, sigma, sigma);

            gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer] = prevImageLayer;
        }
    }

    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 0; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "pyramid/o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer]);
            // TODO: какие ожидания от картинок можно придумать? т.е. как дополнительно проверить что работает разумно?
            // спойлер: подуймайте с чем должна визуально совпадать картинка из октавы? может быть с какой-то из картинок с предыдущей октавы? с какой? как их визуально сверить ведь они разного размера? 
        }
    }

    DoGPyramid.resize(NOCTAVES * OCTAVE_DOG_IMAGES);


    // строим пирамиду разниц гауссиан слоев (Difference of Gaussian, DoG), т.к. вычитать надо из слоя слой в рамках одной и той же октавы - то есть приятный параллелизм на уровне октав

    for (i32 octave: range(NOCTAVES)) {
        for (i32 layer: range(OCTAVE_DOG_IMAGES)) {
            i32 dogIndex = octave * OCTAVE_DOG_IMAGES + layer;
            auto gaussPrev = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer];
            auto gaussCur = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer + 1];
            cv::Mat diff = gaussCur - gaussPrev;
            DoGPyramid[dogIndex] = diff;
        }
    }

    // нам нужны padding-картинки по краям октавы чтобы извлекать экстремумы, но в статье предлагается не s+2 а s+3: [lowe04] We must produce s + 3 images in the stack of blurred images for each octave, so that final extrema detection covers a complete octave
    // TODO: почему OCTAVE_GAUSSIAN_IMAGES=(OCTAVE_NLAYERS + 3) а не например (OCTAVE_NLAYERS + 2)?

    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 0; layer < OCTAVE_DOG_IMAGES; ++layer) {
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "pyramidDoG/o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer]);
            // TODO: какие ожидания от картинок можно придумать? т.е. как дополнительно проверить что работает разумно?
            // спойлер: подуймайте с чем должна визуально совпадать картинка из октавы DoG? может быть с какой-то из картинок с предыдущей октавы? с какой? как их визуально сверить ведь они разного размера? 
        }
    }
}

namespace {
    float parabolaFitting(float x0, float x1, float x2) {
        rassert((x1 >= x0 && x1 >= x2) || (x1 <= x0 && x1 <= x2), 12541241241241);

        // a*0^2+b*0+c=x0
        // a*1^2+b*1+c=x1
        // a*2^2+b*2+c=x2

        // c=x0
        // a+b+x0=x1     (2)
        // 4*a+2*b+x0=x2 (3)

        // (3)-2*(2): 2*a-y0=y2-2*y1; a=(y2-2*y1+y0)/2
        // (2):       b=y1-y0-a
        float a = (x2-2.0f*x1+x0) / 2.0f;
        float b = x1 - x0 - a;
        // extremum is at -b/(2*a), but our system coordinate start (i) is at 1, so minus 1
        float shift = - b / (2.0f * a) - 1.0f;
        return shift;
    }
}

void phg::SIFT::findLocalExtremasAndDescribe(const std::vector<cv::Mat> &gaussianPyramid, const std::vector<cv::Mat> &DoGPyramid,
                                             std::vector<cv::KeyPoint> &keyPoints, cv::Mat &desc) {
    std::vector<std::vector<float>> pointsDesc;

    // 3.1 Local extrema detection
    #pragma omp parallel // запустили каждый вычислительный поток процессора
    {
        // каждый поток будет складировать свои точки в свой личный вектор (чтобы не было гонок и не были нужны точки синхронизации)
        std::vector<cv::KeyPoint> thread_points;
        std::vector<std::vector<float>> thread_descriptors;

        for (size_t octave = 0; octave < NOCTAVES; ++octave) {
            double octave_downscale = pow(2.0, octave);
            for (size_t layer = 1; layer + 1 < OCTAVE_DOG_IMAGES; ++layer) {
                const cv::Mat prev = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer - 1];
                const cv::Mat cur  = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer];
                const cv::Mat next = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer + 1];
                const cv::Mat DoGs[3] = {prev, cur, next};

                // теперь каждый поток обработает свой кусок картинки 
                #pragma omp for
                for (ptrdiff_t j = 1; j < cur.rows - 1; ++j) {
                    for (ptrdiff_t i = 1; i + 1 < cur.cols; ++i) {
                        bool is_max = true;
                        bool is_min = true;
                        float center = DoGs[1].at<float>(j, i);
                        for (int dz = -1; dz <= 1 && (is_min || is_max); ++dz) {
                            for (int dy = -1; dy <= 1 && (is_min || is_max); ++dy) {
                                for (int dx = -1; dx <= 1 && (is_min || is_max); ++dx) {
                                    if (DoGs[1 + dz].at<float>(j + dy, i + dx) > center) {
                                        is_max = false;
                                    }
                                    if (DoGs[1 + dz].at<float>(j + dy, i + dx) < center) {
                                        is_min = false;
                                    }
                                }
                            }
                        }
                        bool is_extremum = (is_min || is_max);

                        if (!is_extremum)
                            continue; // очередной элемент cascade filtering, если не экстремум - сразу заканчиваем обработку этого пикселя

                        // 4 Accurate keypoint localization
                        cv::KeyPoint kp;
                        float dx = 0.0f;
                        float dy = 0.0f;
                        float dvalue = 0.0f;
                        // TODO сделать субпиксельное уточнение (хотя бы через параболу-фиттинг независимо по оси X и оси Y, но лучше через честный ряд Тейлора, матрицу Гессе и итеративное смещение если экстремум оказался в соседнем пикселе)
#if SUBPIXEL_FITTING_ENABLE // такие тумблеры включающие/выключающие очередное улучшение алгоритма позволяют оценить какой вклад эта фича вносит в качество результата если в рамках уже готового алгоритма попробовать ее включить/выключить
                        {
                            // TODO
                        }
#endif
                        // TODO сделать фильтрацию слабых точек по слабому контраст
                        float contrast = center + dvalue;
                        if (contrast < contrast_threshold / OCTAVE_NLAYERS) // TODO почему порог контрастности должен уменьшаться при увеличении числа слоев в октаве?
                            continue;

                        kp.pt = cv::Point2f((i + 0.5 + dx) * octave_downscale, (j + 0.5 + dy) * octave_downscale);

                        kp.response = fabs(contrast);

                        const double k = pow(2.0, 1.0 / OCTAVE_NLAYERS); // [lowe04] k = 2^{1/s} а у нас s=OCTAVE_NLAYERS
                        double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
                        kp.size = 2.0 * sigmaCur * 5.0;

                        // 5 Orientation assignment
                        cv::Mat img = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + 0].clone();
//                        cv::Mat img = gaussianPyramid[0].clone();
                        std::vector<float> votes;
                        float biggestVote;
                        int oriRadius = (int) (ORIENTATION_WINDOW_R * (1.0 + k * (layer - 1)));
                        if (!buildLocalOrientationHists(img, i, j, oriRadius, votes, biggestVote))
                            continue;

                        for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
                            float prevValue = votes[(bin + ORIENTATION_NHISTS - 1) % ORIENTATION_NHISTS];
                            float value = votes[bin];
                            float nextValue = votes[(bin + 1) % ORIENTATION_NHISTS];
                            if (value > prevValue && value > nextValue && votes[bin] > biggestVote * ORIENTATION_VOTES_PEAK_RATIO) {
                                // TODO добавьте уточнение угла наклона - может помочь определенная выше функция parabolaFitting(float x0, float x1, float x2)
                                kp.angle = (bin + 0.5) * (2 * M_PI / ORIENTATION_NHISTS);
                                rassert(kp.angle >= 0.0 && kp.angle <= 2 * M_PI, 123512412412);

                                std::vector<float> descriptor;
                                double descrSampleRadius = (DESCRIPTOR_SAMPLE_WINDOW_R * (1.0 + k * (layer - 1)));
                                if (!buildDescriptor(img, kp.pt.x, kp.pt.y, descrSampleRadius, kp.angle, descriptor))
                                    continue;
                                thread_points.push_back(kp);
                                thread_descriptors.push_back(descriptor);
                            }
                        }
                    }
                }
            }
        }

        // в критической секции объединяем все массивы детектированных точек
        #pragma omp critical
        {
            keyPoints.insert(keyPoints.end(), thread_points.begin(), thread_points.end());
            pointsDesc.insert(pointsDesc.end(), thread_descriptors.begin(), thread_descriptors.end());
        }
    }

    rassert(pointsDesc.size() == keyPoints.size(), 12356351235124);
    desc = cv::Mat(pointsDesc.size(), DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, CV_32FC1);
    for (size_t j = 0; j < pointsDesc.size(); ++j) {
        rassert(pointsDesc[j].size() == DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, 1253351412421);
        for (size_t i = 0; i < pointsDesc[j].size(); ++i) {
            desc.at<float>(j, i) = pointsDesc[j][i];
        }
    }
//    std::ofstream f("descs.txt");
//    f << "rows ";
//    for (int i = 0; i <= pointsDesc[0].size(); ++i) {
//        f << std::setw(20) << i;
//    }
//    f << std::endl;
//    for (int j = 0; j < pointsDesc.size(); ++j) {
//        f << "row " << j << ":" << std::endl;
//        f << "arr: ";
//        for (i32 i: range(pointsDesc[j].size())) {
//            f << std::setw(20) << pointsDesc[j][i];
//        }
//        f << std::endl;
//        f << "mat: ";
//        for (i32 i: range(pointsDesc[j].size())) {
//            f << std::setw(20) << desc.at<float>(j, i);
//        }
//        f << std::endl;
//        f << std::endl;
//    }
//    std::cout << "Done\n";
}

bool phg::SIFT::buildLocalOrientationHists(const cv::Mat &img, size_t i, size_t j, size_t radius,
                                           std::vector<float> &votes, float &biggestVote) {
    // 5 Orientation assignment
    votes.resize(ORIENTATION_NHISTS, 0.0f);
    biggestVote = 0.0;

    if (i-1 < radius - 1 || i+1 + radius - 1 >= img.cols || j-1 < radius - 1 || j+1 + radius - 1 >= img.rows)
        return false;

    float sum[ORIENTATION_NHISTS] = {0.0f};

    for (size_t y = j - radius + 1; y < j + radius; ++y) {
        for (size_t x = i - radius + 1; x < i + radius; ++x) {
            float dy = img.at<float>(y + 1, x) - img.at<float>( y - 1, x);
            float dx = img.at<float>(y, x + 1) - img.at<float>(y, x - 1);
            float magnitude = sqrtf(dx * dx + dy * dy);
            float orientation = atan2f(dy, dx) + M_PI / 2.f;
            while (orientation < 0) orientation += 2 * M_PI;
            while (orientation >= 2 * M_PI) orientation -= 2 * M_PI;
            if (!(0 <= orientation && orientation < 2 * M_PI)) {
                raise(SIGINT);
            }
            rassert(0 <= orientation && orientation < 2 * M_PI, 5361615612);
            static_assert(360 % ORIENTATION_NHISTS == 0, "Inappropriate bins number!");
            size_t bin = (size_t)((orientation / (2 * M_PI)) * ORIENTATION_NHISTS);
            rassert(bin < ORIENTATION_NHISTS, 361236315613);
            sum[bin] += magnitude;
            // TODO может быть сгладить получившиеся гистограммы улучшит результат? 
        }
    }

    for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
        votes[bin] = sum[bin];
        biggestVote = std::max(biggestVote, sum[bin]);
    }

    return true;
}

cv::Point2f rotate2d(cv::Point2f inPoint, float angRad) {
    cv::Point2f outPoint;
    outPoint.x = std::cos(angRad) * inPoint.x - std::sin(angRad) * inPoint.y;
    outPoint.y = std::sin(angRad) * inPoint.x + std::cos(angRad) * inPoint.y;
    return outPoint;
}

bool phg::SIFT::buildDescriptor(const cv::Mat &img, float px, float py, double descrSampleRadius, float angle,
                                std::vector<float> &descriptor) {
//    angle = 0.f; // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    descriptor.resize(DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, 0.0f);
    // descriptor is DESC_SIZE x DESC_SIZE blocks, within each we conduct a vote for in DESCRIPTOR_SAMPLES_N x DESCRIPTOR_SAMPLES_N subsections
    float const boxBeginX = px - descrSampleRadius;
    float const boxBeginy = py - descrSampleRadius;
    float const blockSize = 2 * descrSampleRadius / DESCRIPTOR_SIZE;
    float const subsecSize = blockSize / (DESCRIPTOR_SAMPLES_N + 1);
    cv::Point2f center{px, py};
//    bool logCondition = ((i32)px == 231 && (i32)py == 223) || ((i32)px == 233 && (i32)py == 287);
    bool logCondition = false;
    if (logCondition) {
        std::cout << "pt (" << px << ", " << py << ") angle = " << angle << std::endl;
    }
    for (i32 yBlock: range(DESCRIPTOR_SIZE)) {
        for (i32 xBlock: range(DESCRIPTOR_SIZE)) {
            for (i32 xSubsec: range(DESCRIPTOR_SAMPLES_N)) {
                for (i32 ySubsec: range(DESCRIPTOR_SAMPLES_N)) {
                    cv::Point2f p{
                        boxBeginX + xBlock * blockSize + (xSubsec + 1) * subsecSize,
                        boxBeginy + yBlock * blockSize + (ySubsec + 1) * subsecSize};
                    p = center + rotate2d((p - center), angle);
                    i32 x = (i32) p.x;
                    i32 y = (i32) p.y;

                    if (!(0 <= x && x <= img.cols && 0 <= y && y <= img.rows)) {
                        continue;
                    }
                    float curValue = img.at<float>(y, x);
                    float dy = img.at<float>(y + 1, x) - img.at<float>(y - 1, x);
                    float dx = img.at<float>(y, x + 1) - img.at<float>(y, x - 1);
                    float magnitude = sqrtf(dx * dx + dy * dy);
                    float orientation = atan2f(dy, dx) - angle;
                    while (orientation < 0) orientation += 2 * M_PI;
                    while (orientation >= 2 * M_PI) orientation -= 2 * M_PI;
                    if (logCondition) {
                        std::cout << "\tat pixel point (" << x << ", " << y << ")" << std::endl;
                        std::cout << "\tat real point (" << p.x << ", " << p.y << ")" << std::endl;
                        std::cout << "\t\tpixel offset (" << x - (i32)px << ", " << y - (i32) py<< ")" << std::endl;
                        std::cout << "\t\treal offset (" << p.x - px << ", " << p.y - py << ")" << std::endl;
                        std::cout << "\t\tdy =" << dy << std::endl;
                        std::cout << "\t\tdx =" << dx << std::endl;
                        std::cout << "\t\traw orientation " << atan2f(dy, dx) << std::endl;
                        std::cout << "\t\torientation " << orientation << std::endl;
                        std::cout << "\t\tmagnitude " << magnitude << std::endl;

                    }
                    rassert(0 <= orientation && orientation < 2 * M_PI, 5361615612);
//                    static_assert(360 % ORIENTATION_NHISTS == 0, "Inappropriate bins number!");
                    size_t bin = (size_t)((orientation / (2 * M_PI)) * DESCRIPTOR_NBINS);
                    rassert(bin < DESCRIPTOR_NBINS, 361236315613);
                    descriptor[yBlock * (DESCRIPTOR_SIZE * DESCRIPTOR_NBINS) + xBlock * DESCRIPTOR_NBINS + bin] += magnitude;
                }
            }
        }
    }
    return true;
}
