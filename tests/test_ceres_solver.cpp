#include <gtest/gtest.h>

#include <random>

#include <ceres/ceres.h>
#include <ceres/rotation.h>


//______________________________________________________________________________________________________________________
// Пример из http://ceres-solver.org/nnls_modeling.html#introduction
// 0.5*(10-x)^2
//______________________________________________________________________________________________________________________

class CostFunctor1 {
public:
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = 10.0 - x[0];
        return true;
    }
};

TEST (CeresSolver, HelloWorld1) {
    double initial_x = 5.0;
    double cur_x = initial_x;

    // Создаем функтор
    CostFunctor1 *f = new CostFunctor1();
    // Формулируем Cost Function (она еще называется невязкой - Residual)
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor1, // тип функтора
                                                                         1, // количество невязок
                                                                         1> // число параметров в каждом блоке
                                                                         (f);
    ceres::LossFunction* loss_function = new ceres::TrivialLoss(); // тривиальная функция потерь (т.е. просто квадратичная норма, т.е. cost_function(x)^2)

    // Формулируем задачу
    ceres::Problem problem;
    problem.AddResidualBlock(cost_function, loss_function, &cur_x);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR; // Почему Conjugate gradients не срабатывают?
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    // Параметры и якобианы - это указатели-т.е. список (столько сколько блоков невязки, в нашем случае - 1 блок) из указателей-т.е. списков (каждый такой длинны, сколько параметров в этом отдельном блоке, у нас в единственном блоке единственный параметр - x)
    const int N_RESIDUAL_BLOCKS = 1;
    double* params[N_RESIDUAL_BLOCKS]; // для каждого блока невязки: параметр такого размера, сколько у этого блока параметров (у нас у единственного блока единственный параметр - x)
    double* jacobians[N_RESIDUAL_BLOCKS]; // для каждого блока невязки: якобиан размера=ЧислоНевязок*ЧислоПараметровВЭтомБлоке

    double initial_residual = 0.0;
    double initial_jacobian = 0.0;
    params[0] = &initial_x; // для нашего единственного блока невязки (ResidualBlock) заполняем параметры (один x)
    jacobians[0] = &initial_jacobian; // для нашего единственного блока заполняем якобиан (размерности один т.к. одна переменная x)
    // подробнее про параметры и якобианы можно посмотреть в документации Evaluate:
    cost_function->Evaluate(params, &initial_residual, jacobians);

    double final_residual = 0.0;
    double final_jacobian = 0.0;
    params[0] = &cur_x;
    jacobians[0] = &final_jacobian;
    cost_function->Evaluate(params, &final_residual, jacobians);

    std::cout << "x:     " << initial_x        << " -> " << cur_x << std::endl;
    std::cout << "f(x):  " << initial_residual << " -> " << final_residual << std::endl;
    std::cout << "f'(x): " << initial_jacobian << " -> " << final_jacobian << std::endl;
    // TODO 1: почему результирующая производная не ноль? мы ведь должны были сойтись в минимуме функции 0.5*(10-x)^2

    ASSERT_NEAR(cur_x, 10.0, 1e-6);
}

//______________________________________________________________________________________________________________________
// Пусть есть два фиксированных 3D объекта - параболоид и прямая.
// Хотим найти их точку пересечения. Да, это из пушки по воробьям, но как иллюстрация для тренировки - полезно :)
// Значит у нас две невязки (Residual) - расстояние до параболоида и до прямой.
// И всего один блок параметров состоящий из трех чисел - (x,y,z) - координаты точки (искомого пересечения).
//______________________________________________________________________________________________________________________

// Сначала надо определить функтор находящий расстояние до нашей фиксированной прямой:
class DistanceToFixedLine {
public:
    DistanceToFixedLine(const double linePoint[3], const double lineDirection[3]) {
        double normal_len2 = 0.0;
        for (int d = 0; d < 3; ++d) {
            normal_len2 += lineDirection[d] * lineDirection[d];
        }
        double normal_len = sqrt(normal_len2);

        for (int d = 0; d < 3; ++d) {
            this->linePoint[d] = linePoint[d];
            this->lineDirection[d] = lineDirection[d] / normal_len;
        }
    }
    template <typename T>
    bool operator()(const T* const queryPoint, T* residual) const {
        // Расстояние от точки-запроса queryPoint до прямой можно найти через векторное произведение: |(queryPoint-linePoint) x lineDirection|
        // Важно делать все вычисления в T, чтобы ceres-solver мог подставив туда вместо double - Jet - автоматически посчитать якобиан.
        // Хорошее правило - в функторе никогда не должно быть double переменных (например linePoint[3] и lineDirection[3] мы кастим к T).
        T linePointToQuery[3];
        T n[3];
        for (int d = 0; d < 3; ++d) {
            linePointToQuery[d] = queryPoint[d] - linePoint[d]; // здесь происходит неявное преобразование double linePoint[d] к T-типу
            n[d] = (T) lineDirection[d]; // здесь происходит преобразование double lineDirection[d] к T-типу (который может быть как double, так и Jet)
        }
        T crossProduct[3];
        ceres::CrossProduct<T>(linePointToQuery, n, crossProduct);

        T distance = ceres::sqrt(ceres::DotProduct(crossProduct, crossProduct));
        residual[0] = distance;
        return true;
    }
protected:
    double linePoint[3];
    double lineDirection[3];
};

// Теперь надо определить функтор находящий расстояние до нашего фиксированного упрощенного параболоида вида: z = a*(x-centerX)^2 + b*(y-centerY)^2 + centerZ
class ResidualToParaboloid {
public:
    ResidualToParaboloid(const double center[3], const double a, const double b) : a(a), b(b) {
        for (int d = 0; d < 3; ++d) {
            this->center[d] = center[d];
        }
    }
    template <typename T>
    bool operator()(const T* const queryPoint, T* residual) const {
        // Давайте попробуем искать не истинное расстояние а более простое для рассчета расстояние по оси z
        // Наш упрощенный параболоид имеет вид: z = a*(x-centerX)^2 + b*(y-centerY)^2 + centerZ
        // Помните что нельзя использовать функции оперирующие double, подходят только те операции для которых есть перегрузка для T=Jet
        // Поэтому например для вычисления квадрата - можно просто перемножить T-переменные, а для вычисления произвольной степени - ceres::pow(x, y)
        T dx = queryPoint[0] - center[0];
        T dy = queryPoint[1] - center[1];
        residual[0] = a*dx*dx + b*dy*dy - center[2];
        return true;
    }
protected:
    double center[3];
    double a;
    double b;
};

TEST (CeresSolver, HelloWorld2) {
    // Две невязки: расстояние до 3D прямой и расстояние до параболоида, иначе говоря мы ищем точку их пересечения

    // Формулируем обе Cost Function
    const double line_point[3]  = {10.0, 5.0, 0.0};
    const double line_direction[3] = {0.0, 0.0, 1.0};
    ceres::CostFunction* line_cost_function = new ceres::AutoDiffCostFunction<DistanceToFixedLine,
            1, // количество невязок (размер искомого residual массива переданного в функтор, т.е. размерность искомой невязки)
            3> // число параметров в каждом блоке параметров, у нас один блок параметров из трех координат точек
            (new DistanceToFixedLine(line_point, line_direction));

    const double paraboloid_center[3] = {5.0, 10.0, 100.0};
    const double paraboloid_a = 2.0;
    const double paraboloid_b = 2.0;
    ceres::CostFunction* paraboloid_cost_function = new ceres::AutoDiffCostFunction<ResidualToParaboloid, 1, 3>
            (new ResidualToParaboloid(paraboloid_center, paraboloid_a, paraboloid_b));

    return; // TODO 2 удалите эту строку, затем
    // нарисуйте систему координат на бумажке чтобы найти координаты пересечения прямой и параболоида (параболоид и прямые - простые, поэтому пересечь их довольно просто)
    // и подставьте найденные координаты эталонного ответа в массив:
    const double expected_point_solution[3] = {-1000.0, -1000.0, -1000.0};
    {
        // Проверим что невязка эталонного решения нулевая для обоих функций невязки
        const double* params[1];
        double residual;
        params[0] = expected_point_solution;

        residual = -1.0;
        line_cost_function->Evaluate(params, &residual, NULL);
        ASSERT_NEAR(residual, 0.0, 1e-6);

        residual = -1.0;
        paraboloid_cost_function->Evaluate(params, &residual, NULL);
        ASSERT_NEAR(residual, 0.0, 1e-6);
    }

    // Создаем единственныйы блок параметров: [x, y, z] - точка пересечения которую мы оптимизируем, стартуем из нуля
    double point[3] = {0.0, 0.0, 0.0};

    {
        // Проверим что невязка исходного приближения - очень большая для обеих невязок
        const double* params[1];
        double residual;
        params[0] = point;

        residual = -1.0;
        line_cost_function->Evaluate(params, &residual, NULL);
        ASSERT_GT(abs(residual), 1.0);

        residual = -1.0;
        paraboloid_cost_function->Evaluate(params, &residual, NULL);
        ASSERT_GT(abs(residual), 1.0);
    }

    // Формулируем задачу
    ceres::Problem problem;
    problem.AddResidualBlock(line_cost_function, new ceres::TrivialLoss(), point);
    problem.AddResidualBlock(paraboloid_cost_function, new ceres::TrivialLoss(), point);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Found intersection point: (" << point[0] << ", " << point[1] << ", " << point[2] << ")" << std::endl;

    {
        // Проверим что невязка найденного решения нулевая для обоих функций невязки
        const double* params[1];
        double residual;
        params[0] = point;

        residual = -1.0;
        line_cost_function->Evaluate(params, &residual, NULL);
        ASSERT_NEAR(residual, 0.0, 1e-6);

        residual = -1.0;
        paraboloid_cost_function->Evaluate(params, &residual, NULL);
        ASSERT_NEAR(residual, 0.0, 1e-6);
    }

    for (int d = 0; d < 3; ++d) {
//        EXPECT_NEAR(point[d], expected_point_solution[d], 1e-4);
        // TODO 3: раскомментируйте^, почему он находит не то что ожидалось?
        // либо мы набагали в коде, либо в аналитическом поиске правильного ответа на бумажке (проверьте вычисления на бумажке)
        // если бага в коде, то первые подозреваемые - две функции невязки (только там есть содержательный код)
        // заметьте что у найденного ответа ошибка только по одной из осей
        // какие невязки должны были противиться этой координате в ответе? обе или какая-то одна?
        // отладьте те функции невязки которые по-хорошему не должны соглашаться на такой ответ - поставьте просто точку остановки чуть выше, там где мы проверяли
        // что невязка найденного решения - нулевая, и найдите где вдруг ваше ожидание большой невязки для этого ответа сталкивается с суровой реальностью баги в коде
        // которая приводит к нулевой невязке
    }

    // TODO 4: если любопытно и хватит времени - можете попросить ceres-solver посчитать якобианы в некоторых точках подобно тому как это делалось в конце теста HelloWorld1
    // и сверить что найденные аналитически на бумажке результаты совпадают (через ASSERT_NEAR)
}

//______________________________________________________________________________________________________________________
// Пусть есть сколько-то шумных замеров (потенциально включающих еще и выбросы), хочется их зафиттить прямой
//______________________________________________________________________________________________________________________

typedef std::array<double, 2> double_2; // то же самое что и "const double point[2]" но компилируется под clang на Mac OS

// Сначала надо определить функтор находящий расстояние от конкретной точки-сэмпла до нашей искомой прямой:
class PointObservationError {
public:
    PointObservationError(const double_2 point) {
        for (int d = 0; d < 2; ++d) {
            samplePoint[d] = point[d];
        }
    }

    template <typename T>
    bool operator()(const T* const line, T* residual) const {
        // Блок параметров - line=[a, b, c] - задает прямую вида ax+by+c=0
        // TODO 5 посчитайте единственную невязку - расстояние от нашей точки-замера до текущего состояния прямой (для извлечения корня, помня про T=Jet, нужно использовать ceres::sqrt):
        // обратите внимание что расстояние лучше оставить знаковым, т.к. тогда эта невязка будет хорошо дифференцироваться при расстоянии около нуля
//        residual[0] = ;
        return true;
    }
protected:
    double samplePoint[2];
};

double calcLineY(double x, const double* abc) {
    double y = -(abc[0] * x + abc[2]) / abc[1];
    return y;
}

double calcDistanceToLine2D(double x, double y, const double* abc) {
    double dist = abc[0] * x + abc[1] * y + abc[2];
    dist /= sqrt(abc[0] * abc[0] + abc[1] * abc[1]);
    return dist;
}

void evaluateLine(const std::vector<double_2> &points, const double* line, double sigma, double &fitted_inliers_fraction, double &mean_inliers_distance);

void evaluateLineFitting(double sigma, double &fitted_inliers_fraction, double &mean_inliers_distance, double outliers_fraction=0.0, bool use_huber=false) {
    const double ideal_line[3] = {0.5, -1.0, 100.0}; // 0.5*x - y + 100 = 0

    const size_t n_points = 1000;
    const size_t n_points_outliers = (size_t) (n_points * outliers_fraction);

    std::vector<double_2> points(n_points);

    std::default_random_engine r(212512512391);

    // Определим кусок-прямоугольник на плоскости в котором будем работать
    double min_x = -sigma * n_points;
    double max_x =  sigma * n_points;
    double min_y = calcLineY(min_x, ideal_line);
    double max_y = calcLineY(max_x, ideal_line);
    if (min_y > max_y) std::swap(min_y, max_y);
    min_y -= sigma * n_points;
    max_y += sigma * n_points;

    std::uniform_real_distribution<double> uniform_x(min_x, max_x); // генерирует случайное значение x в рамках выбранного куска плоскости
    std::uniform_real_distribution<double> uniform_y(min_y, max_y); // генерирует случайное значение y в рамках выбранного куска плоскости (для порождения выбросов)
    std::normal_distribution<double>       sigma_shift(0.0, sigma); // нормальное распределение с учетом выбранной sigma (будем генерировать случайное смещение точки от прямой)

    for (size_t i = 0; i < n_points; ++i) {
        // Создаем случайную точку на прямой
        double x = uniform_x(r);
        double y;

        if (i < n_points - n_points_outliers) {
            // Точка - просто слегка шумная, т.е. ее надо сместить недалеко от прямой

            // Проецируем координату x на идеальную прямую, получили идеальную точку, теперь хотим ее немного сместить
            y = calcLineY(x, ideal_line);

            // Выбираем для точки случайный знаковый (т.е. в ту или иную сторону от прямой) сдвиг (нормальное распределение с выбранной сигмой)
            double shift_distance = sigma_shift(r);

            // Определяем направление сдвига - перпендикуляр к прямой - т.е. просто нормаль
            double line_normal_x = ideal_line[0];
            double line_normal_y = ideal_line[1];
            double line_normal_norm = sqrt(line_normal_x * line_normal_x + line_normal_y * line_normal_y);
            line_normal_x /= line_normal_norm;
            line_normal_y /= line_normal_norm;

            // Смещаем точку с учетом выбранного случайного расстояния (знакового, поэтому в любую из двух сторон можем сместиться)
            x = x + line_normal_x * shift_distance;
            y = y + line_normal_y * shift_distance;
        } else {
            // Точка - выброс далеко отстоящий от прямой, т.е. ее надо сместить в случайное место
            y = uniform_y(r);
        }
        points[i][0] = x;
        points[i][1] = y;
    }

    // Формулируем задачу
    ceres::Problem problem;

    // Создаем единственныйы блок параметров: [a, b, c] - прямая которую мы оптимизируем
    // Стартуем из первого приближения - горизонтальной прямой проходящей через ноль
    double line_params[3] = {0.0, 1.0, 0.0};

    for (size_t i = 0; i < n_points; ++i) {
        // Для каждой точки-замера создаем невязку
        ceres::CostFunction* point_residual = new ceres::AutoDiffCostFunction<PointObservationError,
                1, // количество невязок (размер искомого residual массива переданного в функтор, т.е. размерность искомой невязки, у нас это просто расстояние до прямой)
                3> // число параметров в каждом блоке параметров, у нас один блок параметров (искомая прямая) из трех ее параметров - a, b, c
                (new PointObservationError(points[i]));
        return; // TODO 6 удалите этот return сразу после выполнения TODO 5

        ceres::LossFunction* loss;
        if (use_huber) {
            loss = new ceres::HuberLoss(3.0 * sigma);
        } else {
            loss = new ceres::TrivialLoss();
        }
        // обратите внимание что теперь единственный блок параметров - это параметры описывающие нашу оптимизируемую прямую
        problem.AddResidualBlock(point_residual, loss, line_params);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    std::cout << "Found line: (a=" << line_params[0] << ", b=" << line_params[1] << ", c=" << line_params[2] << ")" << std::endl;

    double threshold = 1e-4 * std::max(std::abs(ideal_line[0]), std::max(std::abs(ideal_line[1]), std::abs(ideal_line[2])));
    if (outliers_fraction > 0.0 && !use_huber) {
        threshold *= 10.0; // ослабляем порог если есть выбросы и мы к ним не устойчивы (не робастны за счет loss-функции (функции потерь) Huber-а)
    }
    for (int d = 0; d < 3; ++d) {
//        ASSERT_NEAR(line_params[d], ideal_line[d], threshold);
        // TODO 7 расскоментируйте сверку найденной прямой и эталонной
        // почему они расходятся? как это можно решить? придумайте хотя бы два способа:
        // - пост-обработкой - как-то поправив параметры прямой перед сверкой (при этом не меняя ее положение в пространстве)
        // - формулировкой задачи - можно сформулировать для ceres-solver задчау так чтобы избавиться от неоднозначности убрав степень свободы, т.е. описав прямую как-то иначе, как?
        // TODO 7 поправьте тест так или иначе (хотя бы пост-процессингом)
    }

    // Оцениваем качество идеальной прямой
    double inliers_fraction, mse;
    evaluateLine(points, ideal_line, sigma, inliers_fraction, mse);
//    ASSERT_GT(inliers_fraction, 0.99); // TODO 8 раскоментируйте, почему эта проверка падает? как поправить?
//    ASSERT_LT(mse, 1.1 * sigma * sigma); // TODO 9 раскомментируйте, почему проверка падает? на каких тестах она падает, на каких проходит? попробуйте отладить рассчет mse_inliers_distance в evaluateLine

    // Оцениваем качество найденной прямой
    evaluateLine(points, line_params, sigma, inliers_fraction, mse);
    if (outliers_fraction == 0 || use_huber) {
        // TODO 10 раскоментируйте обе проверки, почему они падают? в каких тестах? поправьте (в т.ч. подобно тому как было с ослаблением порога выше)
//        ASSERT_GT(inliers_fraction, 0.99);
//        ASSERT_LT(mse, 1.1 * sigma * sigma);
    }
}

void evaluateLine(const std::vector<double_2> &points, const double* line,
                  double sigma, double &fitted_inliers_fraction, double &mse_inliers_distance) {
    size_t n = points.size();
    size_t inliers = 0;
    mse_inliers_distance = 0.0; // mean square error
    for (size_t i = 0; i < n; ++i) {
        double dist = calcDistanceToLine2D(points[i][0], points[i][1], line);
        if (dist <= 3 * sigma) {
            ++inliers;
            mse_inliers_distance += dist * dist;
        }
    }
    fitted_inliers_fraction = 1.0 * inliers / n;
    mse_inliers_distance /= inliers;
}

TEST (CeresSolver, FitLineNoise) {
    const double sigma = 1.0;

    double no_outliers_trivial_loss_inliers;
    double no_outliers_trivial_loss_mean_inliers_distance;
    evaluateLineFitting(sigma, no_outliers_trivial_loss_inliers, no_outliers_trivial_loss_mean_inliers_distance);
}

TEST (CeresSolver, FitLineNoiseAndOutliers) {
    const double sigma = 1.0;
    const double outliers_fraction = 0.20; // 20% outliers

    double trivial_loss_inliers;
    double trivial_loss_mean_inliers_distance;
    evaluateLineFitting(sigma, trivial_loss_inliers, trivial_loss_mean_inliers_distance, outliers_fraction);
}

TEST (CeresSolver, FitLineNoiseAndOutliersWithHuberLoss) {
    const double sigma = 1.0;
    const double outliers_fraction = 0.20; // 20% outliers
    const bool   use_huber = true; // using Huber loss

    double huber_loss_inliers;
    double huber_loss_mean_inliers_distance;
    evaluateLineFitting(sigma, huber_loss_inliers, huber_loss_mean_inliers_distance, outliers_fraction, use_huber);
}
