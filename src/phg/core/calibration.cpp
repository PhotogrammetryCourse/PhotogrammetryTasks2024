#include <phg/sfm/defines.h>
#include "calibration.h"


phg::Calibration::Calibration(int width, int height)
    : width_(width)
      , height_(height)
      , cx_(0)
      , cy_(0)
      , k1_(0)
      , k2_(0) {
    // 50mm guess

    double diag_35mm = 36.0 * 36.0 + 24.0 * 24.0;
    double diag_pix = (double) width * (double) width + (double) height * (double) height;

    f_ = 50.0 * std::sqrt(diag_pix / diag_35mm);
}

cv::Matx33d phg::Calibration::K() const {
    return {f_, 0., cx_ + width_ * 0.5, 0., f_, cy_ + height_ * 0.5, 0., 0., 1.};
}

int phg::Calibration::width() const {
    return width_;
}

int phg::Calibration::height() const {
    return height_;
}

double phg::Calibration::r(const cv::Vec2d &pixel) const {
    return std::hypot(pixel[0], pixel[1]);
}

double phg::Calibration::distorsionCoef(const double r) const {
    return 1 + k1_ * std::pow(r, 2) + k2_ * std::pow(r, 4);
}

cv::Vec3d phg::Calibration::project(const cv::Vec3d &point) const {
    double x = point[0] / point[2];
    double y = point[1] / point[2];

    double coef = distorsionCoef(r({x, y}));

    x *= coef;
    y *= coef;

    x *= f_;
    y *= f_;

    x += cx_ + width_ * 0.5;
    y += cy_ + height_ * 0.5;

    return cv::Vec3d(x, y, 1.0);
}

cv::Vec3d phg::Calibration::unproject(const cv::Vec2d &pixel) const {
    double x = pixel[0] - cx_ - width_ * 0.5;
    double y = pixel[1] - cy_ - height_ * 0.5;

    x /= f_;
    y /= f_;

    double coef = distorsionCoef(r({x, y}));

    return cv::Vec3d(x / coef, y / coef, 1.0);
}
