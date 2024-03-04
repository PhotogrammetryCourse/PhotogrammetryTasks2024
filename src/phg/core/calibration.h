#pragma once

#include <opencv2/core.hpp>

namespace phg {

    struct Calibration {

        Calibration(int width, int height);

        cv::Vec3d project(const cv::Vec3d &point) const;
        cv::Vec3d unproject(const cv::Vec2d &pixel) const;

        cv::Matx33d K() const;

        int width() const;
        int height() const;

    private:

        double f_;
        double cx_;
        double cy_;
        int width_;
        int height_;
    };

}
