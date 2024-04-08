#include "cameras_bundler_export.h"

#include <fstream>

#include <libutils/rasserts.h>

#include <phg/sfm/ematrix.h>


// See abound bundler .out v0.3 format in 'Output format and scene representation' section:
// https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6
void phg::exportCameras(const std::string &path,
                 const std::vector<matrix34d> &cameras,
                 size_t ncameras,
                 const phg::Calibration &sensor_calibration,
                 const std::vector<vector3d> &tie_points,
                 const std::vector<phg::Track> &tracks,
                 const std::vector<std::vector<cv::KeyPoint>> &keypoints,
                 int downscale,
                 const std::vector<cv::Vec3b> *tie_points_colors)
{
    rassert(tie_points.size() == tracks.size(), 23921931291023);

    std::ofstream filestream(path);

    //# Bundle file v0.3
    filestream << "# Bundle file v0.3" << std::endl;

    //<num_cameras> <num_points>   [two integers]
    rassert(ncameras <= cameras.size(), 1239129321030);
    size_t npoints = 0;
    for (size_t pi = 0; pi < tracks.size(); ++pi) {
        if (tracks[pi].disabled)
            continue;
        ++npoints;
    }
    filestream << ncameras << " " << npoints << std::endl;

//    <camera1>
//    <camera2>
//    ...
//    <cameraN>
    for (size_t ci = 0; ci < ncameras; ++ci) {
//        <f> <k1> <k2>   [the focal length, followed by two radial distortion coeffs]
//        <R>             [a 3x3 matrix representing the camera rotation]
//        <t>             [a 3-vector describing the camera translation]

//        <f> <k1> <k2>   [the focal length, followed by two radial distortion coeffs]
        filestream << (sensor_calibration.f_ * downscale) << " " << sensor_calibration.k1_ << " " << sensor_calibration.k2_ << std::endl;

        // Bundler .out file doesn't support cx/cy principal axis shifts - so we should optimize intrinsics parameters without cx/cy (i.e. they should be zero)
        rassert(sensor_calibration.cx_ == 0.0, 2932193129031);
        rassert(sensor_calibration.cy_ == 0.0, 2932193129032);

        matrix3d R;
        vector3d O;
        phg::decomposeUndistortedPMatrix(R, O, cameras[ci]);

//        <R>             [a 3x3 matrix representing the camera rotation]
        matrix3d Rbundler;
        // y axis is looking up in bundler.out (see 'The pixel positions' description below):
        // and we are looking at -Z axis
        for (int i = 0; i < 3; ++i) {
            Rbundler(0, i) = R(0, i);
            Rbundler(1, i) = -R(1, i);
            Rbundler(2, i) = -R(2, i);
        }
        
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                filestream << Rbundler(j, i);
                if (i + 1 != 3) {
                    filestream << " ";
                }
            }
            filestream << std::endl;
        }

//        <t>             [a 3-vector describing the camera translation]
        vector3d T = -(Rbundler * O);
        filestream << T[0] << " " << T[1] << " " << T[2] << std::endl;
    }

//    <point1>
//    <point2>
//    ...
//    <pointM>
    for (size_t pi = 0; pi < tracks.size(); ++pi) {
        if (tracks[pi].disabled)
            continue;

//        <position>      [a 3-vector describing the 3D position of the point]
//        <color>         [a 3-vector describing the RGB color of the point]
//        <view list>     [a list of views the point is visible in]

//        <position>      [a 3-vector describing the 3D position of the point]
        filestream << tie_points[pi][0] << " " << tie_points[pi][1] << " " << tie_points[pi][2] << std::endl;

//        <color>         [a 3-vector describing the RGB color of the point]
        if (tie_points_colors == nullptr) {
            filestream << "0 0 0" << std::endl;
        } else {
            rassert(tie_points_colors->size() == tracks.size(), 239123921931088);
            const std::vector<cv::Vec3b> &colors = *tie_points_colors;
            filestream << colors[pi][0] << " " << colors[pi][1] << " " << colors[pi][2] << std::endl;
        }

        //        <view list>     [a list of views the point is visible in]
        // The view list begins with the length of the list (i.e., the number of cameras the point is visible in).
        filestream << tracks[pi].img_kpt_pairs.size() << " ";
        for (int i = 0; i < tracks[pi].img_kpt_pairs.size(); ++i) {
            //<camera> <key> <x> <y>
            size_t camera_key = tracks[pi].img_kpt_pairs[i].first;
            filestream << camera_key << " ";

            // <key> the index of the SIFT keypoint where the point was detected in that camera
            size_t kpt = tracks[pi].img_kpt_pairs[i].second;
            filestream << kpt << " ";

            cv::Vec2f px = keypoints[camera_key][kpt].pt;

            // The pixel positions are floating point numbers in a coordinate system where the origin is the center of the image,
            // the x-axis increases to the right, and the y-axis increases towards the top of the image.
            // Thus, (-w/2, -h/2) is the lower-left corner of the image, and (w/2, h/2) is the top-right corner (where w and h are the width and height of the image).
            double x = px[0] - sensor_calibration.width() / 2.0;
            double y = sensor_calibration.height() / 2.0 - px[1];
            x *= downscale;
            y *= downscale;

            filestream << x << " " << y << " ";
        }
        filestream << std::endl;
    }

    filestream.close();
}
