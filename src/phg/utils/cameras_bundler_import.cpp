#include "cameras_bundler_import.h"

#include <fstream>

#include <libutils/rasserts.h>

#include <phg/sfm/ematrix.h>


// See abound bundler .out v0.3 format in 'Output format and scene representation' section:
// https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6
void phg::importCameras(const std::string &path,
                 std::vector<matrix34d> &cameras,
                 phg::Calibration &sensor_calibration,
                 std::vector<vector3d> &tie_points,
                 std::vector<phg::Track> &tracks,
                 std::vector<std::vector<cv::KeyPoint>> &keypoints,
                 int downscale,
                 std::vector<cv::Vec3b> *tie_points_colors)
{
    rassert(sensor_calibration.width() > 0, 2391293219022);
    rassert(sensor_calibration.height() > 0, 2391293219023);
    sensor_calibration = phg::Calibration(sensor_calibration.width(), sensor_calibration.height());

    cameras.clear();
    tie_points.clear();
    tracks.clear();
    keypoints.clear();
    if (tie_points_colors != nullptr) {
        tie_points_colors->clear();
    }

    std::ifstream filestream(path);
    
    std::string header_line;
    //# Bundle file v0.3
    std::getline(filestream, header_line);
    if (header_line != "# Bundle file v0.3") {
        throw std::runtime_error("Can't import bundler .out file - unexpected header line! " + header_line);
    }

    //<num_cameras> <num_points>   [two integers]
    size_t ncameras, npoints;
    filestream >> ncameras >> npoints;

    cameras.resize(ncameras);

    tracks.resize(npoints);
    tie_points.resize(npoints);
    if (tie_points_colors != nullptr) {
        tie_points_colors->resize(npoints);
    }

    keypoints.resize(ncameras);

//    <camera1>
//    <camera2>
//    ...
//    <cameraN>
    for (size_t ci = 0; ci < ncameras; ++ci) {
//        <f> <k1> <k2>   [the focal length, followed by two radial distortion coeffs]
//        <R>             [a 3x3 matrix representing the camera rotation]
//        <t>             [a 3-vector describing the camera translation]

//        <f> <k1> <k2>   [the focal length, followed by two radial distortion coeffs]
        double f, k1, k2;
        filestream >> f >> k1 >> k2;
        f /= downscale;

        // Bundler .out file doesn't support cx/cy principal axis shifts - so we should optimize intrinsics parameters without cx/cy (i.e. they should be zero)
        double cx = 0.0;
        double cy = 0.0;

        matrix3d R;
        vector3d O;

//        <R>             [a 3x3 matrix representing the camera rotation]
        matrix3d Rbundler;
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                filestream >> Rbundler(j, i);
            }
        }
        // y axis is looking up in bundler.out (see 'The pixel positions' description below):
        // and we are looking at -Z axis
        for (int i = 0; i < 3; ++i) {
            R(0, i) = Rbundler(0, i);
            R(1, i) = -Rbundler(1, i);
            R(2, i) = -Rbundler(2, i);
        }

//        <t>             [a 3-vector describing the camera translation]
        vector3d T;
        filestream >> T[0] >> T[1] >> T[2];

        O = Rbundler.inv() * (-T);

        cameras[ci] = phg::composeCameraMatrixRO(R, O);

        if (ci == 0) {
            sensor_calibration.f_ = f;
            sensor_calibration.cx_ = cx;
            sensor_calibration.cy_ = cy;
            sensor_calibration.k1_ = k1;
            sensor_calibration.k2_ = k2;
            // NOTE THAT WE DID NOT INITIALIZED WIDTH AND HEIGHT! THIS SHOULD BE DONE FROM CLIENT SIDE CODE!
            rassert(sensor_calibration.width() > 0, 23912932190104);
            rassert(sensor_calibration.height() > 0, 23912932190105);
        } else {
            rassert(sensor_calibration.f_ == f, 239123919099);
            rassert(sensor_calibration.cx_ == cx, 239123919100);
            rassert(sensor_calibration.cy_ == cy, 239123919101);
            rassert(sensor_calibration.k1_ == k1, 239123919102);
            rassert(sensor_calibration.k2_ == k2, 239123919103);
            // Ensuring that sensor is the same for all cameras (we support only this case for simplicity)
        }
    }

//    <point1>
//    <point2>
//    ...
//    <pointM>
    for (size_t pi = 0; pi < npoints; ++pi) {
//        <position>      [a 3-vector describing the 3D position of the point]
//        <color>         [a 3-vector describing the RGB color of the point]
//        <view list>     [a list of views the point is visible in]

//        <position>      [a 3-vector describing the 3D position of the point]
        filestream >> tie_points[pi][0] >> tie_points[pi][1] >> tie_points[pi][2];

//        <color>         [a 3-vector describing the RGB color of the point]
        int r, g, b;
        filestream >> r >> g >> b;
        if (tie_points_colors != nullptr) {
            rassert(tie_points_colors->size() == tracks.size(), 239123921931088);
            std::vector<cv::Vec3b> &colors = *tie_points_colors;
            colors[pi][0] = r;
            colors[pi][1] = g;
            colors[pi][2] = b;
        }

        //        <view list>     [a list of views the point is visible in]
        // The view list begins with the length of the list (i.e., the number of cameras the point is visible in).
        size_t nprojections;
        filestream >> nprojections;
        tracks[pi].img_kpt_pairs.resize(nprojections);
        for (int i = 0; i < nprojections; ++i) {
            //<camera> <key> <x> <y>
            filestream >> tracks[pi].img_kpt_pairs[i].first;
            size_t camera_key = tracks[pi].img_kpt_pairs[i].first;
            rassert(camera_key < ncameras, 2381283190145);

            // <key> the index of the SIFT keypoint where the point was detected in that camera
            filestream >> tracks[pi].img_kpt_pairs[i].second;
            size_t kpt = tracks[pi].img_kpt_pairs[i].second;

            // The pixel positions are floating point numbers in a coordinate system where the origin is the center of the image,
            // the x-axis increases to the right, and the y-axis increases towards the top of the image.
            // Thus, (-w/2, -h/2) is the lower-left corner of the image, and (w/2, h/2) is the top-right corner (where w and h are the width and height of the image).
            float x, y;
            filestream >> x >> y;

            rassert(sensor_calibration.width() > 0, 23912932190159);
            rassert(sensor_calibration.height() > 0, 23912932190160);
            x /= downscale;
            y /= downscale;
            x = x + sensor_calibration.width() / 2.0;
            y = sensor_calibration.height() / 2.0 - y;

            float no_size = -1.0f;
            cv::KeyPoint kp(x, y, no_size);

            if (keypoints[camera_key].size() < kpt + 1) {
                keypoints[camera_key].resize(kpt + 1);
            }
            keypoints[camera_key][kpt] = kp;
        }
    }

    filestream.close();

    rassert(tie_points.size() == tracks.size(), 23921931291152);
}
