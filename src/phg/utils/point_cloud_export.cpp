//
//  DataExporter.cpp
//
//  Created by Cedric Leblond Menard on 16-06-27.
//  Copyright © 2016 Cedric Leblond Menard. All rights reserved.
//
#include "point_cloud_export.h"

#include <stdio.h>
#include <string>
#include "opencv2/core/core.hpp"
#include <fstream>
#include <iomanip>


namespace {

// Determine endianness of platform
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

    enum FileFormat {PLY_ASCII, PLY_BIN_BIGEND, PLY_BIN_LITEND};

    class DataExporter  {
    private:
        std::string filename = "";
        FileFormat format;
        cv::Mat data;
        cv::Mat colors;
        std::ofstream filestream;
        unsigned long numElem;

    public:
        DataExporter(cv::Mat outputData, cv::Mat outputColor, std::string outputfile, FileFormat outputformat);
        ~DataExporter();
        void exportToFile();

    };

// Dictionary for file format
    const char* enum2str[] = {
            "ascii",
            "binary_big_endian",
            "binary_little_endian"
    };

// Function to reverse float endianness
    float ReverseFloat( const float inFloat )
    {
        float retVal;
        char *floatToConvert = ( char* ) & inFloat;
        char *returnFloat = ( char* ) & retVal;

        // swap the bytes into a temporary buffer
        returnFloat[0] = floatToConvert[3];
        returnFloat[1] = floatToConvert[2];
        returnFloat[2] = floatToConvert[1];
        returnFloat[3] = floatToConvert[0];

        return retVal;
    }

// Check system endianness
    bool isLittleEndian()
    {
        uint16_t number = 0x1;
        char *numPtr = (char*)&number;
        return (numPtr[0] == 1);
    }

    DataExporter::DataExporter(cv::Mat outputData, cv::Mat outputColor, std::string outputfile, FileFormat outputformat) :
            filename(outputfile), format(outputformat), data(outputData), colors(outputColor)
    {
        // MARK: Init
        // Opening filestream
        switch (format) {
            case FileFormat::PLY_BIN_LITEND:
            case FileFormat::PLY_BIN_BIGEND:
                filestream.open(filename, std::ios::out | std::ios::binary);
                break;
            case FileFormat::PLY_ASCII:
                filestream.open(filename,std::ios::out);
                break;
        }

        // Calculating number of elements
        CV_Assert(data.total() == colors.total());      // If not same size, assert
        CV_Assert(data.type() == CV_32FC3 &&
                  colors.type() == CV_8UC3);            // If not 3 channels and good type
        CV_Assert(data.isContinuous() &&
                  colors.isContinuous());               // If not continuous in memory

        numElem = data.total();

    }

    void DataExporter::exportToFile() {
        // MARK: Header writing
        filestream << "ply" << std::endl <<
                   "format " << enum2str[format] << " 1.0" << std::endl <<
                   "comment file created using code by Cedric Menard" << std::endl <<
                   "element vertex " << numElem << std::endl <<
                   "property float x" << std::endl <<
                   "property float y" << std::endl <<
                   "property float z" << std::endl <<
                   "property uchar red" << std::endl <<
                   "property uchar green" << std::endl <<
                   "property uchar blue" << std::endl <<
                   "end_header" << std::endl;

        // MARK: Data writing

        // Pointer to data
        const float* pData = data.ptr<float>(0);
        const unsigned char* pColor = colors.ptr<unsigned char>(0);
        const unsigned long numIter = 3*numElem;                            // Number of iteration (3 channels * numElem)
        const bool hostIsLittleEndian = isLittleEndian();

        float_t bufferXYZ;                                                 // Coordinate buffer for float type

        // Output format switch
        switch (format) {
            case FileFormat::PLY_BIN_BIGEND:
                // Looping through all
                for (unsigned long i = 0; i<numIter; i+=3) {                                // Loop through all elements
                    for (unsigned int j = 0; j<3; j++) {                                    // Loop through 3 coordinates
                        if (hostIsLittleEndian) {
                            bufferXYZ = ReverseFloat(pData[i+j]);                        // Convert from host to network (Big endian)
                            filestream.write(reinterpret_cast<const char *>(&bufferXYZ),    // Non compiled cast to char array
                                             sizeof(bufferXYZ));
                        } else {
                            bufferXYZ = pData[i+j];
                            filestream.write(reinterpret_cast<const char *>(&bufferXYZ),    // Non compiled cast to char array
                                             sizeof(bufferXYZ));
                        }
                    }
                    for (int j = 2; j>=0; j--) {
                        // OpenCV uses BGR format, so the order of writing is reverse to comply with the RGB format
                        filestream.put(pColor[i+j]);                                        // Loop through RGB
                    }
                }

                break;

            case FileFormat::PLY_BIN_LITEND:                                                // Assume host as little-endian
                for (unsigned long i = 0; i<numIter; i+=3) {                                // Loop through all elements
                    for (unsigned int j = 0; j<3; j++) {                                    // Loop through 3 coordinates
                        if (hostIsLittleEndian) {
                            filestream.write(reinterpret_cast<const char *>(pData+i+j),     // Non compiled cast to char array
                                             sizeof(bufferXYZ));
                        } else {
                            bufferXYZ = ReverseFloat(pData[i+j]);
                            filestream.write(reinterpret_cast<const char *>(&bufferXYZ), sizeof(bufferXYZ));
                        }
                    }
                    for (int j = 2; j>=0; j--) {
                        // OpenCV uses BGR format, so the order of writing is reverse to comply with the RGB format
                        filestream.put(pColor[i+j]);                                        // Loop through RGB
                    }
                }

                break;

            case FileFormat::PLY_ASCII:
                for (unsigned long i = 0; i<numIter; i+=3) {                            // Loop through all elements
                    for (unsigned int j = 0; j<3; j++) {                                // Loop through 3 coordinates
                        filestream << std::setprecision(9) << pData[i+j] << " ";
                    }
                    for (int j = 2; j>=0; j--) {
                        // OpenCV uses BGR format, so the order of writing is reverse to comply with the RGB format
                        filestream << (unsigned short)pColor[i+j] << (j==0?"":" ");                     // Loop through RGB
                    }
                    filestream << std::endl;                                            // End if element line
                }
                break;

            default:
                break;
        }
    }

    DataExporter::~DataExporter() {
        filestream.close();
    }
}

void phg::exportPointCloud(const std::vector<cv::Vec3d> &point_cloud, const std::string &path, const std::vector<cv::Vec3b> &point_cloud_colors_bgr)
{
    if (!point_cloud_colors_bgr.empty() && point_cloud_colors_bgr.size() != point_cloud.size()) {
        throw std::runtime_error("!point_cloud_colors_bgr.empty() && point_cloud_colors_bgr.size() != point_cloud.size()");
    }

    cv::Mat coords3d(1, point_cloud.size(), CV_32FC3);
    cv::Mat img(1, point_cloud.size(), CV_8UC3);

    for (int i = 0; i < (int) point_cloud.size(); ++i) {
        cv::Vec3f x = point_cloud[i];
        cv::Vec3b c = point_cloud_colors_bgr.empty() ? cv::Vec3b{200, 200, 200} : point_cloud_colors_bgr[i];
        coords3d.at<cv::Vec3f>(0, i) = x;
        img.at<cv::Vec3b>(0, i) = c;
    }

    DataExporter data(coords3d, img, path, FileFormat::PLY_BIN_BIGEND);
    data.exportToFile();
}
