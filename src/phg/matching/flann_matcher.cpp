#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
//     параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(45);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    std::vector<std::vector<float>> distances(query_desc.rows, std::vector<float>(k));
    std::vector<std::vector<int>> indices(query_desc.rows, std::vector<int>(k));
    distances.resize(query_desc.rows);
    for (int y = 0; y < query_desc.rows; ++y) {
        flann_index->knnSearch(
                query_desc.rowRange(y, y + 1),
                indices[y], distances[y],
                k, *search_params);
    }
//    postprocesssing
    matches.clear();
    matches.resize(query_desc.rows);
    for (int y = 0; y < query_desc.rows; ++y) {
        for (int i = 0; i < indices[y].size(); ++i) {
            cv::DMatch match;
            match.distance = distances[y][i];
            match.imgIdx = 0;
            match.queryIdx = y;
            match.trainIdx = indices[y][i];
            matches[y].push_back(match);
        }
    }

}
