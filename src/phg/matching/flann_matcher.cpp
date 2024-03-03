#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::Mat neighbours_indices(query_desc.rows, k, CV_32SC1);
    cv::Mat distances(query_desc.rows, k, CV_32FC1);
    flann_index->knnSearch(query_desc, neighbours_indices, distances, k, *search_params);
    for (int query_index = 0; query_index < query_desc.rows; ++query_index) {
        std::vector<cv::DMatch> neighbours;
        for (int neighbour_index = 0; neighbour_index < k; ++neighbour_index) {
            cv::DMatch match(query_index, neighbours_indices.at<int>(query_index, neighbour_index), distances.at<float>(query_index, neighbour_index));
            neighbours.push_back(match);
        }
        matches.push_back(neighbours);
    }
}
