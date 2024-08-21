#include "flann_matcher.h"
#include "flann_factory.h"
#include <iostream>

phg::FlannMatcher::FlannMatcher() {
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(45);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc) {
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const {
    int n_descs = query_desc.rows;
    matches.assign(n_descs, {});
    for (int i = 0; i < n_descs; i++) {
        std::vector<int> indices;
        std::vector<float> dists;
        flann_index->knnSearch(query_desc.row(i), indices, dists, k, *search_params);
        for (int j = 0; j < k; j++) {
            matches[i].push_back(cv::DMatch{i, indices[j], dists[j]});
        }
    }
}
