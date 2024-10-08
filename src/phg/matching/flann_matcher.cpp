#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher() {
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(45);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc) {
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch> > &matches,
                                 int k) const {
    matches.clear();
    matches.resize(query_desc.rows);
    std::vector<std::vector<float> > dists(query_desc.rows, std::vector<float>(k));
    std::vector<std::vector<int> > inds(query_desc.rows, std::vector<int>(k));
    for (int r = 0; r < query_desc.rows; r++) {
        flann_index->knnSearch(query_desc.row(r), inds[r], dists[r], k, *search_params);
    }
    for (int r = 0; r < query_desc.rows; r++) {
        for (int c = 0; c < k; c++) {
            cv::DMatch match(r, inds[r][c], dists[r][c]);
            matches[r].push_back(match);
        }
    }
}
