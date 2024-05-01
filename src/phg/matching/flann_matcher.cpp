#include <iostream>
#include "flann_matcher.h"

#include <iomanip>

#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(5);
    search_params = flannKsTreeSearchParams(45);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    const size_t cnt = query_desc.size[0];
    matches.clear();
    matches.resize(cnt);

    #pragma omp for
    for (size_t i = 0; i < cnt; i++) {
        std::vector<int> indices;
        std::vector<float> dists;
        flann_index->knnSearch(query_desc.row(i), indices, dists, k, *search_params);
        for (size_t j = 0; j < k; j++) {
            cv::DMatch dmatch(i, indices[j], dists[j]);
            matches[i].push_back(dmatch);
        }
    }
}
