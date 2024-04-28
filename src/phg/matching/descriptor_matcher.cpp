#include "descriptor_matcher.h"

#include <opencv2/flann/miniflann.hpp>
#include "flann_factory.h"

namespace {

constexpr float RATIO = 0.6f;

size_t intersectionSize(const std::set<int>& query, const std::set<int>& train) {
    size_t ans = 0;

    auto rightIt = train.begin();

    for (int elem : query) {
        while (rightIt != train.end() && *rightIt < elem) {
            ++rightIt;
        }
        if (rightIt != train.end() && *rightIt == elem) {
            ++ans;
        }
    }

    return ans;
}

} // namespace

void phg::DescriptorMatcher::filterMatchesRatioTest(const std::vector<std::vector<cv::DMatch>> &matches,
                                                    std::vector<cv::DMatch> &filtered_matches)
{
    filtered_matches.clear();

    for (const auto& match : matches) {
        if (!match.size()) {
            continue;
        }

        if (match.size() == 1 || match[0].distance < match[1].distance * RATIO) {
            filtered_matches.push_back(match[0]);
        }
    }
}

void phg::DescriptorMatcher::filterMatchesClusters(const std::vector<cv::DMatch> &matches,
                                                   const std::vector<cv::KeyPoint> keypoints_query,
                                                   const std::vector<cv::KeyPoint> keypoints_train,
                                                   std::vector<cv::DMatch> &filtered_matches)
{
    filtered_matches.clear();

    const size_t  total_neighbours  = 5;  // total number of neighbours to test (including candidate)
    const size_t  consistent_matches  = 3;  // minimum number of consistent matches (including candidate)
    const float  radius_limit_scale  = 2.f;  // limit search radius by scaled median

    const int n_matches = matches.size();

    if (n_matches < total_neighbours) {
        throw std::runtime_error("DescriptorMatcher::filterMatchesClusters : too few matches");
    }

    cv::Mat points_query(n_matches, 2, CV_32FC1);
    cv::Mat points_train(n_matches, 2, CV_32FC1);
    for (int i = 0; i < n_matches; ++i) {
        auto pt1 = keypoints_query[matches[i].queryIdx].pt;
        auto pt2 = keypoints_train[matches[i].trainIdx].pt;
        points_query.at<float>(i * 2) = pt1.x;
        points_query.at<float>(i * 2 + 1) = pt1.y;
        points_train.at<float>(i * 2) = pt2.x;
        points_train.at<float>(i * 2 + 1) = pt2.y;
    }
//
//    // размерность всего 2, так что точное KD-дерево
    std::shared_ptr<cv::flann::IndexParams> index_params = flannKdTreeIndexParams(1);
    std::shared_ptr<cv::flann::SearchParams> search_params = flannKsTreeSearchParams(n_matches);

    std::shared_ptr<cv::flann::Index> index_query = flannKdTreeIndex(points_query, index_params);
    std::shared_ptr<cv::flann::Index> index_train = flannKdTreeIndex(points_train, index_params);
//
//    // для каждой точки найти total neighbors ближайших соседей
    cv::Mat indices_query(n_matches, total_neighbours, CV_32SC1);
    cv::Mat distances2_query(n_matches, total_neighbours, CV_32FC1);
    cv::Mat indices_train(n_matches, total_neighbours, CV_32SC1);
    cv::Mat distances2_train(n_matches, total_neighbours, CV_32FC1);

    index_query->knnSearch(points_query, indices_query, distances2_query, total_neighbours, *search_params);
    index_train->knnSearch(points_train, indices_train, distances2_train, total_neighbours, *search_params);
//
//    // оценить радиус поиска для каждой картинки
//    // NB: radius2_query, radius2_train: квадраты радиуса!
    float radius2_query, radius2_train;
    {
        std::vector<double> max_dists2_query(n_matches);
        std::vector<double> max_dists2_train(n_matches);
        for (int i = 0; i < n_matches; ++i) {
            max_dists2_query[i] = distances2_query.at<float>(i, total_neighbours - 1);
            max_dists2_train[i] = distances2_train.at<float>(i, total_neighbours - 1);
        }

        int median_pos = n_matches / 2;
        std::nth_element(max_dists2_query.begin(), max_dists2_query.begin() + median_pos, max_dists2_query.end());
        std::nth_element(max_dists2_train.begin(), max_dists2_train.begin() + median_pos, max_dists2_train.end());

        radius2_query = max_dists2_query[median_pos] * radius_limit_scale * radius_limit_scale;
        radius2_train = max_dists2_train[median_pos] * radius_limit_scale * radius_limit_scale;
    }
//
//    метч остается, если левое и правое множества первых total_neighbors соседей в радиусах поиска(radius2_query, radius2_train) имеют как минимум consistent_matches общих элементов
    for (int i = 0; i < matches.size(); ++i) {
        std::set<int> query_neighbours, train_neighbours;

        for (int j = 0; j < total_neighbours; ++j) {
            if (distances2_query.at<float>(i, j) < radius2_query) {
                query_neighbours.insert(indices_query.at<int>(i, j));
            }
            if (distances2_train.at<float>(i, j) < radius2_train) {
                train_neighbours.insert(indices_train.at<int>(i, j));
            }
        }

        if (intersectionSize(query_neighbours, train_neighbours) >= consistent_matches) {
            filtered_matches.push_back(matches[i]);
        }
    }
}
