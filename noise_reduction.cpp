#include "noise_reduction.h"
#include <nanoflann.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>

// KD-Tree Data Structure
struct PointCloud {
    std::vector<std::vector<double>> pts;

    // Returns the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the distance between two points
    inline double kdtree_distance(const double* p1, const size_t idx_p2, size_t /*size*/) const {
        const double d0 = p1[0] - pts[idx_p2][0];
        const double d1 = p1[1] - pts[idx_p2][1];
        return d0 * d0 + d1 * d1;
    }

    // Returns coordinate values
    inline double kdtree_get_pt(const size_t idx, int dim) const { return pts[idx][dim]; }

    // Optional bounding box
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

// Remove noise using KD-Tree-based KNN
void remove_noise_knn(std::vector<double>& distances,
                      std::vector<double>& angles,
                      int k, double std_threshold, double mean_threshold) {
    if (distances.size() < k + 1) return;

    // Convert polar to Cartesian coordinates
    PointCloud cloud;
    for (size_t i = 0; i < distances.size(); ++i) {
        double x = distances[i] * std::cos(angles[i]);
        double y = distances[i] * std::sin(angles[i]);
        cloud.pts.push_back({x, y});
    }

    // Define KD-Tree using nanoflann
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, PointCloud>,
        PointCloud, 2 /*dim*/>
        KDTree;

    KDTree index(2, cloud, {10 /* max leaf size */});
    index.buildIndex();

    std::vector<double> filtered_distances;
    std::vector<double> filtered_angles;

    for (size_t i = 0; i < cloud.pts.size(); ++i) {
        const double query_pt[2] = {cloud.pts[i][0], cloud.pts[i][1]};

        std::vector<size_t> ret_index(k);
        std::vector<double> out_dist_sqr(k);

        nanoflann::KNNResultSet<double> resultSet(k);
        resultSet.init(&ret_index[0], &out_dist_sqr[0]);
        index.findNeighbors(resultSet, query_pt, nanoflann::SearchParameters(10));

        // Compute mean and standard deviation
        std::vector<double> neighbor_distances;
        for (double d : out_dist_sqr) {
            neighbor_distances.push_back(std::sqrt(d));
        }

        double mean = std::accumulate(neighbor_distances.begin(), neighbor_distances.end(), 0.0) / neighbor_distances.size();
        double variance = 0.0;
        for (double d : neighbor_distances) {
            variance += (d - mean) * (d - mean);
        }
        variance /= neighbor_distances.size();
        double std_dev = std::sqrt(variance);

        // Apply noise filtering
        if (std_dev < std_threshold && mean < mean_threshold) {
            filtered_distances.push_back(distances[i]);
            filtered_angles.push_back(angles[i]);
        }
    }

    // Update distances and angles
    distances = filtered_distances;
    angles = filtered_angles;
}
