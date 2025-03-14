#include "object_detection.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

// Function to cluster LiDAR points based on a distance threshold
std::vector<std::vector<std::pair<double, double>>> cluster_lidar_points(
    const std::vector<double>& distances,
    const std::vector<double>& angles,
    double dist_max, double threshold) {
    
    std::vector<std::vector<std::pair<double, double>>> clusters;
    std::vector<std::pair<double, double>> current_cluster;

    for (size_t i = 0; i < distances.size(); ++i) {
        double distance = distances[i];
        double angle = angles[i];

        // Ignore points beyond max distance
        if (distance >= dist_max) {
            if (!current_cluster.empty()) {
                clusters.push_back(current_cluster);
                current_cluster.clear();
            }
            continue;
        }

        // If there is a significant gap, start a new cluster
        if (!current_cluster.empty() && 
            std::abs(distance - current_cluster.back().first) > threshold) {
            clusters.push_back(current_cluster);
            current_cluster.clear();
        }

        current_cluster.push_back({distance, angle});
    }

    if (!current_cluster.empty()) {
        clusters.push_back(current_cluster);
    }

    return clusters;
}

// Function to merge clusters if they are close enough
std::vector<std::vector<std::pair<double, double>>> merge_clusters(
    const std::vector<std::vector<std::pair<double, double>>>& clusters,
    double merge_threshold) {
    
    if (clusters.empty()) return {};

    std::vector<std::vector<std::pair<double, double>>> merged_clusters;
    merged_clusters.push_back(clusters[0]);

    for (size_t i = 1; i < clusters.size(); ++i) {
        auto& last_cluster = merged_clusters.back();
        auto& current_cluster = clusters[i];

        // Compute Euclidean distance between last point of previous cluster
        // and first point of the next cluster
        double last_x = last_cluster.back().first * cos(last_cluster.back().second);
        double last_y = last_cluster.back().first * sin(last_cluster.back().second);
        double first_x = current_cluster.front().first * cos(current_cluster.front().second);
        double first_y = current_cluster.front().first * sin(current_cluster.front().second);
        
        double distance_between_clusters = sqrt(pow(last_x - first_x, 2) + pow(last_y - first_y, 2));

        if (distance_between_clusters <= merge_threshold) {
            // Merge clusters
            merged_clusters.back().insert(merged_clusters.back().end(), current_cluster.begin(), current_cluster.end());
        } else if (current_cluster.size() < 3) {
            // Don't make objects with less than three measurements
            continue;
        }else {
            // Keep clusters separate
            merged_clusters.push_back(current_cluster);
        }
    }

    return merged_clusters;
}
