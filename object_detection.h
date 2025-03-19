#ifndef OBJECT_DETECTION_H
#define OBJECT_DETECTION_H

#include <vector>

// Function to cluster LiDAR points based on distance thresholds
std::vector<std::vector<std::pair<double, double>>> cluster_lidar_points(
    const std::vector<double>& distances,
    const std::vector<double>& angles,
    double dist_max = 60.0, double threshold = 2.0);

// Function to merge clusters if they are close enough
std::vector<std::vector<std::pair<double, double>>> merge_clusters(
    const std::vector<std::vector<std::pair<double, double>>>& clusters,
    double merge_threshold);

std::vector<std::vector<std::pair<double, double>>> clusters_to_oriented_rectangles(
    const std::vector<std::vector<std::pair<double, double>>>& merged_clusters);

std::vector<std::vector<std::pair<double, double>>> expand_oriented_rectangles(
    const std::vector<std::vector<std::pair<double, double>>>& rectangles, 
    double expansion_size = 1.0);

std::vector<double> compute_safe_travel_distances(
    const std::vector<std::vector<std::pair<double, double>>>& expanded_rectangles,
    const std::vector<double>& angles,
    double max_distance = 60.0);

#endif // OBJECT_DETECTION_H
