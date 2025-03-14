#ifndef OBJECT_DETECTION_H
#define OBJECT_DETECTION_H

#include <vector>

// Function to cluster LiDAR points based on distance thresholds
std::vector<std::vector<std::pair<double, double>>> cluster_lidar_points(
    const std::vector<double>& distances,
    const std::vector<double>& angles,
    double dist_max = 60.0, double threshold = 1.0);

// Function to merge clusters if they are close enough
std::vector<std::vector<std::pair<double, double>>> merge_clusters(
    const std::vector<std::vector<std::pair<double, double>>>& clusters,
    double merge_threshold);

#endif // OBJECT_DETECTION_H
