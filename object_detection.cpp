#include "object_detection.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

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

    for (const auto& current_cluster : clusters) {
        bool merged = false;

        if (current_cluster.size() == 1) {
            // Single-point cluster: Check if it should merge into an existing cluster
            auto point = current_cluster[0];
            double point_x = point.first * cos(point.second);
            double point_y = point.first * sin(point.second);

            for (auto& existing_cluster : merged_clusters) {
                for (const auto& cluster_point : existing_cluster) {
                    double cluster_x = cluster_point.first * cos(cluster_point.second);
                    double cluster_y = cluster_point.first * sin(cluster_point.second);

                    double distance = sqrt(pow(point_x - cluster_x, 2) + pow(point_y - cluster_y, 2));

                    if (distance <= merge_threshold) {
                        // Merge single-point cluster into existing cluster
                        existing_cluster.push_back(point);
                        merged = true;
                        break;
                    }
                }
                if (merged) break; // Stop checking once merged
            }
        } else {
            // Standard merging logic for multi-point clusters
            if (!merged_clusters.empty()) {
                auto& last_cluster = merged_clusters.back();
                auto& first_point = current_cluster.front();
                auto& last_point = last_cluster.back();

                double last_x = last_point.first * cos(last_point.second);
                double last_y = last_point.first * sin(last_point.second);
                double first_x = first_point.first * cos(first_point.second);
                double first_y = first_point.first * sin(first_point.second);
                
                double distance_between_clusters = sqrt(pow(last_x - first_x, 2) + pow(last_y - first_y, 2));

                if (distance_between_clusters <= merge_threshold) {
                    last_cluster.insert(last_cluster.end(), current_cluster.begin(), current_cluster.end());
                    merged = true;
                }
            }
        }

        if (!merged) {
            // Add as a new cluster if not merged
            merged_clusters.push_back(current_cluster);
        }
    }

    return merged_clusters;
}


std::vector<std::vector<std::pair<double, double>>> clusters_to_oriented_rectangles(
    const std::vector<std::vector<std::pair<double, double>>>& merged_clusters) {
    
    std::vector<std::vector<std::pair<double, double>>> rectangles;

    for (const auto& cluster : merged_clusters) {
        if (cluster.size() < 2) continue;

        // Convert polar to Cartesian coordinates
        std::vector<double> x_coords, y_coords;
        for (const auto& point : cluster) {
            double x = point.first * cos(point.second);
            double y = point.first * sin(point.second);
            x_coords.push_back(x);
            y_coords.push_back(y);
        }

        // Convert to Eigen matrix (N x 2)
        MatrixXd points(cluster.size(), 2);
        for (size_t i = 0; i < cluster.size(); ++i) {
            points(i, 0) = x_coords[i];
            points(i, 1) = y_coords[i];
        }

        // Compute mean
        RowVector2d mean = points.colwise().mean();

        // Center points by subtracting the mean
        MatrixXd centered = points.rowwise() - mean;

        // Compute covariance matrix
        Matrix2d cov = (centered.transpose() * centered) / double(points.rows());

        // Compute eigenvalues & eigenvectors (PCA)
        SelfAdjointEigenSolver<Matrix2d> solver(cov);
        Matrix2d eigenvectors = solver.eigenvectors(); // Principal axes

        // Rotate points to PCA-aligned space
        MatrixXd rotated = centered * eigenvectors;

        // Get bounding box in rotated space
        double min_x = rotated.col(0).minCoeff();
        double max_x = rotated.col(0).maxCoeff();
        double min_y = rotated.col(1).minCoeff();
        double max_y = rotated.col(1).maxCoeff();

        // Define rectangle in rotated space
        MatrixXd rect_corners(4, 2);
        rect_corners << min_x, min_y,
                        max_x, min_y,
                        max_x, max_y,
                        min_x, max_y;

        // Rotate back to original space
        MatrixXd transformed = (rect_corners * eigenvectors.transpose()).rowwise() + mean;

        // Store rectangle as a vector of (x, y) pairs
        std::vector<std::pair<double, double>> rectangle;
        for (int i = 0; i < 4; ++i) {
            rectangle.emplace_back(transformed(i, 0), transformed(i, 1));
        }

        rectangles.push_back(rectangle);
    }

    return rectangles;
}

std::vector<std::pair<double, double>> expand_oriented_rectangle(
    const std::vector<std::pair<double, double>>& rect_corners, double expansion_size) {
    
    if (rect_corners.size() != 4) return rect_corners;  // Ensure valid input

    // Compute rectangle center
    Vector2d center(0, 0);
    for (const auto& point : rect_corners) {
        center += Vector2d(point.first, point.second);
    }
    center /= 4.0;  // Compute centroid

    // Compute edge vectors
    Vector2d edge1(rect_corners[1].first - rect_corners[0].first, rect_corners[1].second - rect_corners[0].second);
    Vector2d edge2(rect_corners[2].first - rect_corners[1].first, rect_corners[2].second - rect_corners[1].second);

    // Normalize edge vectors
    edge1.normalize();
    edge2.normalize();

    // Compute expansion along both axes
    Vector2d expansion_x = edge1 * expansion_size;
    Vector2d expansion_y = edge2 * expansion_size;

    // Compute new expanded rectangle corners
    std::vector<std::pair<double, double>> expanded_corners = {
        {rect_corners[0].first - expansion_x[0] - expansion_y[0], rect_corners[0].second - expansion_x[1] - expansion_y[1]},
        {rect_corners[1].first + expansion_x[0] - expansion_y[0], rect_corners[1].second + expansion_x[1] - expansion_y[1]},
        {rect_corners[2].first + expansion_x[0] + expansion_y[0], rect_corners[2].second + expansion_x[1] + expansion_y[1]},
        {rect_corners[3].first - expansion_x[0] + expansion_y[0], rect_corners[3].second - expansion_x[1] + expansion_y[1]}
    };

    return expanded_corners;
}

// Function to expand all oriented bounding rectangles
std::vector<std::vector<std::pair<double, double>>> expand_oriented_rectangles(
    const std::vector<std::vector<std::pair<double, double>>>& rectangles, double expansion_size) {
    
    std::vector<std::vector<std::pair<double, double>>> expanded_rectangles;
    for (const auto& rect : rectangles) {
        expanded_rectangles.push_back(expand_oriented_rectangle(rect, expansion_size));
    }
    return expanded_rectangles;
}


// Function to compute the intersection of a ray with a rectangle edge
bool ray_intersects_edge(
    const Vector2d& ray_origin, const Vector2d& ray_direction,
    const Vector2d& p1, const Vector2d& p2, double& intersection_distance) {
    
    // Parametric line equations: p = p1 + t * (p2 - p1)
    Vector2d edge_vector = p2 - p1;
    Vector2d ray_to_p1 = p1 - ray_origin;
    
    // Solving intersection equation using determinant method
    double det = (-ray_direction.x() * edge_vector.y() + ray_direction.y() * edge_vector.x());
    
    if (fabs(det) < 1e-6) return false;  // Lines are parallel or coincident

    double t1 = (edge_vector.x() * ray_to_p1.y() - edge_vector.y() * ray_to_p1.x()) / det;
    double t2 = (ray_direction.x() * ray_to_p1.y() - ray_direction.y() * ray_to_p1.x()) / det;

    if (t1 >= 0 && t2 >= 0 && t2 <= 1) {
        Vector2d intersection_point = ray_origin + t1 * ray_direction;
        intersection_distance = intersection_point.norm();
        return true;
    }

    return false;
}

// Function to compute the minimum safe travel distance for each angle
std::vector<double> compute_safe_travel_distances(
    const std::vector<std::vector<std::pair<double, double>>>& expanded_rectangles,
    const std::vector<double>& angles,
    double max_distance) {

    std::vector<double> safe_distances(angles.size(), max_distance);  // Initialize with max distance

    for (const auto& rect : expanded_rectangles) {
        if (rect.size() != 4) continue;  // Ensure valid rectangle

        // Convert rectangle corners to Eigen vectors
        vector<Vector2d> rect_corners;
        for (const auto& point : rect) {
            rect_corners.push_back(Vector2d(point.first, point.second));
        }

        // Compute distance for each angle
        for (size_t i = 0; i < angles.size(); ++i) {
            double angle = angles[i];
            Vector2d ray_origin(0, 0);  // Boat's position at the origin
            Vector2d ray_direction(cos(angle), sin(angle));

            double min_dist = max_distance;  // Start with max distance

            // Check intersection with each edge of the rectangle
            for (size_t j = 0; j < 4; ++j) {
                Vector2d p1 = rect_corners[j];
                Vector2d p2 = rect_corners[(j + 1) % 4];  // Wraps around to close rectangle

                double intersection_distance;
                if (ray_intersects_edge(ray_origin, ray_direction, p1, p2, intersection_distance)) {
                    min_dist = std::min(min_dist, intersection_distance);
                }
            }

            // Ensure the correct distance is used
            safe_distances[i] = std::min(safe_distances[i], min_dist);
        }
    }

    return safe_distances;
}