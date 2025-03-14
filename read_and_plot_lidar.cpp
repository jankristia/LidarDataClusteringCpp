#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>
#include "matplotlib-cpp/matplotlibcpp.h"
#include "helpers.h"
#include "noise_reduction.h"
#include "object_detection.h"

namespace plt = matplotlibcpp;
using json = nlohmann::json;
using namespace std;

// Function to read LiDAR data from a JSONL file
vector<pair<double, double>> read_lidar_data(const string& filename, int frame_idx = 0) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    string line;
    vector<json> frames;
    while (getline(file, line)) {
        frames.push_back(json::parse(line));
    }
    file.close();

    if (frame_idx >= frames.size()) {
        cerr << "Frame index out of range!" << endl;
        exit(1);
    }

    vector<pair<double, double>> lidar_points;
    for (auto& point : frames[frame_idx]["points"]) {
        double x = point[0];
        double y = point[1];
        lidar_points.push_back({x, y});
    }

    return lidar_points;
}

void plot_lidar_data(const vector<pair<double, double>>& lidar_data) {
    vector<double> x, y;
    for (const auto& point : lidar_data) {
        x.push_back(point.first);
        y.push_back(point.second);
    }

    plt::figure_size(800, 800);
    plt::scatter(x, y, 10);  // Scatter plot with point size 10
    plt::xlabel("X (m)");
    plt::ylabel("Y (m)");
    plt::title("LiDAR Data Visualization");
    plt::grid(true);
    plt::show(false);
}

// Function to plot LiDAR data in Polar coordinates
void plot_lidar_polar(const vector<double>& distances, const vector<double>& angles) {
    vector<double> x, y;

    // Convert polar to Cartesian for plotting
    for (size_t i = 0; i < distances.size(); ++i) {
        x.push_back(distances[i] * cos(angles[i]));  // X-coordinate
        y.push_back(distances[i] * sin(angles[i]));  // Y-coordinate
    }

    plt::figure_size(800, 800);
    plt::scatter(x, y, 10);  // Scatter plot with point size 10
    plt::xlabel("X (m)");
    plt::ylabel("Y (m)");
    plt::title("LiDAR Data (Polar Coordinates Transformed)");
    plt::grid(true);
    plt::show(false);
}

// Function to plot clustered data
void plot_clusters(const vector<vector<pair<double, double>>>& clusters) {
    plt::figure_size(800, 800);
    for (size_t i = 0; i < clusters.size(); ++i) {
        vector<double> x, y;
        for (const auto& point : clusters[i]) {
            x.push_back(point.first * cos(point.second));
            y.push_back(point.first * sin(point.second));
        }
        plt::scatter(x, y, 10);
    }
    plt::xlabel("X (m)");
    plt::ylabel("Y (m)");
    plt::title("Clustered LiDAR Data");
    plt::grid(true);
    plt::show(false);
}

int main() {
    string filename = "../shoreline_scen4_2.jsonl";  // Change if needed
    int frame_idx = 150;  // Load first frame

    // Read LiDAR data
    vector<pair<double, double>> lidar_data = read_lidar_data(filename, frame_idx);

    vector<double> distances, angles;
    xy_to_polar(lidar_data, distances, angles);

    plot_lidar_polar(distances, angles);

    remove_noise_knn(distances, angles);

    auto clusters = cluster_lidar_points(distances, angles);
    auto merged_clusters = merge_clusters(clusters, 4.0);
    plot_lidar_polar(distances, angles);
    plot_clusters(merged_clusters);

    plt::show();
    return 0;
}
