#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>
#include <filesystem>
#include "matplotlib-cpp/matplotlibcpp.h"
#include "helpers.h"
#include "noise_reduction.h"
#include "object_detection.h"

namespace plt = matplotlibcpp;
using json = nlohmann::json;
using namespace std;
namespace fs = std::filesystem;

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

void plot_lidar_polar_w_noise_reduction(
    const vector<double>& distances1, const vector<double>& angles1,
    const vector<double>& distances2, const vector<double>& angles2) {
    
    vector<double> x1, y1, x2, y2;

    // Convert polar to Cartesian for first dataset
    for (size_t i = 0; i < distances1.size(); ++i) {
        x1.push_back(distances1[i] * cos(angles1[i]));  
        y1.push_back(distances1[i] * sin(angles1[i]));  
    }

    // Convert polar to Cartesian for second dataset
    for (size_t i = 0; i < distances2.size(); ++i) {
        x2.push_back(distances2[i] * cos(angles2[i]));  
        y2.push_back(distances2[i] * sin(angles2[i]));  
    }

    plt::figure_size(800, 800);
    
    // Scatter plot for first dataset (red)
    plt::named_plot("Free Space Data", x1, y1, "ro");

    // Scatter plot for second dataset (blue)
    plt::named_plot("Detected Noise", x2, y2, "bo");

    // Boat polygon centered at origin
    vector<double> boat_x = {0, 1, 1, -1, -1, 0.0};
    vector<double> boat_y = {3, 1.5, -1.5, -1.5, 1.5, 3};
    plt::named_plot("Boat", boat_x, boat_y, "k-");

    // Axis labels and grid
    plt::xlabel("X (m)");
    plt::ylabel("Y (m)");
    plt::title("Noise Reduction of Free Space Data");
    plt::grid(true);
    plt::axis("equal");
    plt::legend();

    // Show plot (keep open for multiple plots)
    plt::show(false);
}


// Function to plot clustered data with legends
void plot_clusters(const vector<vector<pair<double, double>>>& clusters) {
    plt::figure_size(800, 800);

    for (size_t i = 0; i < clusters.size(); ++i) {
        vector<double> x, y;
        for (const auto& point : clusters[i]) {
            x.push_back(point.first * cos(point.second));
            y.push_back(point.first * sin(point.second));
        }

        // Use named_plot to include the cluster in the legend
        string label = "Cluster " + to_string(i + 1);
        plt::named_plot(label, x, y, "o");
    }
    // Boat polygon centered at origin
    vector<double> boat_x = {0, 1, 1, -1, -1, 0.0};
    vector<double> boat_y = {3, 1.5, -1.5, -1.5, 1.5, 3};
    plt::named_plot("Boat", boat_x, boat_y, "k-");
    
    plt::xlabel("X (m)");
    plt::ylabel("Y (m)");
    plt::title("Clustered Free Space Data");
    plt::grid(true);
    plt::axis("equal");
    plt::legend();

    // Show the plot but keep it open for multiple plots
    plt::show(false);
}

void plot_clusters_and_oriented_rectangles(
    const vector<vector<pair<double, double>>>& clusters,
    const vector<vector<pair<double, double>>>& rectangles,
    const vector<vector<pair<double, double>>>& expanded_rectangles) {

    plt::figure_size(800, 800);

    // Plot clusters
    for (size_t i = 0; i < clusters.size(); ++i) {
        vector<double> x, y;
        for (const auto& point : clusters[i]) {
            x.push_back(point.first * cos(point.second));
            y.push_back(point.first * sin(point.second));
        }
        plt::scatter(x, y, 10);
    }

    // Plot bounding boxes
    for (const auto& rect : rectangles) {
        vector<double> x = {rect[0].first, rect[1].first, rect[2].first, rect[3].first, rect[0].first};
        vector<double> y = {rect[0].second, rect[1].second, rect[2].second, rect[3].second, rect[0].second};
        plt::plot(x, y, "g-");  // Green for bounding boxes
    }

    // Plot expanded bounding boxes
    for (const auto& rect : expanded_rectangles) {
        vector<double> x = {rect[0].first, rect[1].first, rect[2].first, rect[3].first, rect[0].first};
        vector<double> y = {rect[0].second, rect[1].second, rect[2].second, rect[3].second, rect[0].second};
        plt::plot(x, y, "b-");  // Blue for expanded bounding boxes
    }

    // Boat polygon
    vector<double> boat_x = {0, 1, 1, -1, -1, 0.0};
    vector<double> boat_y = {3, 1.5, -1.5, -1.5, 1.5, 3};
    plt::plot(boat_x, boat_y, "k-");  // Black boat outline

    // Manually add legend entries
    plt::named_plot("Bounding box", vector<double>{0, 0}, vector<double>{0, 0}, "g-");
    plt::named_plot("Expanded bounding box", vector<double>{0, 0}, vector<double>{0, 0}, "b-");
    plt::named_plot("Boat", vector<double>{0, 0}, vector<double>{0, 0}, "k-");

    // Formatting
    plt::xlabel("X (m)");
    plt::ylabel("Y (m)");
    plt::title("Clustered Free Space Data with Oriented Bounding Boxes");
    plt::grid(true);
    plt::axis("equal");
    plt::legend();
    plt::show(false);
}


void plot_safe_travel_distances(
    const vector<vector<pair<double, double>>>& clusters,
    const vector<vector<pair<double, double>>>& rectangles,
    const vector<vector<pair<double, double>>>& expanded_rectangles,
    const vector<double>& angles,
    const vector<double>& safe_distances) {

    plt::figure_size(800, 800);
    
    // Plot clusters
    for (size_t i = 0; i < clusters.size(); ++i) {
        vector<double> x, y;
        for (const auto& point : clusters[i]) {
            x.push_back(point.first * cos(point.second));
            y.push_back(point.first * sin(point.second));
        }
        plt::scatter(x, y, 10);
    }

    // Plot bounding boxes
    for (const auto& rect : rectangles) {
        vector<double> x = {rect[0].first, rect[1].first, rect[2].first, rect[3].first, rect[0].first};
        vector<double> y = {rect[0].second, rect[1].second, rect[2].second, rect[3].second, rect[0].second};
        plt::plot(x, y, "g-");  // Green for bounding boxes
    }

    // Plot expanded bounding boxes
    for (const auto& rect : expanded_rectangles) {
        vector<double> x = {rect[0].first, rect[1].first, rect[2].first, rect[3].first, rect[0].first};
        vector<double> y = {rect[0].second, rect[1].second, rect[2].second, rect[3].second, rect[0].second};
        plt::plot(x, y, "b-");  // Blue for expanded bounding boxes
    }

    // Plot safe travel distances
    for (size_t i = 0; i < angles.size(); ++i) {
        double x_end = safe_distances[i] * cos(angles[i]);
        double y_end = safe_distances[i] * sin(angles[i]);
        plt::plot({0, x_end}, {0, y_end}, "r-");  // Red for safe travel distances
    }

    // Boat polygon
    vector<double> boat_x = {0, 1, 1, -1, -1, 0.0};
    vector<double> boat_y = {3, 1.5, -1.5, -1.5, 1.5, 3};
    plt::plot(boat_x, boat_y, "k-");  // Black boat outline

    // Manually add legend entries
    plt::named_plot("Bounding box", vector<double>{0, 0}, vector<double>{0, 0}, "g-");
    plt::named_plot("Expanded bounding box", vector<double>{0, 0}, vector<double>{0, 0}, "b-");
    plt::named_plot("Safe travel distances", vector<double>{0, 0}, vector<double>{0, 0}, "r-");
    plt::named_plot("Boat", vector<double>{0, 0}, vector<double>{0, 0}, "k-");

    // Formatting
    plt::axis("equal");
    plt::xlabel("X (m)");
    plt::ylabel("Y (m)");
    plt::title("Safe Travel Distances");
    plt::grid(true);
    plt::legend();
    plt::show(false);
}




const string frame_output_dir = "../video_frames";

// Function to process and save each frame
void process_and_save_frame(
    const vector<pair<double, double>>& lidar_data,
    int frame_idx) {

    // Convert to polar
    vector<double> distances, angles;
    xy_to_polar(lidar_data, distances, angles);

    
    // Apply noise removal using KD-Tree-based KNN
    vector<double> noise_distances, noise_angles;
    remove_noise_knn(distances, angles, noise_distances, noise_angles);


    // Cluster LiDAR points
    auto clusters = cluster_lidar_points(distances, angles);
    auto merged_clusters = merge_clusters(clusters, 4.0);

    // Compute oriented bounding boxes
    auto oriented_rectangles = clusters_to_oriented_rectangles(merged_clusters);

    // Expand bounding rectangles
    auto expanded_rectangles = expand_oriented_rectangles(oriented_rectangles, 1.0);

    // Define angles for travel distance calculation
    vector<double> travel_angles;
    for (int deg = 0; deg <= 180; deg += 2) {
        travel_angles.push_back(deg * M_PI / 180.0); // Convert degrees to radians
    }

    // Compute safe travel distances
    auto safe_distances = compute_safe_travel_distances(expanded_rectangles, travel_angles);

    // Create a plot for the frame
    plt::figure_size(800, 800);
    
    // Plot clustered points
    for (size_t i = 0; i < clusters.size(); ++i) {
        vector<double> x, y;
        for (const auto& point : clusters[i]) {
            x.push_back(point.first * cos(point.second));
            y.push_back(point.first * sin(point.second));
        }
        plt::scatter(x, y, 10);
    }

    // Plot bounding boxes
    for (const auto& rect : oriented_rectangles) {
        vector<double> x = {rect[0].first, rect[1].first, rect[2].first, rect[3].first, rect[0].first};
        vector<double> y = {rect[0].second, rect[1].second, rect[2].second, rect[3].second, rect[0].second};
        plt::plot(x, y, "g-");
    }

    // Plot expanded bounding boxes
    for (const auto& rect : expanded_rectangles) {
        vector<double> x = {rect[0].first, rect[1].first, rect[2].first, rect[3].first, rect[0].first};
        vector<double> y = {rect[0].second, rect[1].second, rect[2].second, rect[3].second, rect[0].second};
        plt::plot(x, y, "r-");
    }

    // Plot safe travel distances
    for (size_t i = 0; i < travel_angles.size(); ++i) {
        double x_end = safe_distances[i] * cos(travel_angles[i]);
        double y_end = safe_distances[i] * sin(travel_angles[i]);
        plt::plot({0, x_end}, {0, y_end}, "b-");
    }

    // Set axis limits
    plt::xlim(-35,35);
    plt::ylim(-15,55);
    plt::xlabel("X (m)");
    plt::ylabel("Y (m)");
    plt::title("LiDAR Frame: " + to_string(frame_idx));
    plt::grid(true);

    // Save frame as an image
    fs::create_directories(frame_output_dir); // Ensure output directory exists
    string filename = frame_output_dir + "/frame_" + to_string(frame_idx) + ".png";
    plt::save(filename);
    
    // Close figure to free memory
    plt::close();
}

// Function to read all frames and process them
void generate_video_from_lidar_data(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    vector<vector<pair<double, double>>> all_frames;
    string line;
    int frame_idx = 0;

    // Read all LiDAR frames
    while (getline(file, line)) {
        json frame_json = json::parse(line);
        vector<pair<double, double>> lidar_data;
        for (auto& point : frame_json["points"]) {
            lidar_data.push_back({point[0], point[1]});
        }
        all_frames.push_back(lidar_data);
    }
    file.close();

    // Process each frame
    for (size_t i = 0; i < all_frames.size(); ++i) {
        // cout << "Processing frame: " << i << "/" << all_frames.size() << endl;
        process_and_save_frame(all_frames[i], i);
    }

    // Generate video using FFmpeg
    cout << "Generating video..." << endl;
    string ffmpeg_command = "ffmpeg -framerate 10 -i " + frame_output_dir +
                            "/frame_%d.png -c:v libx264 -pix_fmt yuv420p lidar_video.mp4";
    system(ffmpeg_command.c_str());

    cout << "Video saved as lidar_video.mp4" << endl;
}

int main() {
    string filename = "../shoreline_scen4_2.jsonl";  // "../shoreline_scen4_2.jsonl", "../shoreline_scen6.jsonl"
    
    // generate_video_from_lidar_data(filename);


    int frame_idx = 150;  // Load first frame

    // Read LiDAR data
    vector<pair<double, double>> lidar_data = read_lidar_data(filename, frame_idx);

    vector<double> distances, angles;
    xy_to_polar(lidar_data, distances, angles);

    // plot_lidar_polar(distances, angles);

    vector<double> noise_distances, noise_angles;
    remove_noise_knn(distances, angles, noise_distances, noise_angles);

    plot_lidar_polar_w_noise_reduction(distances, angles, noise_distances, noise_angles);

    auto clusters = cluster_lidar_points(distances, angles);
    auto merged_clusters = merge_clusters(clusters, 3.0);
    auto oriented_rectangles = clusters_to_oriented_rectangles(merged_clusters);
    auto expanded_rectangles = expand_oriented_rectangles(oriented_rectangles, 1.0);


    vector<double> candidate_angles;
    for (int deg = 0; deg <= 180; deg += 2) {
        candidate_angles.push_back(deg * M_PI / 180.0);
    }
    auto safe_distances = compute_safe_travel_distances(expanded_rectangles, candidate_angles, 30);

    // // plot_lidar_polar(distances, angles);
    plot_clusters(clusters);
    plot_clusters(merged_clusters);
    plot_clusters_and_oriented_rectangles(merged_clusters, oriented_rectangles, expanded_rectangles);
    plot_safe_travel_distances(merged_clusters, oriented_rectangles, expanded_rectangles, candidate_angles, safe_distances);

    plt::show();
    return 0;
}
