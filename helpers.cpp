#include "helpers.h"

void xy_to_polar(const std::vector<std::pair<double, double>>& xy_data,
                 std::vector<double>& distances,
                 std::vector<double>& angles) {
    distances.clear();
    angles.clear();

    for (const auto& point : xy_data) {
        double x = point.first;
        double y = point.second;
        double distance = std::sqrt(x * x + y * y);
        double angle = std::atan2(y, x);  // Radians

        distances.push_back(distance);
        angles.push_back(angle);
    }
}
