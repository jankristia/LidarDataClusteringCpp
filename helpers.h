#ifndef HELPERS_H
#define HELPERS_H

#include <vector>
#include <cmath>

void xy_to_polar(const std::vector<std::pair<double, double>>& xy_data,
                 std::vector<double>& distances,
                 std::vector<double>& angles);

#endif // HELPERS_H
