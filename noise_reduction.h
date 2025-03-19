#ifndef NOISE_REDUCTION_H
#define NOISE_REDUCTION_H

#include <vector>

// Function to remove noise using KNN with a KD-tree (nanoflann)
void remove_noise_knn(std::vector<double>& distances,
                      std::vector<double>& angles,
                      std::vector<double>& noise_distances,
                      std::vector<double>& noise_angles,
                      int k = 5, double std_threshold = 3, double mean_threshold = 4.0);

#endif // NOISE_REDUCTION_H
