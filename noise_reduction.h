#ifndef NOISE_REDUCTION_H
#define NOISE_REDUCTION_H

#include <vector>

// Function to remove noise using KNN with a KD-tree (nanoflann)
void remove_noise_knn(std::vector<double>& distances,
                      std::vector<double>& angles,
                      int k = 5, double std_threshold = 0.5, double mean_threshold = 2.0);

#endif // NOISE_REDUCTION_H
