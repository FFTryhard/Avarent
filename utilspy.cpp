#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "utilspy.h"

std::vector<std::vector<double>> generate5DClusters2DObvious(
    int num_points, int num_clusters,
    double cluster_spacing, double noise_std)
{
    if (num_points <= 0)
        throw std::invalid_argument("num_points must be positive");
    if (num_clusters <= 0)
        throw std::invalid_argument("num_clusters must be positive");
    if (num_clusters > num_points)
        throw std::invalid_argument("num_clusters cannot exceed num_points");

    std::vector<std::vector<double>> data;
    data.reserve(num_points);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noise_std);

    // --- Generate cluster centers arranged roughly in a circle ---
    std::vector<std::pair<double, double>> centers2D;
    centers2D.reserve(num_clusters);

    const double radius = cluster_spacing * std::max(1.0, std::sqrt(num_clusters) / 2.0);
    const double angle_step = 2.0 * M_PI / num_clusters;

    for (int c = 0; c < num_clusters; ++c) {
        double angle = c * angle_step;
        double x = radius * std::cos(angle);
        double y = radius * std::sin(angle);
        centers2D.emplace_back(x, y);
    }

    // --- Random cluster assignment for variety ---
    std::uniform_int_distribution<> cluster_dist(0, num_clusters - 1);

    for (int i = 0; i < num_points; ++i) {
        int cluster_idx = cluster_dist(gen);
        const auto& center = centers2D[cluster_idx];

        std::vector<double> point(5);
        // Strong separation in first two dimensions
        point[0] = center.first + noise(gen);
        point[1] = center.second + noise(gen);
        // Remaining three: small random variation around 0
        for (int d = 2; d < 5; ++d)
            point[d] = noise(gen);

        data.push_back(std::move(point));
    }

    return data;
}

pybind11::array_t<double> create_2d_vector(std::vector<std::vector<double>> data) {
    // Create a 2D vector (std::vector<std::vector<double>>)


    // Get dimensions
    size_t rows = data.size();
    size_t cols = data[0].size();
 
    // Flatten the 2D vector into a 1D vector to pass to NumPy
    std::vector<double> flattened;
    for (const auto& row : data) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }

    // Create a NumPy array from the flattened data
    pybind11::array_t<double> arr({rows, cols}, flattened.data());

    return arr;
}

