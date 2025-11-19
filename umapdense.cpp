#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <irlba/irlba.hpp>
#include <irlba/parallel.hpp>
#include "umappp/umappp.hpp"

std::vector<std::vector<double>> generate5DClusters2DObvious(int num_points = 200, int num_clusters = 4) {
    std::vector<std::vector<double>> data;
    std::random_device rd;
    std::mt19937 gen(rd());

    // Strong separation in first 2 dimensions
    std::vector<std::pair<double, double>> centers2D = {
        {0.0, 0.0},
        {10.0, 10.0},
        {-10.0, -10.0},
        {5.0, -5.0}
    };

    // Small noise for other dimensions
    std::normal_distribution<> small_noise(0.0, 1.0);

    for (int i = 0; i < num_points; ++i) {
        int cluster_idx = i % num_clusters; // cycle through clusters
        std::vector<double> point(5);

        // First 2 dimensions are strongly clustered
        point[0] = centers2D[cluster_idx].first + small_noise(gen);
        point[1] = centers2D[cluster_idx].second + small_noise(gen);

        // Remaining 3 dimensions just add some variation
        point[2] = small_noise(gen);
        point[3] = small_noise(gen);
        point[4] = small_noise(gen);

        data.push_back(point);
    }

    return data;
}

std::vector<std::vector<double>> matrixToVector2D(const Eigen::MatrixXd& mat) {
    std::vector<std::vector<double>> result(mat.rows(), std::vector<double>(mat.cols()));

    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            result[i][j] = mat(i, j);
        }
    }

    return result;
}

Eigen::MatrixXd vectorToEigenMatrix(const std::vector<std::vector<double>>& data) {

    if (data.empty()) {
        std::cout << "return empty matrix" << std::endl;
        return Eigen::MatrixXd(); // return empty matrix
    }

    size_t rows = data.size();
    size_t cols = data[0].size();

    // Verify all rows have the same size
    for (const auto& row : data) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows in the 2D vector must have the same length.");
        }
    }

    Eigen::MatrixXd mat(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = data[i][j];
        }
    }

    return mat;
}

int main() {
    auto data = generate5DClusters2DObvious(); // std::vector<std::vector<double>>

    for(auto vector : data){
      std::cout << "(";
        for(int i = 0; i < 2; i++){
           std::cout << vector[i];
             if(i == 0){
               std::cout << ", ";
             }
        }
       std::cout << ")";
     std::cout << std::endl;
    }

    size_t nobs = data.size();
    size_t ndim = data[0].size();
    size_t out_dim = 2;

    // Flatten the data into a single contiguous vector
    std::vector<double> flat_data;
    flat_data.reserve(nobs * ndim);
    for (const auto& point : data) {
        flat_data.insert(flat_data.end(), point.begin(), point.end());
    }

    // Prepare embedding space
    std::vector<double> embedding(nobs * out_dim);

    // Build KNN structure
    knncolle::VptreeBuilder<int, double, double> vp_builder(
        std::make_shared<knncolle::EuclideanDistance<double, double>>()
    );

    umappp::Options opt;

    // ✅ Correct call — using flattened data
    auto status = umappp::initialize(
        static_cast<int>(ndim),
        static_cast<int>(nobs),
        flat_data.data(),
        vp_builder,
        static_cast<int>(out_dim),
        embedding.data(),
        opt
    );

    std::cout << "UMAP 2D embedding:\n";
    for (size_t i = 0; i < nobs; ++i) {
        std::cout << "(" << embedding[i * out_dim] << ", " << embedding[i * out_dim + 1] << ")\n";
    }
   
    return 0;
}
