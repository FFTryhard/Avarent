#pragma once

#include <vector>
#include "kmeans.h"
#include <iostream>
#include <Eigen\Dense>

//double MahalabonisDist(std::vector<double> point, std::vector<std::vector<double>> centroids, std::vector<Eigen::Matrix<double, 16, 16>> covmatrices);
struct PolarCoordinates{

    double r;
    double cos_t;
    double sin_t;

      // Define the '<' operator for sorting by radius only
    bool operator<(const PolarCoordinates& other) const {
        return this->r < other.r;  // Compare only the radius
    }

};



bool hasNaN(const std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>& covmatrices);

class SensorData; 

std::vector <std::vector <PolarCoordinates>> polar_csv_slidingwindow_fixed(SensorData& stream, int stride, int windowSize);

std::vector <std::vector <PolarCoordinates>> polar_slidingwindow_fixed(std::vector<std::vector<PolarCoordinates>>, int stride, int windowSize);






