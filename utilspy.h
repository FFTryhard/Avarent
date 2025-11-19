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

std::vector<std::vector<double>> generate5DClusters2DObvious(
    int num_points, int num_clusters,
    double cluster_spacing, double noise_std);

    

pybind11::array_t<double> create_2d_vector(std::vector<std::vector<double>> data);
    // Create a 2D vector (std::vector<std::vector<double>>)


 
