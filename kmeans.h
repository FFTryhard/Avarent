#pragma once
#include "utils.h
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib> 
#include <Eigen\Dense>

double euclideanDist(const std::vector<double>& a, const std::vector<double>& b);

std::vector<double> computeCentroid(const std::vector<double>&);
std::vector<std::vector<double>> initializeCentroidsKMeansPP(const std::vector<std::vector<double>>& data, int k);
// this will compute the mean vector of the random assigned points (vectors)

std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>, double> kmeans(const std::vector<std::vector<double>>& data, int k, int max_iters);
std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>, double> kmeansM(const std::vector<std::vector<double>>& data, int k, int max_iters);
//tuple = function that can return multiple values

//takes in our vectors (2d). asks for how manu clusters

//SILLHOUTTE

double silhouetteScore(const std::vector<std::vector<double>>& X, const std::vector<int>& labels);
double silhouetteScoreMahalanobisOptimized(
    const std::vector<std::vector<double>>& data,
    const std::vector<int>& labels,
    const std::vector<std::vector<std::vector<double>>>& clusters,
    const std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>& covariancematrices
);



