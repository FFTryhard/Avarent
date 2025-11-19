#pragma once
#include <iostream>
#include "utils.h"
#include <vector>

std::vector<double> convertToRank(const std::vector<double> &X, bool);
double spearmanCorrelation(const std::vector<double> &x, const std::vector<double> &y); //double cuz what if 2+1/2 = 1.5
double pearsonCorrelation(std::vector<double> A, std::vector<double> B);
double score(double mean_r, double median_r, double mean_rs, double median_rs);
std::vector<int> knn(std::vector<std::vector<double>> data, double min_dist);
