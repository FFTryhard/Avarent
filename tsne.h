#pragma once

#include <C:\Users\H\Desktop\C++ Tryharding\eigen-3.4.0\eigen-3.4.0\Eigen\Dense>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <limits>

using namespace Eigen;
using namespace std;

MatrixXd pairwiseDist2(const MatrixXd &X);
MatrixXd computeP(const MatrixXd &D, double perplexity = 30.0, double tol = 1e-5);
MatrixXd tsne(const MatrixXd &X, int no_dims = -1, double perplexity = 30.0, int max_iter = 1000,
              double learning_rate = 200.0, double momentum = 0.8);