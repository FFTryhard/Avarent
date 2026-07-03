
#pragma once

#include "rawprocessing.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

struct Timer {

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::chrono::duration<float> duration;

  Timer() { start = std::chrono::high_resolution_clock::now(); }

  ~Timer() {
    end = std::chrono::high_resolution_clock::now();

    duration = end - start;

    float ms = duration.count() * 1000.0f;
    std::cerr << "Timer took: " << ms << "ms" << std::endl;
  }
};

struct MatrixBuffer {
  Eigen::MatrixXd prev;  // value at t-1
  Eigen::MatrixXd curr;  // value at t
  bool has_prev = false; // to know if prev is valid

  // Add a new scalar (shifts curr -> prev)
  void add(const Eigen::MatrixXd &M) {
    if (!init) {
      curr = M;
      init = true;
      has_prev = false;
      return;
    }
    prev = curr;
    curr = M;
    has_prev = (prev.size() > 0 && curr.size() > 0);
  }

  // Access the current value
  const Eigen::MatrixXd &current() const { return curr; }

  // Access the previous value
  const Eigen::MatrixXd &previous() const {
    if (!has_prev)
      throw std::runtime_error("previous not available");
    return prev;
  }

private:
  bool init = false;
};

// --- Validation ---
bool containsNaN(const std::vector<std::vector<double>> &data);
bool containsINF(const std::vector<std::vector<double>> &data);
bool checkForNaN(const std::vector<double> &data);

// --- Scalar utilities ---
double clamp_(double x);
double median(std::vector<double> data);
double mean(const std::vector<double> &data);
double computeMAD(const std::vector<double> &values);
double computeR2(const std::vector<double> &x, const std::vector<double> &y);
double derivativeOLS(const std::vector<double> &v, double dt);
double perc_(std::vector<double> v, double p);

// --- Conversion utilities ---
Eigen::MatrixXd dequeToEigenMatrix(const std::deque<std::vector<double>> &data);
std::vector<std::vector<double>>
deque_to_vector(const std::deque<std::vector<double>> &src);
std::vector<double> deque_to_1vector(const std::deque<double> &src);
Eigen::VectorXd toEigenVector(const std::vector<double> &vec);
Eigen::MatrixXd
vectorToEigenMatrix(const std::vector<std::vector<double>> &data);
std::vector<double> eigenToStdVector(const Eigen::VectorXd &v);

template <typename T>
inline std::vector<T> deque_to_vector_template(const std::deque<T> &src) {
  std::vector<T> dst;
  dst.reserve(src.size());

  for (const auto &v : src) {
    dst.push_back(v);
  }

  return dst;
}

// --- Linear algebra / spectral ---
double effective_Rank(const Eigen::VectorXd &sigma);
double wedin_theoremKGT1(const Eigen::MatrixXd &M_overlap);
double wedin_theoremK1(const Eigen::VectorXd &u_cur,
                       const Eigen::VectorXd &u_prev);
double cosineSimilarity(const Eigen::VectorXd &A, const Eigen::VectorXd &B);

// --- PCA ---
struct PCA_Result {
  std::vector<std::vector<double>> projected;
  Eigen::MatrixXd eigenvectors;
  Eigen::VectorXd eigenvalues;
};
PCA_Result PCA_Full(const Eigen::MatrixXd &data, double varianceThreshold);
std::vector<std::vector<double>> PCA(const Eigen::MatrixXd &data,
                                     double varianceThreshold);
std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
PCA_Eigenvectors(const Eigen::MatrixXd &data, double varianceThreshold);

// --- Dynamical systems ---
std::vector<double> computeSwirlSeverityWindowed(
    const Eigen::MatrixXd &X, // N x d (already z-scored / EMA'd)
    double dt,
    int window, // e.g. 200
    int stride, // e.g. 20
    double speed_floor_percentile =
        0.10 // ignore slowest 10% speeds //default argu in .h only
);

// --- Baseline metrics ---
double eigenvalueDominance(const Eigen::MatrixXd &returns);
Eigen::VectorXd meanAbsoluteReturn(const Eigen::MatrixXd &window);
double covariancee(const std::vector<double> &a, const std::vector<double> &b);
std::vector<double> getColumns(const std::vector<std::vector<double>> &data,
                               int col);
std::vector<std::vector<double>>
covarianceMatrixx(const std::vector<std::vector<double>> &data);
std::vector<double>
volatilityFromCov(const std::vector<std::vector<double>> &cov);