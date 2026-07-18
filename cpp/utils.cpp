#include "utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

#define LOG(x) std::cerr << x << '\n';

// ============================================================
//  Validation
// ============================================================

bool checkForNaN(const std::vector<double> &data)
{
  for (size_t i = 0; i < data.size(); ++i)
  { 
    if (std::isnan(data[i]) || data[i] == 0)
    {
      return true;
    }
  }
  return false;
}

bool containsNaN(const std::vector<std::vector<double>> &data)
{
  for (const auto &row : data)
  {
    for (double value : row)
    {
      if (std::isnan(value))
      {
        return true;
      }
    }
  }
  return false;
}

bool containsINF(const std::vector<std::vector<double>> &data)
{
  for (const auto &row : data)
  {
    for (double value : row)
    {
      if (!std::isfinite(value))
      {
        return true;
      }
    }
  }
  return false;
}

// ============================================================
//  Scalar utilities
// ============================================================

double clamp_(double x) { return std::max(0.0, std::min(1.0, x)); }

double median(std::vector<double> vec)
{
  std::sort(vec.begin(), vec.end());

  if (vec.size() % 2 == 0)
  {
    return 0.5 * (vec[vec.size() / 2 - 1] + vec[vec.size() / 2]);
  }
  else
  {
    return vec[vec.size() / 2];
  }
}

double mean(const std::vector<double> &data)
{

  double sum = 0.0;

  for (auto val : data)
  {
    sum = sum + val;
  }

  if (sum == 0.0)
  {
    return 0.0;
  }

  return sum / data.size();
}

double computeMAD(const std::vector<double> &values)
{
  std::vector<double> absDeviations(values.size());
  double med = 0;

  // Compute the median of the values
  std::vector<double> sortedValues = values;
  std::sort(sortedValues.begin(), sortedValues.end());

  size_t n = sortedValues.size();
  if (n % 2 == 0)
  {
    med = (sortedValues[n / 2 - 1] + sortedValues[n / 2]) / 2.0;
  }
  else
  {
    med = sortedValues[n / 2];
  }

  // Calculate the absolute deviations from the median
  for (size_t i = 0; i < values.size(); ++i)
  {
    absDeviations[i] = std::abs(values[i] - med);
  }

  // Compute the median of the absolute deviations
  std::sort(absDeviations.begin(), absDeviations.end());
  double mad = 0;
  if (n % 2 == 0)
  {
    mad = (absDeviations[n / 2 - 1] + absDeviations[n / 2]) / 2.0;
  }
  else
  {
    mad = absDeviations[n / 2];
  }

  return mad;
}

double computeR2(const std::vector<double> &x, const std::vector<double> &y)
{
  if (x.size() != y.size() || x.empty())
  {
    throw std::runtime_error("Vectors must have same non-zero size");
  }

  size_t n = x.size();

  // Compute means
  double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / n;
  double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / n;

  double sum_xy = 0.0;
  double sum_x2 = 0.0;
  double sum_y2 = 0.0;

  for (size_t i = 0; i < n; ++i)
  {
    double dx = x[i] - mean_x;
    double dy = y[i] - mean_y;

    sum_xy += dx * dy;
    sum_x2 += dx * dx;
    sum_y2 += dy * dy;
  }

  if (sum_x2 == 0.0 || sum_y2 == 0.0)
  {
    throw std::runtime_error("Zero variance in input");
  }

  return (sum_xy * sum_xy) / (sum_x2 * sum_y2);
}

double derivativeOLS(const std::vector<double> &v, double dt)
{
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  int n = (int)v.size();

  for (int k = 0; k < n; ++k)
  {
    double t = k * dt;
    sx += t;
    sy += v[k];
    sxx += t * t;
    sxy += t * v[k];
  }

  double denom = (n * sxx - sx * sx);
  if (std::abs(denom) < 1e-12)
    return 0.0;

  return (n * sxy - sx * sy) / denom;
}

double perc_(std::vector<double> v, double p)
{
  if (v.empty())
    return 0.0;
  p = std::min(1.0, std::max(0.0, p));
  size_t k = (size_t)std::floor(p * (v.size() - 1));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

// ============================================================
//  Conversion utilities
// ============================================================

Eigen::MatrixXd
dequeToEigenMatrix(const std::deque<std::vector<double>> &data)
{
  if (data.empty())
    return Eigen::MatrixXd{}; // empty matrix

  const std::size_t rows = data.size();
  const std::size_t cols = data.front().size();

  // Optional: sanity check
  for (const auto &v : data)
  {
    if (v.size() != cols)
      throw std::runtime_error("Inconsistent vector sizes");
  }

  Eigen::MatrixXd mat(rows, cols);

  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j)
      mat(i, j) = static_cast<double>(data[i][j]);

  return mat;
}

std::vector<std::vector<double>>
deque_to_vector(const std::deque<std::vector<double>> &src)
{
  std::vector<std::vector<double>> dst;
  dst.reserve(src.size()); // avoid reallocations

  for (const auto &v : src)
  {
    dst.push_back(v); // copies each vector
  }

  return dst;
}

std::vector<double> deque_to_1vector(const std::deque<double> &src)
{
  std::vector<double> dst;
  dst.reserve(src.size());

  for (const auto &v : src)
  {
    dst.push_back(v);
  }

  return dst;
}

Eigen::VectorXd toEigenVector(const std::vector<double> &vec)
{
  Eigen::VectorXd eigenVec(vec.size());
  for (size_t i = 0; i < vec.size(); ++i)
  {
    eigenVec(i) = vec[i];
  }
  return eigenVec;
}

Eigen::MatrixXd
vectorToEigenMatrix(const std::vector<std::vector<double>> &data)
{

  if (data.empty())
  {
    std::cerr << "return empty matrix" << std::endl;
    return Eigen::MatrixXd(); // return empty matrix
  }

  size_t rows = data.size();
  size_t cols = data[0].size();

  // Verify all rows have the same size
  for (const auto &row : data)
  {
    if (row.size() != cols)
    {
      throw std::runtime_error(
          "All rows in the 2D vector must have the same length.");
    }
  }

  Eigen::MatrixXd mat(rows, cols);
  for (size_t i = 0; i < rows; ++i)
  {
    for (size_t j = 0; j < cols; ++j)
    {
      mat(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          data[i][j];
    }
  }

  return mat;
}

std::vector<double> eigenToStdVector(const Eigen::VectorXd &v)
{
  return std::vector<double>(v.data(), v.data() + v.size());
}

// ============================================================
//  Linear algebra / spectral
// ============================================================

double effective_Rank(const Eigen::VectorXd &sigma)
{

  double sum_sqrd = 0.0;

  for (auto sv : sigma)
  {
    sum_sqrd = sum_sqrd + sv;
  }
  sum_sqrd = sum_sqrd * sum_sqrd;

  double sum_sqrd_sv = 0.0;
  for (auto sv : sigma)
  {
    sum_sqrd_sv = sum_sqrd_sv + (sv * sv);
  }

  double r_eff =
      sum_sqrd / sum_sqrd_sv; // r_eff≈1: one sane mode, If 𝑟_eff ≈ 2: two sane
                              // modes and if r_eff ≫1: garbage / noise

  return r_eff;
}

double wedin_theoremKGT1(const Eigen::MatrixXd &M_overlap)
{

  Eigen::JacobiSVD<Eigen::MatrixXd> svdM(M_overlap);

  auto sigmaM = svdM.singularValues();
  double sigmaMIN = std::clamp(sigmaM.minCoeff(), 0.0, 1.0);
  double s = std::min(1.0, std::max(0.0, sigmaMIN));
  double sin_norm = std::sqrt(1.0 - s * s);

  LOG("Observed Subspace rotation (sin_norm): " << sin_norm);

  return sin_norm;
}

double wedin_theoremK1(const Eigen::VectorXd &u_cur_in,
                       const Eigen::VectorXd &u_prev_in)
{
  // normalize defensively (SVD U should already be orthonormal, but don't trust
  // anything)
  Eigen::VectorXd u_prev = u_prev_in;
  Eigen::VectorXd u_cur = u_cur_in;

  u_prev.normalize();
  u_cur.normalize();

  double cos_theta = std::abs(u_prev.dot(u_cur));
  cos_theta = std::clamp(cos_theta, 0.0, 1.0);

  double sin_norm = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));

  return sin_norm;
}

double cosineSimilarity(const Eigen::VectorXd &A,
                        const Eigen::VectorXd &B)
{ // takes 2 vectors

  float epsilon = 1e-10f;

  double dot_product = A.dot(B); // A · B
  double norm_A = A.norm();      // ||A||
  double norm_B = B.norm();      // ||B||

  if (norm_A < epsilon || norm_B < epsilon)
  {
    return 0.0;
  }

  return dot_product / (norm_A * norm_B);
}

// ============================================================
//  PCA
// ============================================================

// Combined PCA: single eigen decomposition returning everything the engine
// needs. Eliminates the duplicate covariance + eigen solve that PCA() and
// PCA_Eigenvectors() performed separately.

PCA_Result PCA_Full(const Eigen::MatrixXd &data, double varianceThreshold)
{
  if (data.rows() == 0 || data.cols() == 0)
    throw std::runtime_error("Input data matrix is empty!");
  if (data.rows() == 1)
    throw std::runtime_error("PCA requires at least 2 rows of data.");

  varianceThreshold = std::clamp(varianceThreshold, 0.0, 1.0);

  // Step 1: Mean-center
  const Eigen::MatrixXd centered = data.rowwise() - data.colwise().mean();

  // Step 2: Covariance matrix with Tikhonov regularization
  Eigen::MatrixXd cov =
      (centered.adjoint() * centered) / double(data.rows() - 1);
  double lambda = 1e-3 * cov.trace() / cov.rows();
  cov += lambda * Eigen::MatrixXd::Identity(cov.rows(), cov.cols());

  // Step 3: Eigen decomposition (ascending order from Eigen)
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
  if (eig.info() != Eigen::Success)
    throw std::runtime_error("Eigen decomposition failed.");

  Eigen::VectorXd eigenValues = eig.eigenvalues();
  Eigen::MatrixXd eigenVectors = eig.eigenvectors();

  // Step 4: Reverse to descending order
  Eigen::VectorXd eigenValuesDesc = eigenValues.reverse().eval();
  Eigen::MatrixXd eigenVectorsDesc(eigenVectors.rows(), eigenVectors.cols());
  for (int i = 0; i < eigenVectors.cols(); ++i)
    eigenVectorsDesc.col(i) = eigenVectors.col(eigenVectors.cols() - 1 - i);

  // Step 5: Determine k components to keep
  double totalVariance = eigenValuesDesc.sum();
  double runningSum = 0.0;
  int k = (int)eigenValuesDesc.size();
  for (int i = 0; i < eigenValuesDesc.size(); ++i)
  {
    runningSum += eigenValuesDesc(i);
    if (runningSum / totalVariance >= varianceThreshold)
    {
      k = i + 1;
      break;
    }
  }

  std::cerr << "Keeping " << k << " principal components.\n";

  // Step 6: Project data
  Eigen::MatrixXd reduced = centered * eigenVectorsDesc.leftCols(k);

  // Step 7: Convert to std::vector
  std::vector<std::vector<double>> result(reduced.rows(),
                                          std::vector<double>(reduced.cols()));
  for (int i = 0; i < reduced.rows(); ++i)
    for (int j = 0; j < reduced.cols(); ++j)
      result[i][j] = reduced(i, j);

  return {std::move(result), eigenVectorsDesc.leftCols(k),
          eigenValuesDesc.head(k)};
}

// Backward-compatible wrappers (delegate to PCA_Full)
std::vector<std::vector<double>> PCA(const Eigen::MatrixXd &data,
                                     double varianceThreshold)
{
  return PCA_Full(data, varianceThreshold).projected;
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
PCA_Eigenvectors(const Eigen::MatrixXd &data, double varianceThreshold)
{
  auto r = PCA_Full(data, varianceThreshold);
  return {std::move(r.eigenvectors), std::move(r.eigenvalues)};
}

// ============================================================
//  Dynamical systems — swirl severity
// ============================================================

// returns one SRS value per window
std::vector<double> computeSwirlSeverityWindowed(
    const Eigen::MatrixXd &X, // N x d (already z-scored / EMA'd)
    double dt,
    int window,                   // e.g. 200
    int stride,                   // e.g. 20
    double speed_floor_percentile // ignore slowest 10% speeds
)
{
  const double eps = 1e-12;
  const int N = (int)X.rows();
  const int d = (int)X.cols();
  if (N < window + 2)
    return {};

  // velocity (central diff)
  Eigen::MatrixXd V(N, d);
  V.row(0) = (X.row(1) - X.row(0)) / dt;
  for (int i = 1; i < N - 1; ++i)
    V.row(i) = (X.row(i + 1) - X.row(i - 1)) / (2.0 * dt);
  V.row(N - 1) = (X.row(N - 1) - X.row(N - 2)) / dt;

  // acceleration (central diff on V)
  Eigen::MatrixXd A(N, d);
  A.row(0) = (V.row(1) - V.row(0)) / dt;
  for (int i = 1; i < N - 1; ++i)
    A.row(i) = (V.row(i + 1) - V.row(i - 1)) / (2.0 * dt);
  A.row(N - 1) = (V.row(N - 1) - V.row(N - 2)) / dt;

  std::vector<double> out;
  for (int start = 0; start + window <= N; start += stride)
  {
    int end = start + window;

    // collect speeds to set a floor
    std::vector<double> speeds;
    speeds.reserve(window);
    for (int i = start; i < end; ++i)
      speeds.push_back(V.row(i).norm());
    double v_floor = perc_(speeds, speed_floor_percentile);
    double v_floor2 = v_floor * v_floor;

    // compute omega samples, skip low-speed
    std::vector<double> omega;
    omega.reserve(window);

    for (int i = start; i < end; ++i)
    {
      Eigen::VectorXd v = V.row(i);
      Eigen::VectorXd a = A.row(i);

      double s1 = v.squaredNorm();
      if (s1 <= std::max(v_floor2, 1e-10))
        continue;

      double s2 = a.squaredNorm();
      double s3 = v.dot(a);

      double gram = s1 * s2 - s3 * s3; // ||v||^2||a||^2 - (v·a)^2 = ||v x a||^2
      if (gram < 0.0)
        gram = 0.0;

      double w = std::sqrt(gram) / (s1 + eps); // turning rate (1/time)
      omega.push_back(w);
    }

    if (omega.size() < 10)
    {
      out.push_back(0.0);
      continue;
    }

    // robust summary: 90th percentile of omega (captures "severe swirl" without
    // RMS spikes)
    double w_p90 = perc_(omega, 0.90);

    // optional normalization by typical speed to get curvature-per-distance
    // (1/length)
    double v_med = perc_(speeds, 0.50);
    double srs = w_p90 / (v_med + eps);

    out.push_back(srs);
  }

  return out;
}

// ============================================================
//  Baseline metrics
// ============================================================

double eigenvalueDominance(const Eigen::MatrixXd &returns)
{
  int W = (int)returns.rows();

  Eigen::MatrixXd centered = returns.rowwise() - returns.colwise().mean();
  Eigen::MatrixXd cov = (centered.transpose() * centered) / (W - 1);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);

  double lambda_max = solver.eigenvalues().maxCoeff();
  double trace = cov.trace();

  return lambda_max / trace;
}

Eigen::VectorXd meanAbsoluteReturn(const Eigen::MatrixXd &window)
{
  int W = (int)window.rows();

  Eigen::VectorXd mar = window.cwiseAbs().colwise().sum() / W;

  return mar;
}

// Compute covariance between two vectors
double covariancee(const std::vector<double> &a, const std::vector<double> &b)
{
  double ma = mean(a);
  double mb = mean(b);

  double cov = 0.0;
  for (size_t i = 0; i < a.size(); i++)
  {
    cov += (a[i] - ma) * (b[i] - mb);
  }

  return cov / (a.size() - 1);
}

// Extract column from matrix
std::vector<double> getColumns(const std::vector<std::vector<double>> &data,
                               int col)
{
  std::vector<double> result(data.size());
  for (size_t i = 0; i < data.size(); i++)
  {
    result[i] = data[i][col];
  }
  return result;
}

// Compute covariance matrix for the entire matrix
std::vector<std::vector<double>>
covarianceMatrixx(const std::vector<std::vector<double>> &data)
{
  int N = (int)data[0].size(); // number of columns
  std::vector<std::vector<double>> cov(N, std::vector<double>(N));

  // Cache all column extractions once (avoids N² redundant copies)
  std::vector<std::vector<double>> cols(N);
  for (int i = 0; i < N; i++)
    cols[i] = getColumns(data, i);

  for (int i = 0; i < N; i++)
  {
    for (int j = i; j < N; j++)
    {
      double c = covariancee(cols[i], cols[j]);
      cov[i][j] = c;
      cov[j][i] = c; // exploit symmetry
    }
  }

  return cov;
}

// Convert covariance matrix to volatility vector
std::vector<double>
volatilityFromCov(const std::vector<std::vector<double>> &cov)
{
  int N = (int)cov.size();
  std::vector<double> vol(N);
  for (int i = 0; i < N; i++)
  {
    vol[i] = std::sqrt(cov[i][i]);
  }
  return vol;
}

//------------------------------------------

double distanceSquared(const std::vector<double> &a, const std::vector<double> &b) 
{
  double d = 0.0;

  for (size_t i = 0; i < a.size(); ++i)
  {
    double diff = a[i] - b[i];
    d += diff * diff;
  }

  return d;
}

//------------------------------------------