#pragma once 
#include <Eigen\Dense>
#include <vector>

double median(const std::vector<double> &data);

double mean(const std::vector<double> &data);

std::vector<std::vector<double>> transposeData(const std::vector<std::vector<double>>& matrix);

double sampleVariance(const std::vector<double>& data);

Eigen::VectorXd toEigenVector(const std::vector<double>& vec);

Eigen::MatrixXd vectorToEigenMatrix(const std::vector<std::vector<double>>& data);

std::vector<std::vector<double>> matrixToVector2D(const Eigen::MatrixXd& mat);

Eigen::MatrixXd getInvCov(const Eigen::MatrixXd &covariance);

bool choleskyDecompLLT(const Eigen::MatrixXd &cov);

bool choleskyDecompLDLT(const Eigen::MatrixXd &cov);

bool areVectorsClose(const std::vector<double>& a, const std::vector<double>& b, double tol);

std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> CovarianceMatrix(std::vector<std::vector<std::vector<double>>>& clusters, const std::vector<std::vector<double>>& centroids, const std::vector<std::vector<double>>& meanvecs);

Eigen::MatrixXd computeGlobalCovariance(const std::vector<std::vector<double>>& data);

Eigen::MatrixXd computeRobustCovarianceMAD(const std::vector<std::vector<double>>& data);

double computeMAD(const std::vector<double>& values);

double mahalanobisDistanceSingle(const Eigen::VectorXd& point, const Eigen::VectorXd& centroid, const Eigen::MatrixXd& covariance);
double mahalanobisDistanceSingleFast(const Eigen::VectorXd& point, const Eigen::VectorXd& centroid, const Eigen::MatrixXd& covariance);


std::vector<double> MahalanobisDistMulti(
    const std::vector<double>& point,
    const std::vector<std::vector<double>>& meanvecs,
    const std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>& covmatrices
);

double cosineSimilarity(Eigen::Map<Eigen::VectorXd> &A, Eigen::Map<Eigen::VectorXd> &B);

std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> CovarianceMatrixGPT(
    const std::vector<std::vector<std::vector<double>>>& clusters,
    const std::vector<std::vector<double>>& centroids,
    const std::vector<std::vector<double>>& data);

bool allClustersHaveEnoughPoints(
    const std::vector<std::vector<std::vector<double>>>& clusters,
    size_t featureDim,
    size_t safetyFactor
);

double adaptiveAlpha(int clusterSize);

double distanceBlend(int iter, bool trigger, int startIter, int rampLength);

double avgClusterSize(const std::vector<std::vector<std::vector<double>>>& clusters);

std::vector<std::vector<double>> PCA(const Eigen::MatrixXd& data, double varianceThreshold);

double euclideanDist(const std::vector<double>& a, const std::vector<double>& b);