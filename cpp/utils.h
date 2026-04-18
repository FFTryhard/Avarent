
#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <iomanip>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <limits>
#include "rawprocessing.h"
#include <filesystem>
#include <deque>
#include <cstdio>

struct PolarCoordinates
{

    double r;
    double cos_t;
    double sin_t;

    // Define the '<' operator for sorting by radius only
    bool operator<(const PolarCoordinates &other) const
    {
        return this->r < other.r; // Compare only the radius
    }
};

struct Timer
{

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<float> duration;

    Timer()
    {
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        end = std::chrono::high_resolution_clock::now();

        duration = end - start;

        float ms = duration.count() * 1000.0f;
        std::cerr << "Timer took: " << ms << "ms" << std::endl;
    }
};

struct PCAdata
{
    std::vector<std::vector<double>> pcarawvecs;
    std::vector<std::vector<double>> pcaentropyvecs;
};

struct SinCos
{
    double sinval;
    double cosval;
};

struct Edge
{
    int to;
    double weight;
};

struct Buffer3
{
    double x0 = 0.0; // most recent   (x_k)
    double x1 = 0.0; // previous      (x_{k-1})
    double x2 = 0.0; // oldest        (x_{k-2})
    int count = 0;   // number of valid samples (0–3)
};

struct BufferJ
{
    int count = 0;
    std::vector<Eigen::MatrixXd> jacobians;
};

struct MatrixBuffer
{
    Eigen::MatrixXd prev;  // value at t-1
    Eigen::MatrixXd curr;  // value at t
    bool has_prev = false; // to know if prev is valid

    // Add a new scalar (shifts curr -> prev)
    void add(const Eigen::MatrixXd &M)
    {
        if (!init)
        {
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
    const Eigen::MatrixXd &previous() const
    {
        if (!has_prev)
            throw std::runtime_error("previous not available");
        return prev;
    }

private:
    bool init = false;
};

struct RollingBuffer
{
    std::vector<double> buf;
    size_t idx = 0;
    bool full = false;

    explicit RollingBuffer(size_t N) : buf(N, 0.0) {} // safer

    void push(double x)
    {
        buf[idx] = x;                 // std::cout << index
        idx = (idx + 1) % buf.size(); // idx + 1 / 5. whats whats left whats left
        if (idx == 0)
            full = true; // full
    } // order doesnt matter, we just do percintiles thats why circular is fine

    size_t size() const { return full ? buf.size() : idx; }
    // if full is true, return buf.size. if its false give the index

    // keeps cycling through the latest N values.

    std::vector<double> snapshot() const
    { // snapshot of all the valid values
        size_t n = size();
        std::vector<double> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i)
            out.push_back(buf[i]);
        return out;
    }
};

std::vector<std::string> tostringvector(const std::string &input);

double percentile(std::vector<double> &v, double p);
std::vector<std::vector<double>>
to_2d_vector(const std::vector<std::deque<double>> &input);
struct TauTracker
{
    RollingBuffer rb;
    double p;
    int stride;
    int counter = 0;
    double tau = 0.2;
    double alpha = 0.1;
    double lo = 0.02, hi = 0.8; // clamp range

    TauTracker(size_t N, double p_, int stride_)
        : rb(N), p(p_), stride(stride_) {}
    // make a buffer size, what percentile do we want, compute every N pushes

    void push(double x)
    {
        rb.push(x);
        counter++;
        if (!rb.full)
            return; // So percentile only computed after we have N samples.
        if (!(counter % stride == 0))
            return;

        std::vector<double> snap = rb.snapshot();
        double t = percentile(snap, p);
        if (std::isfinite(t))
        {
            tau = (1.0 - alpha) * tau + alpha * t; // smoothing
            tau = std::clamp(tau, lo, hi);
        }
    }

    bool ready() const { return rb.full; }
};

static std::string uniq(const char *prefix, const char *ext);
std::string runOllama(const std::string &prompt);
static std::string extractFirstJsonObject(const std::string &s);
static std::string buildPrompt_ExtractAndMap(const std::string &strategyText);
static std::string makeTempFileName();
std::vector<double> computeSwirlSeverityWindowed(
    const Eigen::MatrixXd& X, // N x d (already z-scored / EMA’d)
    double dt,
    int window,   // e.g. 200
    int stride,   // e.g. 20
    double speed_floor_percentile = 0.10 // ignore slowest 10% speeds //default argu in .h only
);
double perc_(std::vector<double> v, double p);
static std::string stripAnsi(const std::string &s);
Eigen::MatrixXd computeVelocities(const std::vector<std::vector<double>>& data, double dt);
void plot(std::vector<double> x1, std::vector<double> y1, std::vector<double> x2, std::vector<double> y2);
double dotProduct(const std::vector<double> &vec1, const std::vector<double> &vec2);

double cosineSimilarityCPP(const std::vector<double> &vec1, const std::vector<double> &vec2);

double magnitude(const std::vector<double> &vec);

bool containsNaN(const std::vector<std::vector<double>> &data);

bool containsINF(const std::vector<std::vector<double>> &data);

bool checkForNaN(const std::vector<double> &data);

std::vector<PolarCoordinates> makePolar(std::vector<double>, std::vector<double>);

void transposePolarSignalData(std::vector<std::vector<std::vector<PolarCoordinates>>> &data);

void transposeInPlace(std::vector<std::vector<double>> &matrix);

void print3dPolar(std::vector<std::vector<PolarCoordinates>> data, bool all, int targetcluster, int);

void logTransformAndScale(std::vector<double> &mahalanobisdistances, double minRange, double maxRange);
double clamp_(double x);
std::vector<std::vector<double>> generatePoints(int numVectors, int vectorSize);

void dumpPolarCSV(const std::string &filename,
                  const std::vector<std::vector<double>> &rows,
                  int numClusters,
                  const std::vector<std::string> &cluster_names);

std::vector<double> FourierExpansion(double x, int d, int repeat_count);
double derivativeOLS(const std::vector<double> &v, double dt);
template <typename T>
inline std::vector<T> deque_to_vector_template(const std::deque<T> &src)
{
    std::vector<T> dst;
    dst.reserve(src.size());

    for (const auto &v : src)
    {
        dst.push_back(v);
    }

    return dst;
}

std::vector<double> getCosSim(const std::vector<std::vector<double>> &centroids, std::vector<double> &p);
size_t argmax_(const std::vector<double> &v);
double operatorShockWedin(const Eigen::MatrixXd &Et, double gap);
std::vector<double> eigenToStdVector(const Eigen::VectorXd &v);
double dominantSubspaceRotation(double sigmaMIN);
double medianNTE(const std::vector<double> &data);
double effective_RankCPP(const std::vector<double> &vec);
double wedin_theoremKGT1(
    const Eigen::MatrixXd &M_overlap);
double wedin_theoremK1(const Eigen::VectorXd &u_cur, const Eigen::VectorXd &u_prev);
int effective_rank_energy(std::vector<double> v);
std::size_t argmin(
    const std::vector<double> &vec,
    double *min_value // pass nullptr if you don't want the value
);
double argmax(const std::vector<double> &v);
double argminV(const std::vector<double> &v);
double effective_rank(const std::vector<double> &eigvals);
bool isNoisySignal(
    const Eigen::MatrixXd &X,
    double threshold // >5 is usually noise-dominated
);
bool computeVelocityFromBuffer(
    Buffer3 &buf,
    double dt,
    double &velocity);
void push(Buffer3 &buf, double x);
Eigen::MatrixXd dequeToEigenMatrix(const std::deque<std::vector<double>> &data);
std::vector<std::vector<double>> deque_to_vector(const std::deque<std::vector<double>> &src);
double effective_Rank(const Eigen::VectorXd &sigma);
std::vector<double> normalizeVector(const std::vector<double> &data, double mag);
double median(const std::vector<double> &data);
double dotProduct(const std::vector<double> &vec1, const std::vector<double> &vec2);
double mean(const std::vector<double> &data);
double magnitude(const std::vector<double> &vec);
std::vector<std::vector<double>> transposeData(const std::vector<std::vector<double>> &matrix);
double calculateStdDev(const std::vector<double> &vec, double mean);
double sampleVariance(const std::vector<double> &data);
std::vector<double> zscore(const std::vector<double> &vec);
Eigen::VectorXd toEigenVector(const std::vector<double> &vec);
Eigen::MatrixXd normalizeColumns(Eigen::MatrixXd &X, double epsilon);
Eigen::MatrixXd vectorToEigenMatrix(const std::vector<std::vector<double>> &data);
int kthSmallestIndexDouble(const std::vector<double> &v, int k);
std::vector<std::vector<double>> matrixToVector2D(const Eigen::MatrixXd &mat);
double variance(const std::vector<double> &v);
Eigen::MatrixXd getInvCov(const Eigen::MatrixXd &covariance);

bool choleskyDecompLLT(const Eigen::MatrixXd &cov);
std::vector<double> deque_to_1vector(const std::deque<double> &src);
bool choleskyDecompLDLT(const Eigen::MatrixXd &cov);
double computeR2(const std::vector<double> &x, const std::vector<double> &y);
bool areVectorsClose(const std::vector<double> &a, const std::vector<double> &b, double tol);

std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> CovarianceMatrix(std::vector<std::vector<std::vector<double>>> &clusters, const std::vector<std::vector<double>> &centroids, const std::vector<std::vector<double>> &meanvecs);

Eigen::MatrixXd computeGlobalCovariance(const std::vector<std::vector<double>> &data);

Eigen::MatrixXd computeRobustCovarianceMAD(const std::vector<std::vector<double>> &data);

double computeMAD(const std::vector<double> &values);
std::tuple<Eigen::MatrixXd, Eigen::VectorXd> PCA_Eigenvectors(const Eigen::MatrixXd &data, double varianceThreshold);
double mahalanobisDistanceSingle(const Eigen::VectorXd &point, const Eigen::VectorXd &centroid, const Eigen::MatrixXd &covariance);
double mahalanobisDistanceSingleFast(const Eigen::VectorXd &point, const Eigen::VectorXd &centroid, const Eigen::MatrixXd &covariance);

std::vector<double> MahalanobisDistMulti(
    const std::vector<double> &point,
    const std::vector<std::vector<double>> &meanvecs,
    const std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> &covmatrices);

// double cosineSimilarity(const Eigen::VectorXd> &A, const Eigen::VectorXd &B); //added const
double cosineSimilarityCPP(const std::vector<double> &vec1, const std::vector<double> &vec2);
double cosineSimilarity(const Eigen::VectorXd &A,
                        const Eigen::VectorXd &B);

std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> CovarianceMatrixGPT(
    const std::vector<std::vector<std::vector<double>>> &clusters,
    const std::vector<std::vector<double>> &centroids,
    const std::vector<std::vector<double>> &data);

bool allClustersHaveEnoughPoints(
    const std::vector<std::vector<std::vector<double>>> &clusters,
    size_t featureDim,
    size_t safetyFactor);

double adaptiveAlpha(int clusterSize);

double distanceBlend(int iter, bool trigger, int startIter, int rampLength);

double avgClusterSize(const std::vector<std::vector<std::vector<double>>> &clusters);

std::vector<std::vector<double>> PCA(const Eigen::MatrixXd &data, double varianceThreshold);

double euclideanDist(const std::vector<double> &a, const std::vector<double> &b);

std::vector<std::vector<double>> distanceMatrix(const std::vector<std::vector<double>> &data);

std::vector<std::vector<std::pair<double, int>>> alignedDistanceList(const std::vector<std::vector<double>> &distanceMatrix, const std::vector<std::vector<int>> &knnmatrix);
std::vector<std::vector<Edge>> adjacencyList(const std::vector<std::vector<std::pair<double, int>>> &list);
double dijkstra_AB(const std::vector<std::vector<Edge>> &graph, int A, int B);

std::vector<std::pair<double, int>> getAdjRow(const std::vector<double> &ForeignVector, const std::vector<std::vector<double>> &normaldata, int num_neighbours);
int insertForeignNode(const std::vector<std::pair<double, int>> &neighbours, std::vector<std::vector<Edge>> &graph);
void removeForeignNode(int F, const std::vector<std::pair<double, int>> &neighbors, std::vector<std::vector<Edge>> &graph);
std::vector<double> dijkstra_from_source(const std::vector<std::vector<Edge>> &graph, int src);
std::vector<std::vector<double>> precomputeGeodesicDistances(const std::vector<std::vector<Edge>> &graph);
void generateInvertedMobiusStrip(std::vector<std::vector<double>> &SwissRollshapedData, int numRows, int numDims);
int argmin(const std::vector<std::pair<double, int>> &vec);
std::vector<int> Medoids(const std::vector<std::vector<double>> &data, const std::vector<std::vector<int>> &clusterIndices, const Eigen::MatrixXd &Psi, bool geodesic);
// LEARN ===============================================
struct BridgeEdge
{
    int u, v;
    double dist;
};
std::vector<std::vector<int>> connectedComponents(const std::vector<std::vector<Edge>> &adj);
BridgeEdge findClosestBridge( // returns an object
    const std::vector<int> &A,
    const std::vector<int> &B,
    const std::vector<std::vector<double>> &distanceMatrix);
void connectComponents(
    std::vector<std::vector<Edge>> &adj,
    const std::vector<std::vector<double>> &distanceMatrix);

double meanPAIR(const std::vector<std::pair<double, int>> &neighbors);
int kthSmallestIndex(const std::vector<int> &v, int k);
std::vector<std::vector<double>> shuffle_columns(const std::vector<std::vector<double>> &dataset);
int kthSmallestIndexPairs(const std::vector<std::pair<double, int>> &v, int k);
void signFlipRandomCols(std::vector<std::vector<double>> &dataset);
Eigen::MatrixXd savitzkyGolaySmooth(
    const Eigen::MatrixXd &X);

//DUMB BASELINES
double eigenvalueDominance(const Eigen::MatrixXd& returns);

Eigen::VectorXd meanAbsoluteReturn(const Eigen::MatrixXd& window);
// Compute covariance between two vectors
double covariancee(const std::vector<double>& a, const std::vector<double>& b);

// Extract column from matrix
std::vector<double> getColumns(const std::vector<std::vector<double>>& data, int col);

// Compute covariance matrix for the entire matrix
std::vector<std::vector<double>> covarianceMatrixx(const std::vector<std::vector<double>>& data);
std::vector<double> volatilityFromCov(const std::vector<std::vector<double>>& cov);