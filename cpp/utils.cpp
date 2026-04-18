#include <iostream>
#include "utils.h"
#include <limits>
#include <queue>
#include <algorithm>
#include <execution>
#include <ctime>
#include <cmath>
#include <random>
#include <deque>

#define LOG(x) std::cerr << x << '\n';

bool checkForNaN(const std::vector<double> &data)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        if (std::isnan(data[i] || data[i] == 0))
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

std::vector<double> getCosSim(const std::vector<std::vector<double>> &medoids, std::vector<double> &point)
{

    std::vector<double> cossims;

    for (auto c : medoids)
    {

        if (c.size() != point.size())
        {
            std::cerr << "dimension mismatch!" << std::endl;
        }

        cossims.push_back(cosineSimilarityCPP(c, point));
    }

    return cossims;
}

// file << stream_chunks[0].getTime(x);// << "ms";  // First column is the time step, numerical for now   //IT WAS READING AN EMPTY VECTOR. blew the whole fucking program up  bro

// cluster_names is optional, e.g. {"cluster1","cluster2",...}
// rows = list of points, each point has (r1,theta1,r2,theta2,...)
void dumpPolarCSV(const std::string &filename,
                  const std::vector<std::vector<double>> &rows,
                  int numClusters,
                  const std::vector<std::string> &cluster_names)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // Header row
    if (cluster_names.empty())
    {
        for (int c = 0; c < numClusters; ++c)
        {
            file << "r_cluster" << (c + 1) << ",theta_cluster" << (c + 1);
            if (c != numClusters - 1)
                file << ",";
        }
    }
    else
    {
        for (int c = 0; c < numClusters; ++c)
        {
            file << "r_" << cluster_names[c] << ",theta_" << cluster_names[c];
            if (c != numClusters - 1)
                file << ",";
        }
    }
    file << "\n";

    // Each row of data
    for (const auto &point : rows)
    {
        for (size_t i = 0; i < point.size(); ++i)
        {
            file << point[i];
            if (i != point.size() - 1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
}

std::vector<std::vector<std::complex<double>>> readPolarCSV(const std::string &filepath, int numClusters)
{
    std::ifstream infile(filepath);
    if (!infile.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    std::string line;
    std::getline(infile, line); // skip header

    // one trajectory per cluster
    std::vector<std::vector<std::complex<double>>> clusterTrajectories(numClusters);

    while (std::getline(infile, line))
    {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> values;

        while (std::getline(ss, cell, ','))
        {
            values.push_back(std::stod(cell));
        }

        if (values.size() != 2 * numClusters)
        {
            throw std::runtime_error("Row length does not match 2*numClusters");
        }

        for (int j = 0; j < numClusters; j++)
        {
            double r = values[2 * j];
            double theta = values[2 * j + 1];
            double re = r * std::cos(theta);
            double im = r * std::sin(theta);
            clusterTrajectories[j].emplace_back(re, im);
        }
    }

    return clusterTrajectories;
}

std::vector<std::vector<PolarCoordinates>> slidingWindows(const std::vector<PolarCoordinates> &data, size_t windowSize, size_t stride)
{

    // data is a vector that holds a bunch of points with polar data.

    std::vector<std::vector<PolarCoordinates>> windows; // holds windows [window][data (metric of clusters)]
    if (windowSize == 0 || data.size() < windowSize)
        return windows;

    for (size_t start = 0; start + windowSize <= data.size(); start += stride)
    {
        windows.emplace_back(data.begin() + start, data.begin() + start + windowSize);
    }
    return windows; // slid window over all points/objects in data, each row is the window
}

std::vector<std::vector<double>> generatePoints(int numVectors, int vectorSize)
{
    std::vector<std::vector<double>> points;

    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<double> noiseDist(-0.005, 0.005); // subtle noise
    std::uniform_real_distribution<double> stepDist(0.001, 0.01);    // very small steps

    double value = 0.0;     // persistent value across vectors
    bool trendingUp = true; // start trending up
    int trendLength = 50;   // how long to keep a trend direction
    int trendCounter = 0;   // track steps in current trend

    for (int i = 0; i < numVectors; ++i)
    {
        std::vector<double> newVector;

        for (int j = 0; j < vectorSize; ++j)
        {
            double step = stepDist(rng);
            double noise = noiseDist(rng);

            if (trendingUp)
            {
                value += step + noise;
            }
            else
            {
                value -= step + noise;
            }

            // Clamp between -1 and 1
            value = std::clamp(value, -1.0, 1.0);

            newVector.push_back(value);
        }

        points.push_back(newVector);

        // Update trend counter and possibly reverse direction
        trendCounter++;
        if (trendCounter >= trendLength)
        {
            trendingUp = !trendingUp;      // flip trend direction
            trendCounter = 0;              // reset counter
            trendLength = rng() % 40 + 40; // randomize next trend length (40–80 vectors)
        }
    }

    return points;
}

std::vector<PolarCoordinates> makePolar(std::vector<double> geodesics, std::vector<double> csims)
{ //(R, THETA)

    if (geodesics.size() != csims.size())
    {
        std::cerr << "[ERROR] makePolar: mdist and csims size mismatch!" << std::endl;
        return {};
    }

    std::vector<PolarCoordinates> polarData; // per CLUSTER
    polarData.resize(csims.size());          // shouldnt matter that i used csims

    for (int i = 0; i < csims.size(); i++)
    {
        double r = geodesics[i];
        double theta = std::clamp(csims[i], -1.0, 1.0); // uneven + safety

        polarData[i].r = r;
        polarData[i].cos_t = std::cos(theta); // embed to cartesian
        polarData[i].sin_t = std::sin(theta);
    }

    return polarData;
}

void logTransformAndScale(std::vector<double> &mahalanobisdistances, double minRange, double maxRange)
{
    // Step 1: Log transform all distances first
    for (auto &dist : mahalanobisdistances)
    {
        dist = std::log(dist + 1);
    }

    // Step 2: Find min and max in log domain
    double globalMin = *std::min_element(mahalanobisdistances.begin(), mahalanobisdistances.end());
    double globalMax = *std::max_element(mahalanobisdistances.begin(), mahalanobisdistances.end());

    // Step 3: Handle edge case if min == max
    if (globalMax == globalMin)
    {
        for (auto &dist : mahalanobisdistances)
        {
            dist = minRange; // or (minRange + maxRange)/2
        }
        return;
    }

    // Step 4: Min-max scale log distances to [minRange, maxRange]
    for (auto &dist : mahalanobisdistances)
    {
        dist = (dist - globalMin) / (globalMax - globalMin) * (maxRange - minRange) + minRange;
    }
}

std::vector<double> FourierExpansion(double x, int d, int repeat_count)
{

    double r_error = x;
    double w = 0;

    std::vector<double> features;

    double *freqs;
    freqs = new double[d];

    for (int idx = 0; idx < d; idx++)
    {
        int i = idx / repeat_count;
        freqs[idx] = exp2(i);

        // fill repeated values, int gets rounded. im doing this so multiple dimensions get same frequencies. just to see what happens
        //  same as: * (freqs + x)
    }

    for (int i = 0; i < d; i++)
    {                                     // INCREMENT OVER FREQEUNCY CHANGES
        w = freqs[i];                     // MAKE 10000 DYNAMIC SO ITS A sFOURIER FEATURE ENCODER. make it an array with freqs
        double sinval = sin(r_error * w); // hardcoded 10000 = argument becomes to tiny the wave function collapses
        double cosval = cos(r_error * w); // true fourier not divisions
        features.push_back(sinval);
        features.push_back(cosval);
    }

    delete[] freqs;
    freqs = nullptr;

    return features; // keep in mind output = d * 2; sin + cos
}

void transposeInPlace(std::vector<std::vector<double>> &matrix)
{
    if (matrix.empty() || matrix[0].empty())
    {
        matrix.clear();
        return;
    }

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::vector<std::vector<double>> result(cols, std::vector<double>(rows));

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            result[j][i] = matrix[i][j];
        }
    }

    matrix = std::move(result); // replace the orignal matrix to T matrix. faster
}

// Function to transpose the 3D vector
void transposePolarSignalData(std::vector<std::vector<std::vector<PolarCoordinates>>> &data)
{

    size_t xDim = data.size();
    size_t yDim = data[0].size();
    size_t zDim = data[0][0].size();

    std::vector<std::vector<std::vector<PolarCoordinates>>> transposedData(
        zDim, std::vector<std::vector<PolarCoordinates>>(yDim, std::vector<PolarCoordinates>(xDim)));

    for (size_t i = 0; i < xDim; ++i)
    {
        for (size_t j = 0; j < yDim; ++j)
        {
            for (size_t k = 0; k < zDim; ++k)
            {
                transposedData[k][j][i] = data[i][j][k];
            }
        }
    }
}

void print3dPolar(std::vector<std::vector<PolarCoordinates>> data, bool all, int targetcluster, int polarorder)
{

    for (auto vector : data)
    { // each point has 2 signals ==
        if (all)
        {
            for (int i = 0; i < vector.size(); i++)
            { // this changes, right now EACH signaltype vector stores metrics of different clusters. but we want metrics of the same cluster
                if (polarorder == 0)
                {
                    std::cerr << "(" << vector[i].r << ", " << atan2(vector[i].sin_t, vector[i].cos_t) << ")" << std::endl;
                }
                else
                {
                    std::cerr << "(" << atan2(vector[i].sin_t, vector[i].cos_t) << ", " << vector[i].r << ")" << std::endl;
                }
            }
        }
        else
        {
            if (polarorder == 0)
            {
                std::cerr << "(" << vector[targetcluster].r << ", " << atan2(vector[targetcluster].sin_t, vector[targetcluster].cos_t) << ")" << std::endl;
            }
            else
            {
                std::cerr << "(" << atan2(vector[targetcluster].sin_t, vector[targetcluster].cos_t) << ", " << vector[targetcluster].r << ")" << std::endl;
            }
        }
    }
    std::cerr << std::endl;
    std::cerr << std::endl;
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> getThetasAndRs(
    const std::vector<std::vector<PolarCoordinates>> &data)
{
    if (data.empty())
    {
        throw std::invalid_argument("data empty (getThetas)");
    }

    std::cerr << data.size();
    std::cerr << data[0].size();

    // data has 1000 rows (objects), and k columns (clusters)
    size_t numRows = data.size();
    size_t numClusters = data[0].size();

    // Ensure all rows have same cluster count
    for (const auto &row : data)
    {
        if (row.size() != numClusters)
        {
            throw std::invalid_argument("inconsistent row sizes (getThetas)");
        }
    }

    // result will have 2 columns (clusters), each containing 1000 thetas
    std::vector<std::vector<double>> thetas(numClusters);
    for (auto &col : thetas)
        col.reserve(numRows);

    std::vector<std::vector<double>> rs(numClusters);
    for (auto &col : rs)
        col.reserve(numRows);

    // Loop through data and invert row/column layout
    for (const auto &row : data)
    {
        for (size_t i = 0; i < numClusters; i++)
        {
            double theta = std::atan2(row[i].sin_t, row[i].cos_t);
            thetas[i].push_back(theta); // Now safe

            double r = row[i].r;
            rs[i].push_back(r); // same filling scheme as theta. [i] is cluster, gets filled up lile this ||||| ->
        }
    }

    std::cerr << "[getThetas] Clusters thetas: " << thetas.size()
              << ". First column size (should be 1000): " << thetas[0].size() << "\n";

    std::cerr << "[getThetas] Clusters rs: " << rs.size()
              << ". First column size (should be 1000): " << rs[0].size() << "\n";

    return {thetas, rs};
}
/*
std::string runOllama(const std::string &prompt)
{
    std::string command =
        "powershell -Command \"$input = @'\n" +
        prompt +
        "\n'@; ollama run mistral-assumption-extractor\"";

    std::array<char, 4096> buffer;
    std::string result;

    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(command.c_str(), "r"), _pclose);
    if (!pipe)
        throw std::runtime_error("Failed to run ollama");

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }

    return result;
}
*/

static std::string makeTempFileName()
{
    static std::mt19937_64 rng{std::random_device{}()};
    return "ollama_prompt_" + std::to_string(rng()) + ".txt";
}

static std::string uniq(const char *prefix, const char *ext)
{
    static std::mt19937_64 rng{std::random_device{}()};
    return std::string(prefix) + std::to_string(rng()) + ext;
}

std::string runOllama(const std::string &prompt)
{
    const std::string OLLAMA_EXE =
        "C:\\Users\\H\\AppData\\Local\\Programs\\Ollama\\ollama.exe";

    const std::string promptFile = uniq("ollama_prompt_", ".txt");
    const std::string cmdFile = uniq("ollama_run_", ".cmd");

    // Write prompt
    {
        std::ofstream out(promptFile, std::ios::binary);
        if (!out)
            throw std::runtime_error("Failed to write prompt file");
        out << prompt << "\n";
    }

    // Write cmd script (this is the key)
    {
        std::ofstream out(cmdFile, std::ios::binary);
        if (!out)
            throw std::runtime_error("Failed to write cmd file");

        out << "@echo off\n";
        out << "\"" << OLLAMA_EXE << "\" run mistral-assumption-extractor < \"" << promptFile << "\" 2>&1\n";
    }

    // Run the script
    std::string command = "cmd /c \"" + cmdFile + "\"";

    std::array<char, 4096> buffer{};
    std::string result;

    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(command.c_str(), "r"), _pclose);
    if (!pipe)
        throw std::runtime_error("Failed to spawn cmd script");

    while (fgets(buffer.data(), (int)buffer.size(), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }

    std::remove(promptFile.c_str());
    std::remove(cmdFile.c_str());

    return result;
}

static std::string buildPrompt_ExtractAndMap(const std::string &strategyText)
{
    // Keep this stable. Your model expects TASK up top, then strict schema.
    return "TASK: EXTRACT_AND_MAP\n\n"
           "STRATEGY DESCRIPTION:\n"
           "\"\"\"\n" +
           strategyText + "\n\"\"\"\n\n"
                          "OUTPUT FORMAT (STRICT JSON):\n"
                          "{\n"
                          "  \"assumptions\": [],\n"
                          "  \"state_space_assumption_tokens\": [],\n"
                          "  \"unknowns\": []\n"
                          "}\n";
}

// substr(pos, len);
// substr(start, i - start + 1)

static std::string extractFirstJsonObject(const std::string &s)
{
    // This is a minimal brace-matching extractor.
    // It assumes the JSON object is not embedded inside code fences and that braces are balanced. (stripAnsi took care of this too partly)

    std::size_t start = s.find('{');
    if (start == std::string::npos)
        throw std::runtime_error("No '{' found in LLM output");

    int depth = 0; // This tracks nesting level.
    for (std::size_t i = start; i < s.size(); ++i)
    {
        if (s[i] == '{')
            depth++;
        else if (s[i] == '}')
            depth--;

        if (depth == 0)
        {
            return s.substr(start, i - start + 1);
        }
    }

    throw std::runtime_error("Unbalanced braces while extracting JSON"); //{ found, but it never closed
}

// raw output: \x1B[32m{ "assumptions": [] }\x1B[0m
// what my program sees: [ESC][32m{ "assumptions": [] }[ESC][0m
// output after stripAnsi: { "assumptions": [] }

static std::string stripAnsi(const std::string &s)
{
    std::string out;
    out.reserve(s.size());

    for (size_t i = 0; i < s.size(); ++i)
    {
        unsigned char c = (unsigned char)s[i];
        if (c == 0x1B)
        { // ESC (start ofan ANSI sequence)
            // Skip ESC[
            if (i + 1 < s.size() && s[i + 1] == '[')
            {
                i += 2;
                // Skip until a letter (final byte)
                while (i < s.size())
                {
                    unsigned char d = (unsigned char)s[i];
                    if ((d >= 'A' && d <= 'Z') || (d >= 'a' && d <= 'z'))
                        break;
                    i++;
                }
                // skip the final letter too
                continue;
            }
            // If it's ESC not followed by [, just drop it
            continue;
        }
        out.push_back((char)c);
    }

    return out;
}

double computeR2(const std::vector<double> &x, const std::vector<double> &y)
{
    if (x.size() != y.size() || x.empty())
    {
        throw std::invalid_argument("Vectors must have same non-zero size");
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

double percentile(std::vector<double> &v, double p)
{
    if (v.empty())
        return std::numeric_limits<double>::quiet_NaN();
    p = std::clamp(p, 0.0, 1.0);
    size_t k = (size_t)std::floor(p * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

int effective_rank_energy(std::vector<double> v)
{
    double alpha = 0.95;

    if (v.empty())
        return 0;

    // Compute total energy
    double total_energy = 0.0;
    for (double x : v)
        total_energy += x * x;

    if (total_energy == 0.0)
        return 0;

    // Sort by descending magnitude
    std::sort(v.begin(), v.end(),
              [](double a, double b)
              {
                  return std::abs(a) > std::abs(b);
              });

    double cumulative_energy = 0.0;
    int k = 0;

    for (double x : v)
    {
        cumulative_energy += x * x;
        ++k;

        if (cumulative_energy >= alpha * total_energy)
            break;
    }

    return k;
}

std::vector<std::vector<double>>
to_2d_vector(const std::vector<std::deque<double>> &input)
{
    std::vector<std::vector<double>> result;
    result.reserve(input.size());

    for (const auto &row : input)
    {
        result.emplace_back(row.begin(), row.end());
    }

    return result;
}

double operatorShockWedin(const Eigen::MatrixXd &Et, double gap)
{
    constexpr double EPS = 1e-12;

    if (!Et.allFinite())
        return std::numeric_limits<double>::infinity();
    if (!(gap > EPS))
        return std::numeric_limits<double>::infinity();

    // Et is symmetric if it's (Cov_t - Cov_{t-1}). Use eigen solver.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Et);
    if (es.info() != Eigen::Success)
        return std::numeric_limits<double>::infinity();

    // Spectral norm = max absolute eigenvalue
    const auto &evals = es.eigenvalues();
    double norm2 = 0.0;
    for (int i = 0; i < evals.size(); ++i)
        norm2 = std::max(norm2, std::abs(evals(i)));

    return norm2 / gap;
}

double effective_rank(const std::vector<double> &eigvals)
{
    double s1 = 0.0, s2 = 0.0;
    for (double e : eigvals)
    {
        if (e < 0.0)
            e = 0.0;
        s1 += e;
        s2 += e * e;
    }
    if (s2 <= 1e-12)
        return 0.0; // no variance -> totally "locked"
    return (s1 * s1) / s2;
}

double wedin_theoremKGT1(
    const Eigen::MatrixXd &M_overlap)
{

    Eigen::JacobiSVD<Eigen::MatrixXd> svdM(M_overlap);

    auto sigmaM = svdM.singularValues();
    double sigmaMIN = std::clamp(sigmaM.minCoeff(), 0.0, 1.0);
    double s = std::min(1.0, std::max(0.0, sigmaMIN));
    double sin_norm = std::sqrt(1.0 - s * s);

    LOG("Observed Subspace rotation (sin_norm): " << sin_norm);

    return sin_norm;
}

double wedin_theoremK1(const Eigen::VectorXd &u_cur_in, const Eigen::VectorXd &u_prev_in)
{
    // normalize defensively (SVD U should already be orthonormal, but don’t trust anything)
    Eigen::VectorXd u_prev = u_prev_in;
    Eigen::VectorXd u_cur = u_cur_in;

    u_prev.normalize();
    u_cur.normalize();

    double cos_theta = std::abs(u_prev.dot(u_cur));
    cos_theta = std::clamp(cos_theta, 0.0, 1.0);

    double sin_norm = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));

    return sin_norm;
}

double dominantSubspaceRotation(double sigmaMIN)
{
    return sqrt(1 - (sigmaMIN * sigmaMIN));
}

std::size_t argmin(
    const std::vector<double> &vec,
    double *min_value // pass nullptr if you don't want the value
)
{
    if (vec.empty())
    {
        throw std::invalid_argument("argmin: vector is empty");
    }

    std::size_t min_idx = 0;
    double min_val = vec[0];

    for (std::size_t i = 1; i < vec.size(); ++i)
    {
        if (vec[i] < min_val)
        {
            min_val = vec[i];
            min_idx = i;
        }
    }

    if (min_value)
    {
        *min_value = min_val;
    }

    return min_idx;
}

double argminV(const std::vector<double> &v)
{
    if (v.empty())
        return std::numeric_limits<double>::quiet_NaN();

    double minValue = v[0];

    for (size_t i = 1; i < v.size(); ++i)
    {
        if (v[i] < minValue)
            minValue = v[i];
    }

    return minValue;
}

double argmax(const std::vector<double> &v)
{
    if (v.empty())
    {
        throw std::invalid_argument("argmax: input vector is empty");
    }
    return *std::max_element(v.begin(), v.end());
}

void push(Buffer3 &buf, double x)
{
    buf.x2 = buf.x1;
    buf.x1 = buf.x0;
    buf.x0 = x;

    if (buf.count < 3)
        buf.count++;
}

bool computeVelocityFromBuffer(
    Buffer3 &buf,
    double dt,
    double &velocity)
{
    if (buf.count < 2)
        return false;

    if (buf.count == 2)
    {
        // first-order fallback
        velocity = (buf.x0 - buf.x1) / dt;
        return false;
    }

    // second-order backward difference
    velocity = (3.0 * buf.x0 - 4.0 * buf.x1 + buf.x2) / (2.0 * dt);
    return true;
}

Eigen::MatrixXd dequeToEigenMatrix(
    const std::deque<std::vector<double>> &data)
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
    dst.reserve(src.size()); // avoid reallocations

    for (const auto &v : src)
    {
        dst.push_back(v); // copies each vector
    }

    return dst;
}

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

    double r_eff = sum_sqrd / sum_sqrd_sv; // r_eff≈1: one sane mode, If 𝑟_eff ≈ 2: two sane modes and if r_eff ≫1: garbage / noise

    return r_eff;
}

double clamp_(double x)
{
    return std::max(0.0, std::min(1.0, x));
}

Eigen::MatrixXd computeVelocities(const std::vector<std::vector<double>>& data, double dt)
{
    if (data.empty()) throw std::invalid_argument("[computeVelocities] input empty.");
    if (dt <= 0.0)    throw std::invalid_argument("[computeVelocities] dt must be > 0.");

    const int N = static_cast<int>(data.size());
    const int d = static_cast<int>(data[0].size());
    if (d == 0) throw std::invalid_argument("[computeVelocities] zero dimension.");

    for (int i = 1; i < N; ++i) {
        if (static_cast<int>(data[i].size()) != d)
            throw std::invalid_argument("[computeVelocities] ragged input rows.");
    }
    if (N < 2) throw std::invalid_argument("[computeVelocities] need at least 2 samples.");

    // Pack into Eigen once
    Eigen::MatrixXd X(N, d);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < d; ++j)
            X(i, j) = data[i][j];

    Eigen::MatrixXd V(N, d);

    // forward
    V.row(0) = (X.row(1) - X.row(0)) / dt;

    // central
    for (int i = 1; i < N - 1; ++i)
        V.row(i) = (X.row(i + 1) - X.row(i - 1)) / (2.0 * dt);

    // backward
    V.row(N - 1) = (X.row(N - 1) - X.row(N - 2)) / dt;

    return V;
}

double perc_(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    p = std::min(1.0, std::max(0.0, p));
    size_t k = (size_t)std::floor(p * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

// returns one SRS value per window
std::vector<double> computeSwirlSeverityWindowed(
    const Eigen::MatrixXd& X, // N x d (already z-scored / EMA’d)
    double dt,
    int window,   // e.g. 200
    int stride,   // e.g. 20
    double speed_floor_percentile // ignore slowest 10% speeds
) {
    const double eps = 1e-12;
    const int N = (int)X.rows();
    const int d = (int)X.cols();
    if (N < window + 2) return {};

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
    for (int start = 0; start + window <= N; start += stride) {
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

        for (int i = start; i < end; ++i) {
            Eigen::VectorXd v = V.row(i);
            Eigen::VectorXd a = A.row(i);

            double s1 = v.squaredNorm();
            if (s1 <= std::max(v_floor2, 1e-10)) continue;

            double s2 = a.squaredNorm();
            double s3 = v.dot(a);

            double gram = s1 * s2 - s3 * s3; // ||v||^2||a||^2 - (v·a)^2 = ||v x a||^2
            if (gram < 0.0) gram = 0.0;

            double w = std::sqrt(gram) / (s1 + eps); // turning rate (1/time)
            omega.push_back(w);
        }

        if (omega.size() < 10) {   
            out.push_back(0.0);
            continue;
        }

        // robust summary: 90th percentile of omega (captures "severe swirl" without RMS spikes)
        double w_p90 = perc_(omega, 0.90);

        // optional normalization by typical speed to get curvature-per-distance (1/length)
        double v_med = perc_(speeds, 0.50);
        double srs = w_p90 / (v_med + eps);

        out.push_back(srs);
    }

    return out;
}

double derivativeOLS(const std::vector<double> &v, double dt)
{
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = v.size();

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

double effective_RankCPP(const std::vector<double> &sigma)
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

    double r_eff = sum_sqrd / sum_sqrd_sv; // r_eff≈1: one sane mode, If 𝑟_eff ≈ 2: two sane modes and if r_eff ≫1: garbage / noise

    return r_eff;
}

// Function to calculate standard deviation of a vector
double calculateStdDev(const std::vector<double> &vec, double mean)
{
    double sum = 0.0;
    for (double val : vec)
    {
        sum += (val - mean) * (val - mean);
    }
    return std::sqrt(sum / vec.size());
}

// Function to calculate Z-scores for a vector
std::vector<double> zscore(const std::vector<double> &vec)
{

    if (vec.empty())
    {
        std::invalid_argument("[zscore] vec empty.");
    }

    double mu = mean(vec);
    double stdDev = calculateStdDev(vec, mu);

    std::vector<double> zscores;
    for (double val : vec)
    {
        zscores.push_back((val - mu) / stdDev);
    }

    return zscores;
}

std::vector<double> normalizeVector(const std::vector<double> &data, double mag)
{
    if (data.empty())
    {
        throw std::invalid_argument("[normalizeVector] data empty.");
    }
    if (mag == 0.0)
    {
        throw std::invalid_argument("[normalizeVector] mag 0.");
    }
    std::vector<double> vec(data.size());
    for (int i = 0; i < data.size(); i++)
    {
        vec[i] = data[i] / mag;
    }
    if (vec.size() != data.size())
    {
        std::invalid_argument("[normalizeVector] vec != data size.");
    }
    return vec;
}

double dotProduct(const std::vector<double> &vec1, const std::vector<double> &vec2)
{
    if (vec1.size() != vec2.size())
    {
        throw std::invalid_argument("Vectors must be the same size");
    }
    double result = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i)
    {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// Function to calculate the magnitude of a vector
double magnitude(const std::vector<double> &vec)
{
    double sumOfSquares = 0.0;
    for (size_t i = 0; i < vec.size(); ++i)
    {
        sumOfSquares += vec[i] * vec[i];
    }
    return std::sqrt(sumOfSquares);
}

double median(const std::vector<double> &data)
{

    std::vector<double> vec = data;
    std::sort(vec.begin(), vec.end());

    if (data.size() % 2 == 0)
    {

        return 0.5 * (vec[vec.size() / 2 - 1] + vec[vec.size() / 2]);
    }
    else
    {

        return vec[vec.size() / 2];
    }
}

double meanPAIR(const std::vector<std::pair<double, int>> &data)
{

    double sum = 0.0;

    for (auto val : data)
    {
        sum = sum + val.first;
    }

    if (sum == 0.0)
    {
        return 0.0;
    }

    return sum / data.size();
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
        std::cerr << "[mean] sum is 0 btw";
    }

    return sum / data.size();
}

std::vector<std::vector<double>> transposeData(const std::vector<std::vector<double>> &matrix)
{

    if (matrix.empty() || matrix[0].empty())
    {
        throw std::invalid_argument("[transposeData] Matrix invalid");
    }

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::vector<std::vector<double>> result(cols, std::vector<double>(rows));

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}

std::vector<std::vector<double>> matrixToVector2D(const Eigen::MatrixXd &mat)
{
    std::vector<std::vector<double>> result(mat.rows(), std::vector<double>(mat.cols()));

    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            result[i][j] = mat(i, j);
        }
    }

    return result;
}

double avgClusterSize(const std::vector<std::vector<std::vector<double>>> &clusters)
{
    double sum = 0;
    for (auto cluster : clusters)
    {
        sum = sum + cluster.size();
    }
    return sum / clusters.size();
}

double distanceBlend(int iter, bool trigger, int startIter, int rampLength)
{

    if (!trigger)
        return 0.0; // fully Euclidean
    if (iter >= startIter + rampLength)
        return 1.0;                               // fully Mahalanobis after past this amount of iterations
    return double(iter - startIter) / rampLength; // returns the actual blend value
}

double adaptiveAlpha(int clusterSize)
{

    int minSize = 10;
    int maxSize = 500;
    double alphaMin = 0.05;
    double alphaMax = 0.3;
    // Clamp cluster size
    clusterSize = std::max(minSize, std::min(clusterSize, maxSize));
    // Linear mapping clusterSize -> alpha in [alphaMax, alphaMin]
    double alpha = alphaMax - (clusterSize - minSize) * (alphaMax - alphaMin) / (maxSize - minSize);
    return alpha;
}

bool allClustersHaveEnoughPoints(
    const std::vector<std::vector<std::vector<double>>> &clusters,
    size_t featureDim,
    size_t safetyFactor)
{
    if (clusters.empty())
        return false;

    for (const auto &cluster : clusters)
    {
        if (cluster.size() < safetyFactor * featureDim)
        {
            return false;
        }
    }
    return true;
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

Eigen::MatrixXd vectorToEigenMatrix(const std::vector<std::vector<double>> &data)
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
            throw std::invalid_argument("All rows in the 2D vector must have the same length.");
        }
    }

    Eigen::MatrixXd mat(rows, cols);
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            mat(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = data[i][j];
        }
    }

    return mat;
}

bool choleskyDecompLLT(const Eigen::MatrixXd &cov)
{

    if (cov.rows() != cov.cols() || cov.rows() == 0)
        return false;

    Eigen::MatrixXd sym = 0.5 * (cov + cov.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt(sym);
    return llt.info() == Eigen::Success; // better than numerical issue, checks more
}

bool choleskyDecompLDLT(const Eigen::MatrixXd &cov)
{

    if (cov.rows() != cov.cols() || cov.rows() == 0)
        return false;

    Eigen::MatrixXd sym = 0.5 * (cov + cov.transpose());
    Eigen::LDLT<Eigen::MatrixXd> llt(sym);
    return llt.info() == Eigen::Success;
}

Eigen::MatrixXd getInvCov(const Eigen::MatrixXd &data)
{

    Eigen::MatrixXd cov = data;

    cov = 0.5 * (cov + cov.transpose()); // exacty symmetric

    // Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> llt(cov);
    Eigen::MatrixXd cov_inv;

    // Check if the matrix is positive definite
    if (llt.info() == Eigen::NumericalIssue)
    {
        std::cerr << "[getInvCov] Matrix is not positive definite! Using Pseudo-Inverse!" << std::endl;
        cov += Eigen::MatrixXd::Identity(cov.rows(), cov.cols()) * 1e-3;
        cov_inv = cov.inverse(); // slower but safer fallback
    }
    else
    {
        // Compute the inverse using the Cholesky factor
        cov_inv = llt.solve(Eigen::MatrixXd::Identity(cov.rows(), cov.cols()));
    }

    return cov_inv;
}

std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> CovarianceMatrix(std::vector<std::vector<std::vector<double>>> &clusters, const std::vector<std::vector<double>> &centroids, const std::vector<std::vector<double>> &data)
{

    int d = centroids[0].size();
    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> results;

    // small clusters + high-dim = guaranteed numerical chaos.

    // Iterate over all clusters
    for (size_t z = 0; z < clusters.size(); ++z)
    {
        Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(d, d);

        // Iterate over each vector inside one cluster
        for (size_t i = 0; i < clusters[z].size(); ++i)
        {
            Eigen::VectorXd centered(d);

            // Center the vector by subtracting the centroid
            for (size_t j = 0; j < clusters[z][i].size(); ++j)
            {
                centered(j) = clusters[z][i][j] - centroids[z][j];
            }

            // Add the outer product of the centered vector to the covariance matrix
            cov += centered * centered.transpose();
        }

        int n = static_cast<int>(clusters[z].size());
        if (n <= 1)
        {
            results.push_back(Eigen::MatrixXd::Identity(d, d) * 1e-1);
            continue; // When a cluster has fewer than 2 points, (n – 1) blows up or I end up with garbage covariance.
        }

        Eigen::MatrixXd global_cov = computeRobustCovarianceMAD(data); // global covariance of data

        // Divide by (n-1) for unbiased covariance estimate
        cov /= (clusters[z].size() - 1);

        cov += Eigen::MatrixXd::Identity(d, d) * 1e-4; // Add small regularization. affect off diagonals

        // Eigen::Matrix<double, 14, 14> I = Eigen::Matrix<double, 14, 14>::Identity();
        double lambda = 5e-4; // or adaptive: lambda = 0.1 * cov.trace() / 14.0;

        // double lambda = 1e-4 + 1e-3 * (1.0 / sqrt(clusters[z].size())); //adaptive? smaller clusters get more pull

        // Eigen::Matrix<double, 14, 14> T = (cov.trace() / 14) * I;
        cov = (1 - lambda) * cov + lambda * global_cov;

        cov *= (d / cov.trace()); // normalize average variance ~ 1.0. global cov step // or do det based

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
        auto evals = es.eigenvalues();
        bool bad = (evals.minCoeff() <= 0);
        if (bad)
            cov += Eigen::MatrixXd::Identity(d, d) * 1e-2;
        // Numerical noise in high-dim data can produce slightly negative eigenvalues even when it “shouldn’t.”

        // Store the covariance matrix (7x7) for the cluster
        results.push_back(cov);
    }

    return results;
}

std::vector<double> eigenToStdVector(const Eigen::VectorXd &v)
{
    return std::vector<double>(v.data(), v.data() + v.size());
}

std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> CovarianceMatrixGPT(
    const std::vector<std::vector<std::vector<double>>> &clusters,
    const std::vector<std::vector<double>> &centroids,
    const std::vector<std::vector<double>> &data)
{
    int d = centroids[0].size();
    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> results;

    // Compute global covariance once for shrinkage
    Eigen::MatrixXd global_cov = computeRobustCovarianceMAD(data); // user-defined
    double global_trace = global_cov.trace();

    for (size_t z = 0; z < clusters.size(); ++z)
    {
        int n = static_cast<int>(clusters[z].size());

        // Small clusters -> fallback
        if (n <= 1)
        {
            results.push_back(Eigen::MatrixXd::Identity(d, d) * 1e-4);
            continue;
        }

        // Compute centered data matrix for cluster (d x n)
        Eigen::MatrixXd centered(d, n);
        for (int i = 0; i < n; ++i)
        {
            centered.col(i) = Eigen::Map<const Eigen::VectorXd>(clusters[z][i].data(), d) - Eigen::Map<const Eigen::VectorXd>(centroids[z].data(), d);
        }

        // Unbiased covariance
        Eigen::MatrixXd cov = (centered * centered.transpose()) / (n - 1);

        // Regularization: small diagonal scaled by average variance
        double diag_eps = std::max(1e-6, 1e-4 * cov.trace() / d); // so its not too weak
        cov += Eigen::MatrixXd::Identity(d, d) * diag_eps;

        // Adaptive shrinkage toward global covariance
        double lambda = 1e-4; //+ 1e-2 * (1.0 / std::sqrt(n)); // smaller clusters get more shrinkage
        cov = (1.0 - lambda) * cov + lambda * global_cov;

        // Normalize average variance ~1 for numerical stability
        // cov *= d / cov.trace(); might kill variance

        // Eigenvalue check for numerical issues
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
        Eigen::VectorXd evals = es.eigenvalues();
        double min_eval = evals.minCoeff();

        double eps_threshold = 1e-8;                                // lower threshhold
        double adjust_eps = std::max(1e-6, 1e-4 * cov.trace() / d); // adaptive guard

        if (min_eval <= eps_threshold)
        {
            cov += Eigen::MatrixXd::Identity(d, d) * adjust_eps;
        }

        results.push_back(cov);
    }

    return results;
}

double mahalanobisDistanceSingle(const Eigen::VectorXd &point, const Eigen::VectorXd &centroid, const Eigen::MatrixXd &covariance)
{

    // Eigen::Matrix<double, 14, 14> cov_reg = covariance;
    // cov_reg += 1e-3 * Eigen::Matrix<double, 14, 14>::Identity();   //pretty sure we dont need this since we already did shrinking in the covariancematrix function

    // Defensive copy to avoid modifying the original matrix
    Eigen::MatrixXd cov = covariance;

    cov = 0.5 * (cov + cov.transpose()); // exacty symmetric

    // Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> llt(covariance);
    Eigen::MatrixXd cov_inv;

    // Check if the matrix is positive definite
    if (llt.info() == Eigen::NumericalIssue)
    {
        std::cerr << "[MdistSingle]Matrix is not positive definite! Using Pseudo-Inverse!" << std::endl;
        cov += Eigen::MatrixXd::Identity(cov.rows(), cov.cols()) * 1e-3;
        cov_inv = cov.inverse(); // slower but safer fallback
    }
    else
    {
        // Compute the inverse using the Cholesky factor
        cov_inv = llt.solve(Eigen::MatrixXd::Identity(covariance.rows(), covariance.cols())); // redeclaration fucked me up
    }

    // Calculate the difference between the point and the mean
    Eigen::VectorXd diff = point - centroid;

    if (diff.size() != cov_inv.rows())
    {
        std::cerr << "[MdistSingle] Dimension mismatch in Mahalanobis: diff=" << diff.size()
                  << " cov=" << cov_inv.rows() << std::endl;
    }

    // Compute Mahalanobis distance
    double distance = std::sqrt(diff.transpose() * cov_inv * diff); // someone said to do this, idk why lol

    if (distance < 0)
        distance = 0.0;

    return distance;
}

std::vector<double> MahalanobisDistMulti(
    const std::vector<double> &point,
    const std::vector<std::vector<double>> &meanvecs,
    const std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> &covmatrices)
{

    // Convert input point into Eigen
    Eigen::VectorXd x = toEigenVector(point);

    std::vector<double> distances;
    distances.reserve(meanvecs.size());

    for (size_t c = 0; c < meanvecs.size(); ++c)
    {
        // Convert mean

        Eigen::VectorXd mu = toEigenVector(meanvecs[c]);

        // Regularize covariance to avoid singularities
        Eigen::MatrixXd cov_reg = covmatrices[c];

        // cov_reg += 1e-3 * Eigen::Matrix<double, 14, 14>::Identity();
        cov_reg = 0.5 * (cov_reg + cov_reg.transpose());

        // Inverse covariance
        Eigen::MatrixXd cov_inv;
        // Cholesky decomposition
        Eigen::LLT<Eigen::MatrixXd> llt(cov_reg);

        // Check if the matrix is positive definite
        if (llt.info() == Eigen::NumericalIssue)
        {
            std::cerr << "[MdistMulti] Matrix is not positive definite!" << std::endl;
        }
        else
        {
            // Compute the inverse using the Cholesky factor
            Eigen::MatrixXd cov_inv = llt.solve(Eigen::MatrixXd::Identity(cov_reg.rows(), cov_reg.cols()));
        }

        // Compute diff
        Eigen::VectorXd diff = x - mu;

        // Mahalanobis distance
        double dist_sq = (diff.transpose() * cov_inv * diff)(0, 0); // That ensures im not silently storing a matrix object with garbage or misaligned memory.
        distances.push_back(std::sqrt(dist_sq));
    }

    return distances;
}

double cosineSimilarity(const Eigen::VectorXd &A,
                        const Eigen::VectorXd &B)
{ // takes 2 vectors

    float epsilon = 1e-10f;

    // i think a is comparing it a point thats not there?

    double dot_product = A.dot(B); // A · B  //does A . B
    double norm_A = A.norm();      // ||A||
    double norm_B = B.norm();      // ||B||  //length vector from origin. "length" of a vector. // why is B 0

    if (norm_A < epsilon || norm_B < epsilon)
    {
        // if shits too small
        return 0.0;
    }

    // dividing by the norm allows you to focus on the direction of the vectors (the angle between them), rather than their magnitudes.

    return dot_product / (norm_A * norm_B);
}

double cosineSimilarityCPP(const std::vector<double> &vec1, const std::vector<double> &vec2)
{

    float epsilon = 1e-10f;

    double dot = dotProduct(vec1, vec2);
    double mag1 = magnitude(vec1);
    double mag2 = magnitude(vec2);

    if (mag1 < epsilon || mag2 < epsilon)
    {
        // if shits too small
        return 0.0;
    }

    return dot / (mag1 * mag2);
}

double mahalanobisDistanceSingleFast(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const Eigen::MatrixXd &invCov)
{

    Eigen::VectorXd diff = x - y;
    double distsquared = std::sqrt((diff.transpose() * invCov * diff).value());
    double dist = std::sqrt(std::max(distsquared, 0.0));
    if (dist < 0)
        dist = 0;

    if (diff.size() != invCov.rows())
    {
        std::cerr << "[MdistSingleFast] Dimension mismatch: diff=" << diff.size()
                  << " cov=" << invCov.rows() << std::endl;
    }

    return dist;
}

Eigen::MatrixXd computeGlobalCovariance(const std::vector<std::vector<double>> &data)
{
    size_t n = data.size();
    size_t d = data[0].size();
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
    for (auto &x : data)
        for (size_t j = 0; j < d; ++j)
            mean(j) += x[j];
    mean /= n;

    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(d, d);
    for (auto &x : data)
    {
        Eigen::VectorXd diff(d);
        for (size_t j = 0; j < d; ++j)
            diff(j) = x[j] - mean(j);
        cov += diff * diff.transpose();
    }

    cov /= (n - 1);
    return cov;
}

Eigen::MatrixXd computeRobustCovarianceMAD(const std::vector<std::vector<double>> &data)
{
    size_t n = data.size();
    if (n < 2)
    {
        throw std::invalid_argument("[ComputeRCovMAD] At least two data points are required to compute covariance.");
    }

    size_t d = data[0].size();
    for (const auto &x : data)
    {
        if (x.size() != d)
        {
            throw std::invalid_argument("[ComputeRCovMAD] Inconsistent row size detected.");
        }
    }

    // Step 1: Compute the mean of each dimension
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
    for (const auto &x : data)
    {
        for (size_t j = 0; j < d; ++j)
        {
            mean(j) += x[j];
        }
    }
    mean /= n;

    // Step 2: Compute the MAD for each dimension
    Eigen::MatrixXd madMatrix = Eigen::MatrixXd::Zero(d, d);

    // For each dimension, compute the MAD
    for (size_t j = 0; j < d; ++j)
    {
        std::vector<double> columnValues;
        for (const auto &x : data)
        {
            columnValues.push_back(x[j]);
        }

        double mad = computeMAD(columnValues);

        // The "robust covariance" is simply MAD applied to each dimension for now
        madMatrix(j, j) = mad * mad; // Using MAD^2 as a measure of spread for variance
    }

    return madMatrix;
}

double computeMAD(const std::vector<double> &values)
{
    std::vector<double> absDeviations(values.size());
    double median = 0;

    // Compute the median of the values
    std::vector<double> sortedValues = values;
    std::sort(sortedValues.begin(), sortedValues.end());

    size_t n = sortedValues.size();
    if (n % 2 == 0)
    {
        median = (sortedValues[n / 2 - 1] + sortedValues[n / 2]) / 2.0;
    }
    else
    {
        median = sortedValues[n / 2];
    }

    // Calculate the absolute deviations from the median
    for (size_t i = 0; i < values.size(); ++i)
    {
        absDeviations[i] = std::abs(values[i] - median);
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

std::vector<std::vector<double>> PCA(const Eigen::MatrixXd &data, double varianceThreshold)
{

    if (data.rows() == 0 || data.cols() == 0)
    {
        throw std::runtime_error("Input data matrix is empty!");
    }

    if (data.rows() == 1)
    {
        throw std::runtime_error("PCA requires at least 2 rows of data.");
    }

    varianceThreshold = std::clamp(varianceThreshold, 0.0, 1.0);
    // Step 1: Mean-center the data
    const Eigen::MatrixXd centered = data.rowwise() - data.colwise().mean();

    // Step 2: Compute covariance matrix
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(data.rows() - 1);

    // Step 3: Eigen decomposition (Eigen gives ascending order)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);

    if (eig.info() != Eigen::Success)
    {
        throw std::runtime_error("Eigen decomposition failed.");
    }

    Eigen::VectorXd eigenValues = eig.eigenvalues();
    Eigen::MatrixXd eigenVectors = eig.eigenvectors();

    // Step 4: Reverse to get descending order (largest eigenvalue first)
    Eigen::VectorXd eigenValuesDesc = eigenValues.reverse().eval();

    Eigen::MatrixXd eigenVectorsDesc(eigenVectors.rows(), eigenVectors.cols());
    for (int i = 0; i < eigenVectors.cols(); ++i)
    {
        eigenVectorsDesc.col(i) = eigenVectors.col(eigenVectors.cols() - 1 - i);
    }

    // Step 5: Determine number of components to keep based on variance threshold
    double totalVariance = eigenValuesDesc.sum();
    double runningSum = 0.0;
    int k = eigenValuesDesc.size();
    for (int i = 0; i < eigenValuesDesc.size(); ++i)
    {
        runningSum += eigenValuesDesc(i);
        if (runningSum / totalVariance >= varianceThreshold)
        {
            k = i + 1;
            break;
        }
    }

    std::cerr << "Keeping " << k << " principal components.\n"; // DID CUSTOM

    // Step 6: Project data onto top-k components
    Eigen::MatrixXd reduced = centered * eigenVectorsDesc.leftCols(k);

    // Step 7: Convert Eigen matrix → std::vector<std::vector<double>>
    std::vector<std::vector<double>> result(reduced.rows(), std::vector<double>(reduced.cols()));
    for (int i = 0; i < reduced.rows(); ++i)
        for (int j = 0; j < reduced.cols(); ++j)
            result[i][j] = reduced(i, j);

    return result;
}

std::vector<std::string> tostringvector(const std::string &input)
{
    std::vector<std::string> result;
    std::istringstream iss(input);
    std::string token;

    while (iss >> token)
    { // splits on whitespace
        result.push_back(token);
    }

    return result;
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> PCA_Eigenvectors(const Eigen::MatrixXd &data, double varianceThreshold)
{

    if (data.rows() == 0 || data.cols() == 0)
    {
        throw std::runtime_error("Input data matrix is empty!");
    }

    if (data.rows() == 1)
    {
        throw std::runtime_error("PCA requires at least 2 rows of data.");
    }

    varianceThreshold = std::clamp(varianceThreshold, 0.0, 1.0);
    // Step 1: Mean-center the data
    const Eigen::MatrixXd centered = data.rowwise() - data.colwise().mean();

    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinV);
    // Eigen::MatrixXd V = svd.matrixV();
    // Eigen::VectorXd S = svd.singularValues(); //strength (energy)
    // S = (S.array().square() / double(data.rows() - 1)).matrix(); //optional eigenvalues now

    // Step 2: Compute covariance matrix
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(data.rows() - 1);

    double lambda = 1e-3 * cov.trace() / cov.rows(); // scale-aware
    cov += lambda * Eigen::MatrixXd::Identity(cov.rows(), cov.cols());

    // Step 3: Eigen decomposition (Eigen gives ascending order)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);

    if (eig.info() != Eigen::Success)
    {
        throw std::runtime_error("Eigen decomposition failed.");
    }

    Eigen::VectorXd eigenValues = eig.eigenvalues();
    Eigen::MatrixXd eigenVectors = eig.eigenvectors();

    // Step 4: Reverse to get descending order (largest eigenvalue first)
    Eigen::VectorXd eigenValuesDesc = eigenValues.reverse().eval(); // was eigenValues

    Eigen::MatrixXd eigenVectorsDesc(eigenVectors.rows(), eigenVectors.cols()); // was S
    for (int i = 0; i < eigenVectors.cols(); ++i)
    {
        eigenVectorsDesc.col(i) = eigenVectors.col(eigenVectors.cols() - 1 - i);
    }

    // Step 5: Determine number of components to keep based on variance threshold
    double totalVariance = eigenValuesDesc.sum();
    double runningSum = 0.0;
    int k = eigenValuesDesc.size();
    for (int i = 0; i < eigenValuesDesc.size(); ++i)
    {
        runningSum += eigenValuesDesc(i);
        if (runningSum / totalVariance >= varianceThreshold)
        {
            k = i + 1;
            break;
        }
    }

    std::cerr << "Keeping " << k << " principal components.\n"; // CHANGE TO K FOR NORMAL

    Eigen::MatrixXd eigenVectorsDescK = eigenVectorsDesc.leftCols(k);
    Eigen::VectorXd eigenValuesDescK = eigenValuesDesc.head(k);

    return {eigenVectorsDescK, eigenValuesDescK};
}

double sampleVariance(const std::vector<double> &data)
{
    if (data.size() <= 1)
    {
        throw std::invalid_argument("Sample size must be greater than 1 for variance calculation.");
    }

    double mean = 0.0;
    double sum_of_squares = 0.0;

    // Compute the mean
    for (const auto &val : data)
    {
        mean += val;
    }
    mean /= data.size();

    // Calculate the sum of squared deviations from the mean
    for (const auto &val : data)
    {
        sum_of_squares += (val - mean) * (val - mean);
    }

    // Return sample variance (divide by n-1)
    return sum_of_squares / (data.size() - 1);
}

size_t argmax_(const std::vector<double> &v)
{
    if (v.empty())
    {
        throw std::invalid_argument("argmax: empty vector");
    }

    size_t max_index = 0;
    double max_value = v[0];

    for (size_t i = 1; i < v.size(); ++i)
    {
        if (v[i] > max_value)
        {
            max_value = v[i];
            max_index = i;
        }
    }

    return max_index;
}

double euclideanDist(const std::vector<double> &a, const std::vector<double> &b)
{

    // takes in vector
    if (a.size() != b.size())
    {
        std::cerr << "[euclideanDist] Dimension mismatch: a.size() = " << a.size()
                  << ", b.size() = " << b.size() << std::endl;
        exit(1); // Or th3row exception
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }

    // sums up the squared differences of each single dimension
    // measures how far theyre apart in high dimensional space
    return sqrt(sum);
}

std::vector<std::pair<double, int>> getAdjRow(const std::vector<double> &ForeignVector, const std::vector<std::vector<double>> &normaldata, int num_neighbours)
{
    // std::cout << "Entered: [getAdjRow]\n";
    std::vector<std::pair<double, int>> weights_neighbours;

    for (size_t i = 0; i < normaldata.size(); i++)
    {
        double dist = euclideanDist(ForeignVector, normaldata[i]); // since its local, euclidean is fine
        weights_neighbours.emplace_back(dist, i);
    }

    std::sort(weights_neighbours.begin(), weights_neighbours.end(),
              [](const auto &a, const auto &b)
              { return a.first < b.first; });

    // number of neighbors you want
    if (weights_neighbours.size() > num_neighbours)
    {
        weights_neighbours.resize(num_neighbours);
    }
    // std::cout << "Exited: [getAdjRow]\n";
    return weights_neighbours;
}

int insertForeignNode(const std::vector<std::pair<double, int>> &neighbours, std::vector<std::vector<Edge>> &graph)
{
    // F = new node index
    int F = graph.size();
    graph.push_back(std::vector<Edge>()); // creates a fresh new adjacency list for the new vector (node). this is at index F (next new slot)

    // add edges both ways                               //dist idx
    for (auto &[dist, idx] : neighbours)
    {                                    // iterate over: [6.6, 23][7.7, 2][weight, index(neighbour)]
        graph[F].push_back({idx, dist}); // edge weight = dist
        graph[idx].push_back({F, dist}); // f is the new id of the node  symmetric edge, same weight
    }

    return F;
}

void removeForeignNode(int F, const std::vector<std::pair<double, int>> &neighbors, std::vector<std::vector<Edge>> &graph)
{
    // remove edges pointing to F from neighbors
    for (auto &[dist, idx] : neighbors)
    {
        auto &vec = graph[idx];
        vec.erase(
            remove_if(vec.begin(), vec.end(),
                      [&](const Edge &e)
                      { return e.to == F; }),
            vec.end());
    }

    // remove the foreign node's adjacency list
    graph.pop_back();
}

int argmin(const std::vector<std::pair<double, int>> &vec)
{
    // Find the element with the minimum `.first` value (the double)
    auto it = std::min_element(vec.begin(), vec.end(),
                               [](const std::pair<double, int> &a, const std::pair<double, int> &b)
                               {
                                   return a.first < b.first; // Compare based on `.first`
                               });
    return std::distance(vec.begin(), it); // Return the index of the minimum element
}

/*

std::vector<int> Medoids(const std::vector<std::vector<double>> &data,
                         const std::vector<std::vector<int>> &clusterIndices,
                         const Eigen::MatrixXd &Psi,
                         bool geodesic)
{
    if (data.size() != Psi.rows())
    {
        throw std::invalid_argument("[Medoid] ClusterIndices and Geodesics don't align!");
    }

    std::cerr << "Data size: " << data.size() << ", Geodesics size: " << Psi.rows() << std::endl;

    std::vector<int> medoids;

    for (int i = 0; i < clusterIndices.size(); i++)
    { // loop over all clusters

        if (clusterIndices[i].empty())
        {
            throw std::invalid_argument("[Medoid] Cluster " + std::to_string(i) + " is empty!");
        }

        std::cerr << "[Medoid] Processing cluster " << i << " with " << clusterIndices[i].size() << " points." << std::endl;

        std::vector<std::pair<double, int>> distsums;

        for (int j = 0; j < clusterIndices[i].size(); j++)
        { // loop over the point ids in a cluster [i]
            double distsum = 0.0;
            int seed = clusterIndices[i][j];

            // Check if seed is a valid index for data and geodesics
            if (seed >= data.size())
            {
                throw std::out_of_range("[Medoid] Invalid index in clusterIndices for data: " + std::to_string(seed));
            }
            if (seed >= Psi.rows())
            {
                throw std::out_of_range("[Medoid] Invalid index in clusterIndices for geodesics: " + std::to_string(seed));
            }

            for (const auto &pid : clusterIndices[i])
            {
                if (pid >= data.size())
                {
                    throw std::out_of_range("[Medoid] Invalid index in clusterIndices for data: " + std::to_string(pid));
                }
                if (pid >= Psi.rows())
                {
                    throw std::out_of_range("[Medoid] Invalid index in clusterIndices for geodesics: " + std::to_string(pid));
                }

                if (seed != pid)
                {
                    if (geodesic)
                    {
                        double geodesicDist = diffusion_distance(seed, pid, Psi);
                        distsum += geodesicDist;
                    }
                    else
                    {
                        double eucDist = euclideanDist(data[seed], data[pid]);
                        distsum += eucDist;
                    }
                }
            }

            distsums.emplace_back(distsum, j); // oh wait the seed can be out of bounds, so i need to push back j. thats the location in the cluster
        }

        // Debug: Print the size of distsums and minval_index
        std::cerr << "[Medoid] distsums size: " << distsums.size() << std::endl;

        // Ensure argmin is within bounds

        // int minval_index = argmin(distsums); !!
        int minval_index = kthSmallestIndexPairs(distsums, 0); // 0 is smallest

        // Debug: Check the value of minval_index before accessing distsums
        std::cerr << "[Medoid] argmin index: " << minval_index << ", distsums size: " << distsums.size() << std::endl;

        if (minval_index >= distsums.size())
        {
            throw std::out_of_range("[Medoid] argmin index out of bounds: " + std::to_string(minval_index));
        }

        int local_index = distsums[minval_index].second; // .second is the index in the cluster
        int medoidID = clusterIndices[i][local_index];   // NOW this is a valid global index

        // Debug: Print the value of medoidID and the size of clusterIndices[i]
        std::cerr << "[Medoid] medoidID: " << medoidID << ", clusterIndices[" << i << "] size: " << clusterIndices[i].size() << std::endl;

        // This is the line where the crash happens:
        medoids.push_back(medoidID); // something is going wrong here
    }

    return medoids;
    // Mentor: You confused the cluster-local index with the global data index!! thats what i thought actually
    // Mentor: You were feeding global indices into a place that expects cluster-local indices. Classic medoid-implementation faceplant.
}

*/

std::vector<std::vector<std::pair<double, int>>>
alignedDistanceList(
    const std::vector<std::vector<double>> &distanceMatrix,
    const std::vector<std::vector<int>> &knnList)
{
    std::cerr << "knnlist size: " << knnList.size() << std::endl;
    if (distanceMatrix.size() != knnList.size())
        throw std::invalid_argument("distanceMatrix and knnList size mismatch.");

    int n = distanceMatrix.size();
    std::vector<std::vector<std::pair<double, int>>> list;
    list.reserve(n);

    for (int i = 0; i < n; i++)
    {
        std::vector<std::pair<double, int>> alignedRow;
        alignedRow.reserve(knnList[i].size());

        for (int j = 0; j < knnList[i].size(); j++)
        {
            int idx = knnList[i][j];
            if (idx < 0 || idx >= n)
                throw std::out_of_range("Invalid neighbor index.");

            double dist = distanceMatrix[i][idx];
            alignedRow.emplace_back(dist, idx);
        }

        list.push_back(std::move(alignedRow));
    }

    return list;
}

std::vector<std::vector<double>> distanceMatrix(const std::vector<std::vector<double>> &data)
{
    std::cerr << "data size: " << data.size() << std::endl;
    // std::cout << "Entered: [distanceMatrix]\n";
    //  why does this need to be euclidian, because theyre local. and thats fine. we need local data to connect together to get global distances (geodesic)
    std::vector<std::vector<double>> matrix;
    matrix.reserve(data.size());

    for (const auto &vec : data)
    {
        std::vector<double> localbuffer;
        for (const auto &othervec : data)
        {
            double dist = euclideanDist(vec, othervec);
            localbuffer.push_back(dist);
        }
        matrix.push_back(localbuffer);
    }
    // std::cout << "Exited: [distanceMatrix]\n";
    return matrix;
}

std::vector<std::vector<Edge>> adjacencyList(const std::vector<std::vector<std::pair<double, int>>> &alignedDistList)
{

    std::cerr << "alignedDistList size: " << alignedDistList.size() << std::endl;

    int N = alignedDistList.size();
    std::vector<std::vector<Edge>> graph(N);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < alignedDistList[i].size(); j++)
        {

            double w = alignedDistList[i][j].first;
            int idx = alignedDistList[i][j].second;

            if (idx < 0 || idx >= N)
            {
                throw std::runtime_error("Invalid neighbor index.");
            }

            graph[i].push_back({idx, w});
        }
    }

    return graph;
}

double dijkstra_AB(const std::vector<std::vector<Edge>> &graph, int A, int B)
{
    // std::cout << "Entered: [DijkstraAB]\n";
    int N = graph.size();
    const double INF = std::numeric_limits<double>::infinity();

    std::vector<double> dist(N, INF);
    std::vector<bool> visited(N, false);

    // This line defines a min-heap priority queue of (double, int) pairs, where the queue always keeps the pair with the smallest double at the top.
    std::priority_queue<
        std::pair<double, int>,
        std::vector<std::pair<double, int>>,
        std::greater<std::pair<double, int>>>
        pq;

    dist[A] = 0.0;
    pq.push({0.0, A});

    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();

        if (visited[u])
            continue;
        visited[u] = true;

        // Early exit when we reach B
        if (u == B)
            return d;

        for (auto &e : graph[u])
        {
            int v = e.to;
            double w = e.weight;

            if (d + w < dist[v])
            {
                dist[v] = d + w;
                pq.push({dist[v], v});
            }
        }
    }
    // std::cout << "Exited: [dijkstraAB]\n";
    return INF; // unreachable
}

std::vector<double> dijkstra_from_source(const std::vector<std::vector<Edge>> &graph, int src)
{
    // std::cout << "Entered: [DijkstraSRC]\n";
    int n = graph.size();
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> dist(n, INF); // default: distance to each node is infinite
    dist[src] = 0.0;

    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>> pq;
    pq.push({0.0, src});

    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u])
            continue;

        for (const auto &e : graph[u])
        {
            int v = e.to;
            double w = e.weight;
            if (dist[u] + w < dist[v])
            {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    // std::cout << "Exited: [DijkstraSRC]\n";
    return dist;
}

std::vector<std::vector<double>> precomputeGeodesicDistances(const std::vector<std::vector<Edge>> &graph)
{
    // std::cout << "Entered: [precomputeGeo]\n";
    int n = graph.size();
    std::vector<std::vector<double>> geo(n);

    for (int i = 0; i < n; i++)
    {
        geo[i] = dijkstra_from_source(graph, i); // fil it up with vector distances
    }
    // std::cout << "Exited: [precomputeGeo]\n";
    return geo; // so the first row is the first vector. and its SP to every other vector// second row second vector
}

void generateInvertedMobiusStrip(
    std::vector<std::vector<double>> &data,
    int numRows,
    int numDims)
{

    double width = 0.4;
    double R = 1.0;
    double tiltAmplitude = 0.35; // controls how much the strip tilts in/out of plane
    double zWaveAmp = 0.15;      // gentle centerline Z drifting

    if (numRows < 1)
        return;

    data.assign(numRows, std::vector<double>(numDims, 0.0));
    std::srand((unsigned)std::time(nullptr));

    const double twistSign = -1.0; // inverted twist

    for (int i = 0; i < numRows; ++i)
    {

        double u = 2.0 * 3.1415 * (double)i / (double)numRows;

        // width coordinate
        double v = width * (((std::rand() % 10001) / 10000.0) - 0.5);

        // centerline dir vectors
        double rx = std::cos(u), ry = std::sin(u), rz = 0.0;
        double tx = -std::sin(u), ty = std::cos(u), tz = 0.0;

        // Möbius twist rotation
        double halfTheta = twistSign * 0.5 * u;
        double c = std::cos(halfTheta);
        double s = std::sin(halfTheta);

        // XY width direction before tilt
        double wx = c * rx + s * tx;
        double wy = c * ry + s * ty;

        // vertical component added for beauty and anti-pancaking
        double wz = tiltAmplitude * std::sin(3.0 * u);

        // normalize full width direction
        double mag = std::sqrt(wx * wx + wy * wy + wz * wz);
        wx /= mag;
        wy /= mag;
        wz /= mag;

        // center circle base position
        double cx = R * rx;
        double cy = R * ry;
        double cz = zWaveAmp * std::sin(2.0 * u); // smoother vertical sweep

        // final point
        double x = cx + v * wx;
        double y = cy + v * wy;
        double z = cz + v * wz;

        data[i][0] = x;
        data[i][1] = y;
        data[i][2] = z;

        // high-dimensional noise
        for (int j = 3; j < numDims; ++j)
            data[i][j] = ((std::rand() % 10001) / 10000.0) - 0.5;
    }
}

//------------------------------------LEARN

std::vector<std::vector<int>> connectedComponents(const std::vector<std::vector<Edge>> &adj)
{
    int N = adj.size();
    std::vector<int> visited(N, 0);
    std::vector<std::vector<int>> comps;

    std::vector<int> stack;
    stack.reserve(N);

    for (int start = 0; start < N; start++)
    {
        if (visited[start])
            continue;

        // New component
        comps.emplace_back();
        auto &comp = comps.back();

        // DFS
        stack.push_back(start);
        visited[start] = 1;

        while (!stack.empty())
        {
            int u = stack.back();
            stack.pop_back();
            comp.push_back(u);

            for (auto &e : adj[u])
            {
                int v = e.to;
                if (!visited[v])
                {
                    visited[v] = 1;
                    stack.push_back(v);
                }
            }
        }
    }

    return comps;
}

BridgeEdge findClosestBridge( // returns an object
    const std::vector<int> &A,
    const std::vector<int> &B,
    const std::vector<std::vector<double>> &distanceMatrix)
{
    BridgeEdge best{-1, -1, std::numeric_limits<double>::infinity()};

    for (int i : A)
    {
        for (int j : B)
        {
            double d = distanceMatrix[i][j];
            if (d < best.dist)
            {
                best = {i, j, d};
            }
        }
    }

    return best;
}

void connectComponents(
    std::vector<std::vector<Edge>> &adj,
    const std::vector<std::vector<double>> &distanceMatrix)
{
    auto comps = connectedComponents(adj);
    int C = comps.size();

    if (C <= 1)
        return; // already fully connected

    std::cerr << "Before connecting, number of components: " << comps.size() << std::endl;

    // Connect components linearly in order of closest possible edges
    for (int i = 0; i < C - 1; i++)
    {
        BridgeEdge b = findClosestBridge(comps[i], comps[i + 1], distanceMatrix);

        if (b.u == -1)
            throw std::runtime_error("Failed to find bridging edge? Something is wrong.");

        // Add bidirectional edge
        adj[b.u].push_back({b.v, b.dist});
        adj[b.v].push_back({b.u, b.dist});
    }

    auto comps_after = connectedComponents(adj);
    if (comps_after.size() > 1)
    {
        std::cerr << "Warning: Graph still has disconnected components!" << std::endl;
    }
    else
    {
        std::cerr << "Graph is fully connected!" << std::endl;
    }
}

int kthSmallestIndex(const std::vector<int> &v, int k)
{
    if (k < 0 || k >= static_cast<int>(v.size()))
        throw std::out_of_range("k out of range");

    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                     [&](int a, int b)
                     { return v[a] < v[b]; });

    // but if.. ah wait. kth smallest != index. the indexes only represent locations, and location (NODE ID) has nothing to do if its "smaller".
    // i just have to switch to, .first, right

    return idx[k];
}

int kthSmallestIndexDouble(const std::vector<double> &v, int k)
{
    if (k < 0 || k >= static_cast<int>(v.size()))
        throw std::out_of_range("k out of range");

    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                     [&](int a, int b)
                     { return v[a] < v[b]; });

    // but if.. ah wait. kth smallest != index. the indexes only represent locations, and location (NODE ID) has nothing to do if its "smaller".
    // i just have to switch to, .first, right

    return idx[k];
}

int kthSmallestIndexPairs(const std::vector<std::pair<double, int>> &v, int k)
{
    if (k < 0 || k >= static_cast<int>(v.size()))
        throw std::out_of_range("k out of range");

    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::nth_element(
        idx.begin(),
        idx.begin() + k,
        idx.end(),
        [&](int a, int b)
        {
            const double va = v[a].first;
            const double vb = v[b].first;

            // Handle NaNs: push them to the end
            if (std::isnan(va))
                std::cerr << "NAN detected [kthSmallestIndexPairs]";
            return false;
            if (std::isnan(vb))
                std::cerr << "NAN detected [kthSmallestIndexPairs]";
            return true;

            // Compare by .first, stable tie-break by index
            return va < vb || (va == vb && a < b);
        });

    return idx[k]; // index of the k-th smallest by .first
}

// Normalize columns of matrix X
Eigen::MatrixXd normalizeColumns(Eigen::MatrixXd &Xin, double epsilon)
{

    std::vector<std::vector<double>> X = matrixToVector2D(Xin);

    int rows = X.size();
    if (rows == 0)
        return Xin;
    int cols = X[0].size();

    for (int j = 0; j < cols; j++)
    {
        // Extract column j
        std::vector<double> col(rows);
        for (int i = 0; i < rows; i++)
        {
            col[i] = X[i][j];
        }

        // Compute variance of column j
        double var = variance(col);

        // Compute denominator sqrt(var + epsilon)
        double denom = sqrt(var + epsilon);

        // Normalize column j
        for (int i = 0; i < rows; i++)
        {
            X[i][j] = X[i][j] / denom;
        }
    }

    return vectorToEigenMatrix(X);
}

double variance(const std::vector<double> &v)
{
    double mean = 0.0;
    for (double x : v)
    {
        mean += x;
    }
    mean /= v.size();

    double var = 0.0;
    for (double x : v)
    {
        var += (x - mean) * (x - mean);
    }
    var /= v.size();

    return var;
}

std::vector<std::vector<double>> shuffle_columns(const std::vector<std::vector<double>> &dataset)
{

    int num_columns = dataset[0].size();
    int num_rows = dataset.size();

    // Create a vector of column indices
    std::vector<int> column_indices(num_columns);
    for (int i = 0; i < num_columns; ++i)
    {
        column_indices[i] = i;
    }

    // Randomly shuffle the column indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(column_indices.begin(), column_indices.end(), g);

    // Create a new dataset with shuffled columns
    std::vector<std::vector<double>> shuffled_dataset(num_rows, std::vector<double>(num_columns));

    // Reassign columns in the shuffled order
    for (int i = 0; i < num_rows; ++i)
    {
        for (int j = 0; j < num_columns; ++j)
        {
            shuffled_dataset[i][j] = dataset[i][column_indices[j]];
        }
    }

    // Copy shuffled dataset back into the original dataset
    return shuffled_dataset;
}

void signFlipRandomCols(std::vector<std::vector<double>> &dataset)
{
    // Copy the dataset to avoid modifying the original
    // Get the number of rows and columns
    int numRows = dataset.size();
    int numCols = dataset[0].size();

    // Seed the random number generator
    std::srand(std::time(0));

    // Loop through all columns
    for (int col = 0; col < numCols; ++col)
    {
        // Randomly decide if we should flip this column's sign
        if (std::rand() % 2 == 0)
        { // 50% chance to flip the sign
            // Flip the sign of all elements in this column
            for (int row = 0; row < numRows; ++row)
            {
                dataset[row][col] *= -1;
            }
        }
    }
}

bool isNoisySignal(
    const Eigen::MatrixXd &X,
    double threshold // >5 is usually noise-dominated
)
{
    const int N = X.rows();
    const int d = X.cols();
    if (N < 4)
        return false;

    double signal_var = 0.0;
    double diff_var = 0.0;

    for (int j = 0; j < d; ++j)
    {
        Eigen::VectorXd x = X.col(j);
        Eigen::VectorXd dx = x.tail(N - 1) - x.head(N - 1);

        signal_var += x.array().square().mean();
        diff_var += dx.array().square().mean();
    }

    signal_var /= d;
    diff_var /= d;

    if (signal_var < 1e-12)
        return false;

    double ratio = diff_var / signal_var;
    return ratio > threshold;
}

Eigen::MatrixXd savitzkyGolaySmooth(
    const Eigen::MatrixXd &X)
{
    // window = 5, poly order = 2
    static const double c[5] = {-3, 12, 17, 12, -3};
    static const double norm = 35.0;

    const int N = X.rows();
    const int d = X.cols();
    if (N < 5)
        return X;

    Eigen::MatrixXd Y = X;

    for (int i = 2; i < N - 2; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            Y(i, j) =
                (c[0] * X(i - 2, j) +
                 c[1] * X(i - 1, j) +
                 c[2] * X(i, j) +
                 c[3] * X(i + 1, j) +
                 c[4] * X(i + 2, j)) /
                norm;
        }
    }

    // copy edges
    Y.row(0) = X.row(0);
    Y.row(1) = X.row(1);
    Y.row(N - 2) = X.row(N - 2);
    Y.row(N - 1) = X.row(N - 1);

    return Y;
}

//DUMB BASELINES
double eigenvalueDominance(const Eigen::MatrixXd& returns)
{
    int W = returns.rows();

    Eigen::MatrixXd centered = returns.rowwise() - returns.colwise().mean();
    Eigen::MatrixXd cov = (centered.transpose() * centered) / (W - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);

    double lambda_max = solver.eigenvalues().maxCoeff();
    double trace = cov.trace();

    return lambda_max / trace;
}

Eigen::VectorXd meanAbsoluteReturn(const Eigen::MatrixXd& window)
{
    int W = window.rows();

    Eigen::VectorXd mar = window.cwiseAbs().colwise().sum() / W;

    return mar;
}

// Compute covariance between two vectors
double covariancee(const std::vector<double>& a, const std::vector<double>& b) {
    double ma = mean(a);
    double mb = mean(b);

    double cov = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        cov += (a[i] - ma) * (b[i] - mb);
    }

    return cov / (a.size() - 1);
}

// Extract column from matrix
std::vector<double> getColumns(const std::vector<std::vector<double>>& data, int col) {
    std::vector<double> result(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result[i] = data[i][col];
    }
    return result;
}

// Compute covariance matrix for the entire matrix
std::vector<std::vector<double>> covarianceMatrixx(const std::vector<std::vector<double>>& data) {
    int N = data[0].size();  // number of columns
    std::vector<std::vector<double>> cov(N, std::vector<double>(N));

    for (int i = 0; i < N; i++) {
        auto xi = getColumns(data, i);
        for (int j = 0; j < N; j++) {
            auto xj = getColumns(data, j);
            cov[i][j] = covariancee(xi, xj);
        }
    }

    return cov;
}

// Convert covariance matrix to volatility vector
std::vector<double> volatilityFromCov(const std::vector<std::vector<double>>& cov) {
    int N = cov.size();
    std::vector<double> vol(N);
    for (int i = 0; i < N; i++) {
        vol[i] = std::sqrt(cov[i][i]);
    }
    return vol;
}