#pragma once

#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>

class IncrementalKMeans {
public:
    struct Cluster {
        std::vector<double> centroid;
        std::vector<double> variance; // EMA variance estimate

        size_t count = 0;
        double learningRate = 1.0;
    };

    IncrementalKMeans(
        size_t k,
        size_t dim,
        size_t initSamples = 512,
        double driftThreshold = 9.0,
        double minLearningRate = 0.02)
        : k(k),
          dim(dim),
          initSamples(initSamples),
          driftThreshold(driftThreshold),
          minLearningRate(minLearningRate),
          rng(std::random_device{}())
    {
        clusters.resize(k);

        for (auto& c : clusters) {
            c.centroid.assign(dim, 0.0);
            c.variance.assign(dim, 1.0);
        }

        reservoir.reserve(initSamples);
    }

    void partialFit(const std::vector<double>& x)
    {
        // Collect initialization samples.
        if (!initialized) {
            reservoir.push_back(x);

            if (reservoir.size() >= initSamples) {
                initialize();
                reservoir.clear();
                reservoir.shrink_to_fit();
                initialized = true;
            }

            return;
        }

        size_t idx = nearestCluster(x);
        updateCluster(idx, x);
    }

    void partialFitBatch(const std::vector<std::vector<double>>& batch)
    {
        for (const auto& x : batch)
            partialFit(x);
    }

    size_t predict(const std::vector<double>& x) const
    {
        double best = std::numeric_limits<double>::max();
        size_t bestIdx = 0;

        for (size_t i = 0; i < k; ++i) {
            double d = distanceSquared(x, clusters[i].centroid);

            if (d < best) {
                best = d;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    const std::vector<Cluster>& getClusters() const
    {
        return clusters;
    }

private:
    size_t k;
    size_t dim;
    size_t initSamples;

    double driftThreshold;
    double minLearningRate;

    bool initialized = false;

    std::vector<Cluster> clusters;
    std::vector<std::vector<double>> reservoir;

    std::mt19937 rng;

    //------------------------------------------

    double distanceSquared(
        const std::vector<double>& a,
        const std::vector<double>& b) const
    {
        double d = 0.0;

        for (size_t i = 0; i < dim; ++i) {
            double diff = a[i] - b[i];
            d += diff * diff;
        }

        return d;
    }

    //------------------------------------------

    void initialize()
    {
        if (reservoir.size() < k)
            return;

        std::uniform_int_distribution<size_t> randomIndex(
            0,
            reservoir.size() - 1);

        clusters[0].centroid = reservoir[randomIndex(rng)];

        std::vector<double> distances(reservoir.size());

        for (size_t c = 1; c < k; ++c) {

            double total = 0.0;

            for (size_t i = 0; i < reservoir.size(); ++i) {

                double best = std::numeric_limits<double>::max();

                for (size_t j = 0; j < c; ++j)
                    best = std::min(
                        best,
                        distanceSquared(
                            reservoir[i],
                            clusters[j].centroid));

                distances[i] = best;
                total += best;
            }

            std::uniform_real_distribution<double> pick(0.0, total);

            double target = pick(rng);
            double cumulative = 0.0;

            size_t chosen = 0;

            for (size_t i = 0; i < reservoir.size(); ++i) {

                cumulative += distances[i];

                if (cumulative >= target) {
                    chosen = i;
                    break;
                }
            }

            clusters[c].centroid = reservoir[chosen];
        }
    }

    //------------------------------------------

    size_t nearestCluster(const std::vector<double>& x) const
    {
        double best = std::numeric_limits<double>::max();
        size_t idx = 0;

        for (size_t i = 0; i < k; ++i) {

            double d = distanceSquared(
                x,
                clusters[i].centroid);

            if (d < best) {
                best = d;
                idx = i;
            }
        }

        return idx;
    }

    //------------------------------------------

    void updateCluster(
        size_t idx,
        const std::vector<double>& x)
    {
        Cluster& c = clusters[idx];

        ++c.count;

        double baseLR =
            std::max(
                minLearningRate,
                1.0 / std::sqrt(static_cast<double>(c.count)));

        double distance = distanceSquared(
            x,
            c.centroid);

        double expected = 0.0;

        for (double v : c.variance)
            expected += std::max(v, 1e-6);

        bool drift = distance > driftThreshold * expected;

        c.learningRate = drift ? 0.5 : baseLR;

        for (size_t i = 0; i < dim; ++i) {

            double delta = x[i] - c.centroid[i];

            c.centroid[i] += c.learningRate * delta;

            double delta2 = delta * delta;

            c.variance[i] =
                (1.0 - c.learningRate) * c.variance[i] +
                c.learningRate * delta2;
        }
    }
};