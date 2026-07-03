#pragma once
#include <vector>
#include <cmath>

struct RollingBinBuffer
{
private:
    int candidate_bin = 0;
    int last_bin = 0;
    int count = 0;
    int buffer_size;
    std::vector<double> buffer;

    // Bin edge thresholds for 5-level GEC derivative discretization
    double BIN_EDGE_STRONG_NEG;
    double BIN_EDGE_WEAK_NEG;
    double BIN_EDGE_WEAK_POS;
    double BIN_EDGE_STRONG_POS;
    int    BIN_PERSISTENCE_THRESHOLD;

public:
    // Full constructor — Engine uses this with config values
    RollingBinBuffer(int size, double bin_weak, double bin_strong, int persistence)
        : buffer_size(size),
          BIN_EDGE_STRONG_NEG(-bin_strong), BIN_EDGE_WEAK_NEG(-bin_weak),
          BIN_EDGE_WEAK_POS(bin_weak), BIN_EDGE_STRONG_POS(bin_strong),
          BIN_PERSISTENCE_THRESHOLD(persistence)
    {
        buffer.reserve(buffer_size);
    }

    // Legacy constructor — backward compat for standalone runner
    RollingBinBuffer(int size)
        : RollingBinBuffer(size, 0.10, 0.20, 2) {}

    // push a new value
    void push(double val)
    {
        if (buffer.size() >= buffer_size)
        {
            buffer.erase(buffer.begin());
        }
        buffer.push_back(val);
    }

    int eval(double val) const
    {
        if (val < BIN_EDGE_STRONG_NEG)
            return -2;
        if (val < BIN_EDGE_WEAK_NEG)
            return -1;
        if (val < BIN_EDGE_WEAK_POS)
            return 0;
        if (val < BIN_EDGE_STRONG_POS)
            return 1;
        return 2;
    }

    // compute the most frequent bin currently in the buffer
    int exec()
    {
        if (buffer.empty())
            return 0; // fallback

        double val = derivativeOLS(buffer, 1.0);

        int bin = eval(val);

        update(bin);

        return last_bin;
    }

    void update(int new_bin)
    {
        if (new_bin == candidate_bin)
        {
            count++;
        }
        else
        {
            candidate_bin = new_bin;
            count = 1;
        }

        if (count >= BIN_PERSISTENCE_THRESHOLD)
        {
            last_bin = candidate_bin;
            count = 0;
        }
    }

    double derivativeOLS(const std::vector<double> &v, double dt = 1.0)
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
};