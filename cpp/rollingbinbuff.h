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

public:
    RollingBinBuffer(int size) : buffer_size(size)
    {
        buffer.reserve(buffer_size);
    }

    // push a new value
    void push(double val)
    {
        if (buffer.size() >= buffer_size)
        {
            buffer.erase(buffer.begin());
        }
        buffer.push_back(val);
    }

    int eval(double val) const // cant change anything in it
    {
        // trivial thresholds, tweak later
        if (val < -0.20) // 0.20, 0.10 with 6 and dt1.0seems goodd. just need to work on delay
            return -2;
        if (val < -0.10)
            return -1;
        if (val < 0.10)
            return 0;
        if (val < 0.20)
            return 1;
        return 2;
    }

    // compute the most frequent bin currently in the buffer
    int exec()
    {
        if (buffer.empty())
            return 0; // fallback

        double val = derivativeOLS(buffer, 1.0);

        int bin = eval(val); // or s

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

        if (count >= 2) // if is ee flicker bump to three
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