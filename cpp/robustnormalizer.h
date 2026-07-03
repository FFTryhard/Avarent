#pragma once

#include <deque>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "utils.h"

struct EMA
{
public:
    explicit EMA(double alpha) : alpha(alpha) {}

    double update(double x)
    {
        if (!initialized)
        {
            value = x;
            initialized = true;
            return value;
        }

        value += alpha * (x - value);
        return value;
    }

    double get() const { return value; }

    bool ready() const { return initialized; }

    void reset()
    {
        initialized = false;
        value = 0.0;
    }

private:
    double alpha;       // smoothing factor (0..1)
    double value = 0.0;
    bool initialized = false;
};

struct ScalarBuffer_
{
public:

    void add(double value)
    {
        if (!has_curr_)
        {
            curr_ = value;
            has_curr_ = true;
            return;
        }

        prev_ = curr_;
        curr_ = value;
        has_prev_ = true;
    }

    bool hasCurrent() const
    {
        return has_curr_;
    }

    bool hasPrevious() const
    {
        return has_prev_;
    }

    double current() const
    {
        if (!has_curr_)
            throw std::runtime_error("ScalarBuffer: current value not initialized");
        return curr_;
    }

    double previous() const
    {
        if (!has_prev_)
            throw std::runtime_error("ScalarBuffer: previous value not available");
        return prev_;
    }

    void reset()
    {
        has_curr_ = false;
        has_prev_ = false;
    }

private:
    double prev_ = 0.0;
    double curr_ = 0.0;

    bool has_curr_ = false;
    bool has_prev_ = false;
};

class RobustScore {
public:
    enum class Mode {
        Signed,      // z = (x - med) / sigma
        Positive,    // z = max(0, signed_z)
        Absolute     // z = abs(signed_z)
    };

    struct Out {
        double raw = 0.0;
        double median = 0.0;
        double mad = 0.0;
        double sigma = 0.0;
        double z_signed = 0.0;
        double z = 0.0;        // post-mode transformed
        bool valid = false;
    };

    RobustScore(
        int history_max = 100,
        int warmup_min = 30,
        double mad_floor = 1e-6,
        double cap_value = 8.0
    )
        : history_max_(std::max(5, history_max)),
          warmup_min_(std::max(5, warmup_min)),
          mad_floor_(std::max(1e-12, mad_floor)),
          cap_value_(std::max(0.1, cap_value)) {}

    void reset() {
        hist_.clear();
    }

    Out process(double x, Mode mode) {
        Out out;
        out.raw = x;

        if ((int)hist_.size() < warmup_min_) {
            hist_.push_back(x);
            trim();
            out.valid = false;
            return out;
        }

        std::vector<double> v(hist_.begin(), hist_.end());
        const double med = median(v);

        for (double& a : v) a = std::abs(a - med);
        const double mad = median(v);

        const double sigma = std::max(1.4826 * mad, mad_floor_);
        const double z_signed = (x - med) / sigma;

        double z = 0.0;
        switch (mode) {
            case Mode::Signed:
                z = z_signed;
                break;
            case Mode::Positive:
                z = std::max(0.0, z_signed);
                break;
            case Mode::Absolute:
                z = std::abs(z_signed);
                break;
        }

        if (cap_value_ > 0.0) {
            if (mode == Mode::Signed) {
                z = std::max(-cap_value_, std::min(cap_value_, z));
            } else {
                z = std::min(cap_value_, z);
            }
        }

        out.median = med;
        out.mad = mad;
        out.sigma = sigma;
        out.z_signed = z_signed;
        out.z = z;
        out.valid = true;

        hist_.push_back(x);
        trim();
        return out;
    }

private:
    int history_max_;
    int warmup_min_;
    double mad_floor_;
    double cap_value_;
    std::deque<double> hist_;

    void trim() {
        while ((int)hist_.size() > history_max_) hist_.pop_front();
    }
};