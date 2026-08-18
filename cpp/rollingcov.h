#include <deque>
#include <stdexcept>
#include <vector>

class RollingCovariance {
public:
    RollingCovariance(std::size_t window, std::size_t dimensions)
        : window_(window),
          dim_(dimensions),
          sum_(dimensions, 0.0),
          sumOuter_(dimensions,
                    std::vector<double>(dimensions, 0.0))
    {
        if (window < 2)
            throw std::invalid_argument("Window must be at least 2.");

        if (dimensions == 0)
            throw std::invalid_argument("Dimension must be positive.");
    }

    void add(const std::vector<double>& x)
    {
        if (x.size() != dim_)
            throw std::invalid_argument("Dimension mismatch.");

        if (samples_.size() == window_)
            removeOldest();

        samples_.push_back(x);

        for (std::size_t i = 0; i < dim_; ++i)
        {
            sum_[i] += x[i];

            for (std::size_t j = 0; j < dim_; ++j)
                sumOuter_[i][j] += x[i] * x[j];
        }
    }

    bool ready() const
    {
        return samples_.size() == window_;
    }

    std::vector<std::vector<double>> covariance() const
    {
        std::size_t n = samples_.size();

        std::vector<std::vector<double>> cov(
            dim_, std::vector<double>(dim_, 0.0));

        if (n < 2)
            return cov;

        for (std::size_t i = 0; i < dim_; ++i)
        {
            for (std::size_t j = 0; j < dim_; ++j)
            {
                cov[i][j] =
                    (sumOuter_[i][j]
                     - sum_[i] * sum_[j] / static_cast<double>(n))
                    / static_cast<double>(n - 1);
            }
        }

        return cov;
    }

private:
    void removeOldest()
    {
        const auto& x = samples_.front();

        for (std::size_t i = 0; i < dim_; ++i)
        {
            sum_[i] -= x[i];

            for (std::size_t j = 0; j < dim_; ++j)
                sumOuter_[i][j] -= x[i] * x[j];
        }

        samples_.pop_front();
    }

    std::size_t window_;
    std::size_t dim_;

    std::deque<std::vector<double>> samples_;

    std::vector<double> sum_;
    std::vector<std::vector<double>> sumOuter_;
};
