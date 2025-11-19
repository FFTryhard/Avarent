#pragma once

#include <vector>
#include <deque>

class RawProcessing{

    public:

    std::vector<double> EMA_Online_Normalized(const std::vector<double>& row, double alpha);
    
    private: 
    std::vector<double> ema_mean;
    std::vector<double> ema_var;
    bool initialized = false;
    double epsilon = 1e-8;      // small number to avoid div by zero
    
    int o = 10;

};

class OnlineEntropy {
public:
    OnlineEntropy(size_t feature_dim, size_t window_size, int num_bins)
        : window_size(window_size), num_bins(num_bins)
    {
        windows.resize(feature_dim); // one buffer per feature
        bin_counts.resize(feature_dim, std::vector<int>(num_bins, 0));
    }

    std::vector<double> update(const std::vector<double>& normalized_row) {
        std::vector<double> entropies(normalized_row.size());

        for (size_t i = 0; i < normalized_row.size(); ++i) {
            double val = normalized_row[i];
            int bin_idx = binIndex(val);
            
            // Update sliding window
            if (windows[i].size() == window_size) {
                // Remove oldest
                int old_bin = binIndex(windows[i].front());
                bin_counts[i][old_bin]--;
                windows[i].pop_front();
            }

            // Add new
            windows[i].push_back(val);
            bin_counts[i][bin_idx]++;

            // Compute entropy
            entropies[i] = computeEntropy(bin_counts[i], windows[i].size());
        }

        return entropies;
    }

private:
    size_t window_size;
    int num_bins;
    std::vector<std::deque<double>> windows; // ring buffer per feature
    std::vector<std::vector<int>> bin_counts; // histogram per feature

    int binIndex(double val) {
        // Assume tanh-normalized range [-1, 1]
        double clipped = std::max(-1.0, std::min(1.0, val));
        int idx = static_cast<int>((clipped + 1.0) / 2.0 * num_bins);
        if (idx >= num_bins) idx = num_bins - 1;
        return idx;
    }

    double computeEntropy(const std::vector<int>& counts, size_t total_count) {
        double entropy = 0.0;
        double smooth_total = total_count + num_bins; // Laplace smoothing

        for (int c : counts) {
            double p = (c + 1) / smooth_total; // Laplace smoothing
            entropy += p * std::log2(1.0 / p);
        }

        // Normalize to [0, 1] if desired:
        double max_entropy = std::log2(num_bins);
        return entropy / max_entropy;
    }
};
 /*

 you can see where i injected noise, 
 and we can see the intrinsic turbulence of the data. 
 a lot values repeat but thats just because the resolution of my data is very low and alot of values fall into the same bin.
 once i get more complicated data that ranges from 1 - 100 instead of 20 to 30. we can see more. but this looks good
 
 
 */

