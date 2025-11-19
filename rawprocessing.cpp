#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "rawprocessing.h"

std::vector<double> RawProcessing::EMA_Online_Normalized(const std::vector<double>& row, double alpha) {

        if (row.empty()) return {};

            if (!initialized) {
                ema_mean = row;
                ema_var.resize(row.size(), 1e-4); // small initial variance to avoid zero std  1e- 2This smooths startup behavior (less erratic normalization in first few frames).
                initialized = true;

                // First output is zeros since no variance yet
                return std::vector<double>(row.size(), 0.0);
        }

        std::vector<double> normalized(row.size());
        const double variance_threshold = 1e-6;  // I can adjust this threshold
        

        for (size_t i = 0; i < row.size(); ++i) {

            //double alpha = base_alpha / (1.0 + ema_var[i] * o); //making k dynamic can be too much freedom, clustering needs consistency

            double x = row[i];

            // Update EMA mean
            double prev_mean = ema_mean[i];
            ema_mean[i] = alpha * x + (1 - alpha) * ema_mean[i];

            // Update EMA variance (variance of samples around the mean)
            double diff = x - prev_mean;
            ema_var[i] = alpha * (diff * diff) + (1 - alpha) * ema_var[i];

            if (ema_var[i] < variance_threshold) {
            normalized[i] = 0.0;  
            continue;  // Skip further processing for this dimension
            }

            /*
            Ifyour vector suddenly jumps from its proper z-score (maybe 0.2 or -0.5) to 1, regardless of context.
            When Mahalanobis multiplies that by an inverse covariance with small eigenvalues (1/0.01 = 100), 
            the resulting contribution to distance is ≈100² * (Δ=1)² → 10,000, easily explaining those 9k+ spikes.
            */

            // Calculate std deviation, avoid division by zero
            double std_dev = std::sqrt(ema_var[i]) + epsilon;

            // Normalize: z-score, tanh was fucking me up destroys
            double z = (x - ema_mean[i]) / std_dev;
            normalized[i] = z; /// (1 + std::abs(z)); //this can compress cluster seperation
        }

        return normalized;  
}
