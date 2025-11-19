#include "entropybinningfixed.h"
#include "timechunking.h"

Entropy::Entropy(const std::vector<std::vector<float>>& chunks_z_scored, int bins, bool normalize) {
    if (bins <= 1) bins = 2; // must be >=2

    // 1) Find global min/max across all windows
    bool any_value = false;
    double global_min = 0.0, global_max = 0.0;
    for (const auto& w : chunks_z_scored) {
        if (w.empty()) continue;
        double local_min = *std::min_element(w.begin(), w.end());
        double local_max = *std::max_element(w.begin(), w.end());
        if (!any_value) {
            global_min = local_min;
            global_max = local_max;
            any_value = true;
        } else {
            if (local_min < global_min) global_min = local_min;
            if (local_max > global_max) global_max = local_max;
        }
    }

    // If no data at all, nothing to do
    if (!any_value) return;

    // If global range is zero, every window is constant -> entropy 0
    double range = global_max - global_min;
    if (range == 0.0) {
        entropies.assign(chunks_z_scored.size(), 0.0);
        return;
    }

    const double bin_width = range / static_cast<double>(bins);

    entropies.reserve(chunks_z_scored.size());
    const double max_entropy = std::log2(static_cast<double>(bins)); // for normalization

    for (const auto& window : chunks_z_scored) {
        if (window.empty()) {
            entropies.push_back(0.0);
            continue;
        }

        std::vector<std::size_t> counts(static_cast<std::size_t>(bins), 0u);

        // Fill bins using GLOBAL min
        for (double v : window) {
            int idx = static_cast<int>(std::floor((v - global_min) / bin_width));
            if (idx < 0) idx = 0;
            if (idx >= bins) idx = bins - 1;
            counts[static_cast<std::size_t>(idx)]++;
        }

        // Compute probabilities and entropy
        const double n = static_cast<double>(window.size());
        double H = 0.0;
        for (std::size_t c : counts) {
            if (c == 0) continue;
            double p = static_cast<double>(c) / n;
            H -= p * std::log2(p); // -sum p log2 p
        }

        if (normalize) {
            if (max_entropy > 0.0) entropies.push_back(H / max_entropy);
            else entropies.push_back(0.0);
        } else {
            entropies.push_back(H);
        }
    }
}

std::vector<float> Entropy::getEntropies() const {
    return entropies;
}