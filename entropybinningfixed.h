/*

#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include "timechunking.h"
class Entropy {
public:
    // Constructor computes entropies for all windows.
    // If normalize == true, entropies are scaled to [0,1] by dividing by log2(bins).
    Entropy(const std::vector<std::vector<float>>& chunks_z_scored, int bins = 30, bool normalize = true);

    std::vector<float> getEntropies() const;

private:
    std::vector<float> entropies;
};

*/