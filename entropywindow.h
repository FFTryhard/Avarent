#pragma once
#include "entropywindow.h"
#include "timechunking.h"
#include <vector>
#include <deque>

class Entropy{

    public:

    Entropy(std::vector<std::vector<float>>&, bool);

    std::vector<float> getEntropies() const;

    private:

    std::vector <float> entropies;

};




