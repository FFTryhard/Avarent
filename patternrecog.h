#pragma once
#include <vector>
#include <iostream>
#include "outlierlogic.h"

class Regime{

    public:

     //return 0 if can be ignored, 1 for analysis, 2 for heavy alert

    int stateAnalysisPerCluster(std::vector<std::vector<PolarCoordinates>> data);
    int stateAnalysisPerCluster2(std::vector<std::vector<PolarCoordinates>> data);





    private:

    int currentRegime = -1; // Start with 'aligned' regime
    int stableSamples = 0;
    const int debounceThreshold = 3;
    const double hysteresisMargin = 0.1;
    double lastTheta = 0.0;

    int classifyRegime(double theta);
    void printLog(const std::string& message);
};


