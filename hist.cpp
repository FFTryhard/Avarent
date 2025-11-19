#include <iostream>
#include "utils.h"
#include <vector>
#include "hist.h"
#include <algorithm> // For min_element and max_element
#include <cmath>


void histogramAnalysis(const std::vector<double> &data){

    std::vector<double> vec = data;

    int size = vec.size();
    double minIt = *std::min_element(vec.begin(), vec.end());
    double maxIt = *std::max_element(vec.begin(), vec.end());

    double bin_width = FreedmanDiaconisRuleIQR(data, size);

    int numBins = calculateNumBins(maxIt, minIt, bin_width, size);

    std::cout << "Calculated Bin Width: " << bin_width << std::endl;
    std::cout << "Number of bins: " << numBins << std::endl;

    std::vector <int> bin_counts;
    bin_counts.resize(numBins); //each index corresponds to a bin (just cross reference while keeping bin width in mind)

    for(auto val : data){

      int bin_idx = binIndex(val, numBins);
      bin_counts[bin_idx]++;

    }

    std::cout << "Histogram:" << std::endl;

    double range = 0.0;

    for(auto val : bin_counts){
        std::cout << val << " [" << 1 * range << " - " << bin_width + range << "]";
        range = range + bin_width;
    }

    std::cout << std::endl;

    bin_counts.clear();
}


double FreedmanDiaconisRuleIQR(const std::vector<double> &data, int size){

    if (data.size() <= 10) {
        throw std::invalid_argument("Sample size must be greater than 10.");
    }

   std::vector<double> vec = data;

   std::sort(vec.begin(), vec.end());

   int q1idx = size / 4; // 100 / 4 = 25. = q1
   int q3idx = (size * 3) / 4; //100 * 3 = 300. 300 / 4 = 75

   double Q1 = vec[q1idx];
   double Q3 = vec[q3idx];
 
   double IQR = Q3 - Q1;

   double bin_width = (2 * IQR) / std::pow(size, 1.0/3.0);

   return bin_width;
}

double ScottsNormalReferenceRule(const std::vector<double>& data) {  //ion like this
    if (data.size() <= 1) {
        throw std::invalid_argument("Sample size must be greater than 1 for variance calculation.");
    }

    double mean = 0.0;
    double sum_of_squares = 0.0;

    // Compute the mean
    for (const auto& val : data) {
        mean += val;
    }
    mean /= data.size();

    // Calculate the sum of squared deviations from the mean
    for (const auto& val : data) {
        sum_of_squares += (val - mean) * (val - mean);
    }
    
    double stdev = sqrt(sum_of_squares / data.size());
    
    double bin_width = (3.5 * stdev) / std::pow(data.size(), 1.0/3.0);
    
    return bin_width;
}


// Function to calculate the number of bins
int calculateNumBins(double maxValue, double minValue, double binWidth, int n) {
    // Calculate the range of the data
    double range = maxValue - minValue;  // Expected to be 1 for your data range

    // Calculate the number of bins (using Freedman-Diaconis' bin width)
    int numBins = std::ceil(range / binWidth); /// mod this?

    return numBins;
}


int binIndex(double val, int num_bins) {
        int idx = static_cast<int>(val * num_bins); //drop the scaling
        if (idx >= num_bins) idx = num_bins - 1;
        return idx;
}

/*
int binIndex(double val, int num_bins) {
        // Assume tanh-normalized range [-1, 1]
        double clipped = std::max(-1.0, std::min(1.0, val));
        int idx = static_cast<int>((clipped + 1.0) / 2.0 * num_bins);
        if (idx >= num_bins) idx = num_bins - 1;
        return idx;
}
*/
/*
int binIndex(double val, int num_bins, double minValue, double maxValue) {
    // Scale the value between 0 and 1
    double scaled_val = (val - minValue) / (maxValue - minValue);
    int idx = static_cast<int>(scaled_val * num_bins);

    if (idx >= num_bins) idx = num_bins - 1; // Ensure the index is within bounds
    return idx;
}
*/