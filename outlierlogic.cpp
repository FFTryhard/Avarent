#include <vector>
#include "kmeans.h"
#include <iostream>
#include <Eigen\Dense>
#include <cmath>
#include "outlierlogic.h"
#include "RDS.h"
#include "timechunking.h"
#include "utils.h"                                  
                                                                  
std::vector <std::vector <PolarCoordinates>> polar_csv_slidingwindow_fixed(SensorData &stream, int stride, int windowSize){ //  makes sliding window chunks

    std::vector<std::vector<PolarCoordinates>> windows;
    std::vector<PolarCoordinates> buffer;

    while (stream.hasNext()) {
        double r = stream.getNext();
        double cos_sim = stream.getNext();

        // Convert to angle        //acos converts into angle
        double theta = std::acos(std::clamp(cos_sim, -1.0, 1.0)); //safety boundaries
        //// your cosine similarity calc can spit out 1.00000002 or -1.0000001 because of floating-point rounding. Feed that raw into acos → boom, NaN.
        // Embed                     
        PolarCoordinates point { r, std::cos(theta), std::sin(theta) }; //set the point with embedded angle on sin and cos waves
        //geometrical contuinity
        buffer.push_back(point);

        // If buffer is full, save window
        if ((int)buffer.size() == windowSize) {
            windows.push_back(buffer);

            // Slide
            if (stride < windowSize) {
                buffer.erase(buffer.begin(), buffer.begin() + stride);
            } else {
                buffer.clear();
            }
        }
    }

    // Optional: keep last partial
    if (!buffer.empty()) {
        windows.push_back(buffer);
    }

           
    return windows;
}

bool hasNaN(const std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>& covmatrices) {
    for (const auto& matrix : covmatrices) {
        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j < 7; ++j) {
                if (std::isnan(matrix(i, j))) {
                    return true; // Found a NaN value
                }
            }
        }
    }
    return false; // No NaN values found
}






  


