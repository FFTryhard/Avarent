#include <iostream>
#include <cmath>
#include <fstream> // For file output

int main() {
    // Initial conditions
    double r_start1 = 1.5, r_start2 = 5.9;        // Different starting r values for each cluster
    double r_increment1 = 0.76, r_increment2 = 0.39; // Different increments for diversity
    int time_steps = 1000;
    double osc_freq1 = 9 * M_PI / time_steps;
    double osc_freq2 = 5 * M_PI / time_steps;

    // Open a CSV file for writing
    std::ofstream file("C:/Users/H/Desktop/C++ Tryharding/.csv files/synthetic data/polarcoordinatesSE.csv");

    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    // Output header
    file << "r_cluster1,theta_cluster1,r_cluster2,theta_cluster2\n";

    for (int t = 0; t < time_steps; ++t) {
        // Cluster 1
        double r1 = r_start1 + r_increment1 * t;
        double theta1 = cos(osc_freq1 * t) * 0.7 * M_PI;

        // Cluster 2
        double r2 = r_start2 + r_increment2 * t;
        double theta2 = sin(osc_freq2 * t) * 0.7 * M_PI;

        // Write all values on the same row
        file << r1 << "," << theta1 / M_PI << ","
             << r2 << "," << theta2 / M_PI << "\n";
    }

    file.close();
    return 0;
}