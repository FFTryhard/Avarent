#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>

double generate_anomaly(int i) {
    return (rand() % 200 - 100) * 0.1; // spikes between -10 and 10
}

int main() {
    srand(time(0));
    const int N = 10000;  // number of timesteps
    const int streams = 7;

    std::vector<std::string> filepaths = {
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/synthetic data/data1.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/synthetic data/data2.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/synthetic data/data3.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/synthetic data/data4.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/synthetic data/data5.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/synthetic data/data6.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/synthetic data/data7.csv"
    };

    std::ofstream outfile("structured_streams.csv");
    if (!outfile) { std::cerr << "Error opening file!\n"; return 1; }

    outfile << "t,"; 
    for(int s=0;s<streams;s++){
        outfile << "s" << s+1;
        if(s<streams-1) outfile << ",";
    }
    outfile << "\n";

    for(int i=0;i<N;i++){
        outfile << i << ",";
        for(int s=0;s<streams;s++){
            double base = 20 * sin(i*0.01 + s) + 5 * cos(i*0.03 + 2*s);  // slightly different per stream
            double noise = (rand() % 1000 - 500) * 0.001;                // small high-freq noise
            double val = base + noise;

            // Insert anomaly 0.5% of the time
            if(rand() % 200 == 0) val += generate_anomaly(i);

            outfile << std::fixed << std::setprecision(3) << val;
            if(s < streams-1) outfile << ",";
        }
        outfile << "\n";
    }

    outfile.close();
    std::cout << "Structured streams generated.\n";
    return 0;
}

























 