#include <iostream>
#include <vector>
#include <cmath>
#include "vectorangle.h"


std::vector<double> positionalEncode(double x, int d) {

    std::vector<double> encoding(d, 0.0); 
    
    for (int k = 0; k < d / 2; ++k) {
        double div = std::pow(10000.0, 2.0 * k / d);
        encoding[2 * k] = std::sin(x / div);
        if (2 * k + 1 < d) {
            encoding[2 * k + 1] = std::cos(x / div);
        }
    }

    return encoding;
}

//test

int main(){

    double error = 0.123; // example reconstruction error
    std::vector <double> x = positionalEncode(error, 41);

    std::vector<std::vector<double>> stressvectors;
    stressvectors.push_back(x);
    

    std::cout << "Encoded vector:\n";
    for (auto vector : stressvectors) {
        for(auto val : vector){

            std::cout << val << " ";

        }
    }
       
    std::cout << std::endl;

    return 0;

}