#include <iostream>
#include "utils.h"
#include <vector>
#include "preservetopo.h"
#include <algorithm>
#include <execution>

//TRANSPOSE DATA  BEFORE FEEDING IT IN, WE NEED PER FEATURE EVO. Column by column analysis

//If you sort the actual x, you permanently lose the positional mapping — 
//which means you can’t compare x[i] with y[i] anymore because x[i] no longer means “original index i”.

double spearmanCorrelation(const std::vector<double> &raw, const std::vector<double> &processed){

    std::vector<double> A = convertToRank(raw, true);
    std::vector<double> B = convertToRank(processed, true);

    if(A.size() != B.size()){
        throw std::invalid_argument("[Spearman] Vectors need to be the same size!");
    }

    double rs = 0.0;

    int n = A.size();

    double sum = 0.0;

    for(int i = 0; i < n; i++){
        double di2 = (A[i] - B[i]) * (A[i] - B[i]);
        sum = sum + di2;
    }

    rs = 1 - ((6 * sum) / (n * ((n * n) - 1)));

    return rs; 
}

double pearsonCorrelation(std::vector<double> X, std::vector<double> Y){

    if(X.size() != Y.size()){
        throw std::invalid_argument("[Pearson] Vectors need to be the same size!");
    }

    double r = 0.0;

    int n = Y.size();

    double meanX = 0.0;
    double meanY = 0.0;

    for(size_t i = 0; i < n; i++){
        meanX = meanX + X[i];
        meanY = meanY + Y[i];
    }

    double sumXY = 0.0;
    double sumXsqrd = 0.0;
    double sumYsqrd = 0.0;

    for(size_t i = 0; i < n; i++){
        double xy = (X[i] - meanX) * (Y[i] - meanY);
        sumXY = sumXY + xy;

        double xsqrd = (X[i] - meanX) * (X[i] - meanX);
        sumXsqrd = sumXsqrd + xsqrd;

        double ysqrd = (Y[i] - meanY) * (Y[i] - meanY);
        sumYsqrd = sumYsqrd + ysqrd;
    }

    r = sumXY / sqrt(sumXsqrd * sumYsqrd);

    if (r == 0) {
        // Handle the case where denominator might be zero (e.g., perfect constant correlation)
        throw std::invalid_argument("[Pearson] Denominator is zero, correlation is undefined.");
    }

    return r;
}


std::vector<double> convertToRank(const std::vector<double> &X, bool toggleTieHandling){ 

    int n = X.size();
    std::vector<double> ranks(n);

    std::vector<std::pair<double, int>> vp;
    vp.reserve(n);
    for(int i = 0; i < n; i++){                       
        vp.emplace_back(X[i], i); //e.g. vp = [(23.5,0), (22.4,1), (11.1,2)]
    }

    std::sort(std::execution::par, //fuck it
        vp.begin(), vp.end(), //If I don’t provide a comparison function, it sorts using the < operator on the elements themselves
              [](const auto& a, const auto& b){ return a.first < b.first; }); //Need to avoid that cuz thats lexicographic bs.
    //Now it's vp = [(11.1, 2), (22.4, 1), (23.5, 0)]. 

    //TIE HANDLING MODULE (dont know if this is needed tho since were comparing with EMA + Z-scored vectors)
    if(toggleTieHandling){

       int i = 0;
        while(i < n){

        int j = i + 1;
         const double val = vp[i].first;

        // Find tie block end
         while(j < n && vp[j].first == val)
           j++;

               // Compute average rank (1-based)
            const double avg = ( (i+1) + j ) * 0.5;

        // Assign to all tied elements (vectorizable)
        for(int k = i; k < j; k++)
             ranks[vp[k].second] = avg;

          i = j;
       }

    }
    else{

      for(int r = 0; r < n; r++) 
       ranks[vp[r].second] = r + 1; //[3  2  1]     
                
    }

    return ranks;
}

double score(double mean_r, double median_r, double mean_rs, double median_rs){

    //I can use MAD here?

        double p_ratio = mean_r / median_r;
        double s_ratio = mean_rs / median_rs;

     return mean_rs + mean_r - std::abs(p_ratio - 1.0) - std::abs(s_ratio - 1.0); //the - 1 ensures we only like values pushing 1 
}

std::vector<int> knn(std::vector<std::vector<double>> data){

    double min_dist = 0.0;

    /* TOOLS
       euclidianDist'
       mean
       median
       MAD
    */

    std::vector<int> neighboursofeachvec;

    for(const auto &currentvector : data){
        for(const auto &vector_i : data){
            double dist = euclideanDist(currentvector, vector_i); //distance of each vector_i in the dataset to the currentvector

        }
    }








}

