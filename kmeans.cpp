#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib> 
#include "kmeans.h"
#include <unordered_map>
#include <numeric>
#include <iostream>
#include <ctime>
#include <Eigen\Dense>
#include "utils.h"
#include <omp.h>
#include <random>

std::vector<double> computeCentroid(const std::vector<std::vector<double>>& points) {

    //takes in clusters and produces the centroid (mean of the coordinates)

    std::vector<double> centroid(points[0].size(), 0.0); //initializes a vector waiting to get data
    for (auto& p : points) {
        for (size_t i = 0; i < p.size(); i++) {
            centroid[i] = centroid [i] + p[i]; //every index in the centroid vector gets the value of the point in question
        }
    }
    for (size_t i = 0; i < centroid.size(); i++) {
        centroid[i] /= points.size(); //every value gets divided by the total size to get the mean
    }
    return centroid;

}

std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>, double> kmeans(const std::vector<std::vector<double>>& data, int k, int max_iters) {
    int n = data.size(); //number of vectors
    int d = data[0].size(); //num dimensions in each vector

    for (size_t i = 0; i < data.size(); ++i) {
    if (data[i].empty()) {
        std::cerr << "[kmeans] Error: Row " << i << " in X is empty!" << std::endl;
    }
    }

    std::vector<std::vector<std::vector<double>>> clusters(k); //
 
    std::vector<std::vector<double>> centroids = initializeCentroidsKMeansPP(data, k);
    
    std::vector<int> labels(n, 0); //makes a vector with n dimensions to 0

    for (int iter = 0; iter < max_iters; iter++) {

        bool changed = false;

        for (int c = 0; c < k; ++c) {
           clusters[c].clear();    //perf
        }

        // 2. Assign each point to closest centroid
        for (int i = 0; i < n; i++) {

            double bestDist = std::numeric_limits<double>::max(); //returns max value with type<>
            int bestCluster = 0;

            for (int c = 0; c < k; c++) {

                double dist = euclideanDist(data[i], centroids[c]); //takes each vector, calculates distance to the centroids that are k num

                if (dist < bestDist) {
                    bestDist = dist;
                    bestCluster = c;
                }
            }
            if (labels[i] != bestCluster) {
                labels[i] = bestCluster;
                changed = true;
            }
        }

        // 3. Recompute centroids

        // - 
       
        for (int i = 0; i < n; i++) {
            clusters[labels[i]].push_back(data[i]);
        }

        for (int c = 0; c < k; c++) {
            if (!clusters[c].empty()) {
                centroids[c] = computeCentroid(clusters[c]);
            }
        }

        // stop early if no labels changed
        if (!changed) break;
    }
    /*
    for(auto i : labels){
        std::cout << i << " ";
    }
    */
    int total = 0;
    for (const auto& cluster : clusters) {
    total += cluster.size();
     
     }
    std::cout << "Total points in clusters = " << total << std::endl;
    std::cout << std::endl;

    std::cout << "Calling silhouetteScore..." << std::endl;

    double score = silhouetteScore(data, labels); //how can i get a vector here

    //std::cout << "Silhouette score = " << score << std::endl;
 
    return {clusters, centroids, score}; //tuple. clusters (cluster id, vectors. specific dimensions)
}

std::vector<std::vector<double>> initializeCentroidsKMeansPP(const std::vector<std::vector<double>>& data, int k) {
    int n = data.size();
    std::vector<std::vector<double>> centroids;
    std::vector<double> distances(n, std::numeric_limits<double>::max());

    int x = 42;

    std::mt19937 gen(x);
    std::uniform_int_distribution<> distrib(0, n - 1);

    // 1. Randomly choose the first centroid
    centroids.push_back(data[distrib(gen)]);
    
    for (int c = 1; c < k; ++c) {
        double totalDist = 0.0;

        // 2. Compute distance from each point to nearest centroid
        for (int i = 0; i < n; ++i) {
            double dist = euclideanDist(data[i], centroids.back());
            dist *= dist;
            if (dist < distances[i]) {
                distances[i] = dist;
            }
            totalDist += distances[i];
        }

        // 3. Choose a new point weighted by distance
        std::uniform_real_distribution<> realDist(0.0, totalDist);
        double r = realDist(gen);
        double cumulative = 0.0;
        int selectedIndex = 0;

        for (int i = 0; i < n; ++i) {
            cumulative += distances[i];
            if (r <= cumulative) {
                selectedIndex = i;
                break;
            }
        }

        centroids.push_back(data[selectedIndex]);
    }

    return centroids;
}


double silhouetteScore(const std::vector<std::vector<double>>& X, const std::vector<int>& labels) {
    int N = X.size();
    if (N == 0) {
        std::cout << "[silhouetteScore] Data is empty. Returning 0.0" << std::endl;
        return 0.0;
    }

    std::cout << "[silhouetteScore] Starting. N = " << N << ", labels = " << labels.size() << std::endl;

    // Map cluster -> indices of its points
    std::unordered_map<int, std::vector<int>> clusters;

    for (int i = 0; i < N; i++) {
        clusters[labels[i]].push_back(i);
    }

    double totalScore = 0.0;

    for (int i = 0; i < N; i++) {
        
        int cluster_i = labels[i];

        // --- Compute a(i): intra-cluster distance ---
        double a = 0.0;
        int clusterSize = clusters[cluster_i].size();

        if (clusterSize > 1) {
            for (int j : clusters[cluster_i]) {
                if (j == i) continue;
                double dist = euclideanDist(X[i], X[j]);
                if (std::isnan(dist)) {
                    std::cerr << "[Error in Sscore] NaN distance in intra-cluster at point " << i << std::endl;
                    return 0.0;
                }
                a += dist;
            }
            a /= (clusterSize - 1);
        }

        // --- Compute b(i): nearest-cluster distance ---
        double b = std::numeric_limits<double>::max();
        for (auto& kv : clusters) {
            int cluster_j = kv.first;
            if (cluster_j == cluster_i) continue;

            if (kv.second.empty()) {
                std::cerr << "[Warning in Sscore] Skipping empty cluster " << cluster_j << std::endl;
                continue;
            }

            double distSum = 0.0;
            for (int j : kv.second) {
                double dist = euclideanDist(X[i], X[j]);
                if (std::isnan(dist)) {
                    std::cerr << "[Error in Sscore] NaN distance in inter-cluster at point " << i << " to cluster " << cluster_j << std::endl;
                    return 0.0;
                }
                distSum += dist;
            }

            double avgDist = distSum / kv.second.size();
            if (avgDist < b) b = avgDist;
        }

        // --- Compute silhouette for point i ---
        double s = 0.0;
        if (clusterSize > 1 && std::max(a, b) > 0.0) {
            s = (b - a) / std::max(a, b);
        }

        if (std::isnan(s)) {
            std::cerr << "[Error in Sscore] NaN silhouette score for point " << i << " (a = " << a << ", b = " << b << ")" << std::endl;
            return 0.0;
        }

        totalScore += s;

        // Optional: show s(i) for each point
        // std::cout << "  s(" << i << ") = " << s << std::endl;
    }

    double finalScore = totalScore / N;

    return finalScore;
}

std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>, double> kmeansM(const std::vector<std::vector<double>>& data, int k, int max_iters) {

    //euclidean warmp up 
    int n = data.size(); //number of vectors
    int d = data[0].size(); //num dimensions in each vector

    for (size_t i = 0; i < data.size(); ++i) {
    if (data[i].empty()) {
        std::cerr << "[kmeansM] Error: Row " << i << " in X is empty!" << std::endl;
    }
    }

    int stablerounds = 0;

    bool trigger = false;

    bool WarmupComplete = true;

    std::vector<std::vector<std::vector<double>>> clusters(k); //
 
    std::vector<std::vector<double>> centroids = initializeCentroidsKMeansPP(data, k);

    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> covariancematrices;
    
    std::vector<int> labels(n, 0); //makes a vector with n dimensions to 0

    for (int iter = 0; iter < max_iters; iter++) {

        bool changed = false;

        for (int c = 0; c < k; ++c) {
           clusters[c].clear();    //perf
        }

        // 2. Assign each point to closest centroid
        for (int i = 0; i < n; i++) {

            double bestDist = std::numeric_limits<double>::max(); //returns max value with type<>
            int bestCluster = 0;

            for (int c = 0; c < k; c++) {
                
                double dist = 0.0;

                if(trigger){
                    double smoothness = std::max(2.0, std::log(avgClusterSize(clusters) + 1.0));
                    double blend = distanceBlend(iter, trigger, iter, smoothness); //5 or smoothness
                    dist = (1 - blend) * euclideanDist(data[i], centroids[c])  //if blend is 1 its fully mahalanobis, if its 0. euclidean
                      + blend * mahalanobisDistanceSingle(toEigenVector(data[i]), toEigenVector(centroids[c]), covariancematrices[c]);
                }
                else{
                    dist = euclideanDist(data[i], centroids[c]);
                }

                if (dist < bestDist) {
                    bestDist = dist;
                    bestCluster = c;
                }

            }
            if (labels[i] != bestCluster) {
                labels[i] = bestCluster;
                changed = true;
            }
        }

        // 3. Recompute centroids

        // - 

        int failcount = 0;
       
        for (int i = 0; i < n; i++) {
            clusters[labels[i]].push_back(data[i]);
        }

        for (int c = 0; c < k; c++) {

            if (!clusters[c].empty()) {
                centroids[c] = computeCentroid(clusters[c]);
            }

        }
        
        for(int o = 0; o < covariancematrices.size(); o++){

          if (clusters[o].size() < d) {
                covariancematrices[o] = 1e-4 * Eigen::MatrixXd::Identity(d, d); //If a cluster has fewer points than dimensions, you literally can’t compute a stable covariance.
                continue;
           }

        }

       if(iter % 3 == 0){   

          covariancematrices = CovarianceMatrixGPT(clusters, centroids, data);
          WarmupComplete = true; //otherwise once its false it will never recover

          for(auto &matrix : covariancematrices){

              if (matrix.rows() != matrix.cols() || matrix.rows() == 0){
                  WarmupComplete = false;  //should never happen
                  continue;
              } 

                if(!choleskyDecompLLT(matrix)){
                    WarmupComplete = false; //let covariance handle reguralization
                    failcount++;
                    if(failcount > 2){
                        matrix = 1e-4 * Eigen::MatrixXd::Identity(d, d);
                    }
                }
                else{
                    failcount = 0;
                }

           }
       }

       if(WarmupComplete) 
          stablerounds++;
       else
          stablerounds = 0;

       if (iter > 10 && stablerounds >= 5 && allClustersHaveEnoughPoints(clusters, d, 2)) //2 is fine, standard
          trigger = true;

       if (trigger && iter % 2 == 0){
          auto new_covariances = CovarianceMatrixGPT(clusters, centroids, data);
           for (int c = 0; c < k; c++) {
           double alphaCov = adaptiveAlpha(clusters[c].size());
           covariancematrices[c] = (1 - alphaCov) * covariancematrices[c] + alphaCov * new_covariances[c];
          }     
       }

        // stop early if no labels changed
        
        if (!changed) break;
    }
   
    int total = 0;
    for (const auto& cluster : clusters) {
    total += cluster.size();
     
     }
    std::cout << "Total points in clusters = " << total << std::endl;
    std::cout << std::endl;

    std::cout << "Calling silhouetteScore..." << std::endl;

    if (covariancematrices.empty()) {
       covariancematrices = CovarianceMatrixGPT(clusters, centroids, data);
       std::cout << "Cov matrices: " << covariancematrices.size() << " clusters: " << clusters.size() << std::endl;
    }

    double score = silhouetteScoreMahalanobisOptimized(data, labels, clusters, covariancematrices); 

    
    WarmupComplete = false;

    trigger = false;

    stablerounds = 0;
 
    return {clusters, centroids, score}; //tuple. 
}

double silhouetteScoreMahalanobisOptimized(
    const std::vector<std::vector<double>>& data,
    const std::vector<int>& labels,
    const std::vector<std::vector<std::vector<double>>>& clusters,
    const std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>& covariancematrices
) {
    
    const int N = (int)data.size();
    if (N == 0) return 0.0;
    if (labels.size() != data.size()) {
        std::cerr << "[ERROR in SscoreMO] labels.size() != data.size()" << std::endl;
        return 0.0;
    }

    const int K = (int)clusters.size();

    // --- Precompute inverse covariances
    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> invCovs(K);
    #pragma omp parallel for
    for (int c = 0; c < K; ++c)
        invCovs[c] = getInvCov(covariancematrices[c]);

    // --- Precompute Eigen data
    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::VectorXd>> eigenData(N); 
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
        eigenData[i] = toEigenVector(data[i]);

    // --- Precompute Eigen clusters
    std::vector<std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::VectorXd>>> eigenClusters(K); 
    #pragma omp parallel for
    for (int c = 0; c < K; ++c) {
        const auto& pts = clusters[c];
        eigenClusters[c].resize(pts.size());
        for (size_t i = 0; i < pts.size(); ++i)
            eigenClusters[c][i] = toEigenVector(pts[i]);
    }

    // --- Precompute intra-cluster distances
    std::vector<std::vector<std::vector<double>>> intraDist(K);
    #pragma omp parallel for
    for (int c = 0; c < K; ++c) {
        const int nPts = (int)eigenClusters[c].size();
        intraDist[c].assign(nPts, std::vector<double>(nPts, 0.0));
        for (int i = 0; i < nPts; ++i) {
            for (int j = i + 1; j < nPts; ++j) {
                double dist = mahalanobisDistanceSingleFast(eigenClusters[c][i], eigenClusters[c][j], invCovs[c]);
                intraDist[c][i][j] = dist;
                intraDist[c][j][i] = dist;
            }
        }
    }

    auto almostEqual = [](const std::vector<double>& a, const std::vector<double>& b) {
                  if (a.size() != b.size()) return false;
                    for (size_t k = 0; k < a.size(); ++k)
                      if (std::abs(a[k] - b[k]) > 1e-9)
                                return false;
                         return true;
                       };

    // --- Map each point to its cluster-local index for O(1) lookup
    std::vector<int> localIndex(N, -1);
    #pragma omp parallel for
    for (int c = 0; c < K; ++c) {
        const auto& cluster = clusters[c];
        for (size_t j = 0; j < cluster.size(); ++j) {
            const auto& pt = cluster[j];
            // Assume equality by position rather than by floating-point compare
            // (since these should be the same references as in data)
            for (size_t i = 0; i < data.size(); ++i) {
                if (almostEqual(pt, data[i])) {  // if same reference (if not, match by content if needed)
                    localIndex[i] = (int)j;
                    break;
                }
            }
        }
    }

    double totalScore = 0.0;

    // --- Compute silhouette scores (parallelized)
    #pragma omp parallel for reduction(+:totalScore)
    for (int i = 0; i < N; ++i) {
        int cluster_i = labels[i];
        if (cluster_i < 0 || cluster_i >= K) continue;

        int idxInCluster = localIndex[i];
        if (idxInCluster < 0) continue;

        const int clusterSize = (int)eigenClusters[cluster_i].size();
        if (clusterSize == 0) continue;

        // --- a(i): average intra-cluster distance
        double a = 0.0;
        if (clusterSize > 1) {
            for (int j = 0; j < clusterSize; ++j) {
                if (j == idxInCluster) continue;
                a += intraDist[cluster_i][idxInCluster][j];
            }
            a /= (clusterSize - 1);
        }

        // --- b(i): min average distance to another cluster
        double b = std::numeric_limits<double>::max();
        const auto& xi = eigenData[i];
        for (int c = 0; c < K; ++c) {
            if (c == cluster_i || eigenClusters[c].empty()) continue;
            double avgDist = 0.0;
            for (const auto& pt : eigenClusters[c])
                avgDist += mahalanobisDistanceSingleFast(xi, pt, invCovs[c]);
            avgDist /= eigenClusters[c].size();
            if (avgDist < b) b = avgDist;
        }

        double s = 0.0;
        if (clusterSize > 1 && std::max(a, b) > 0.0)
            s = (b - a) / std::max(a, b);

        totalScore += s;
    }

    return totalScore / N;
}













 