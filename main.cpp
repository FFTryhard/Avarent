#include <iostream>
#include "RDS.h"
#include "timechunking.h"
#include "entropywindow.h"
#include <fstream>
#include "entropybinningfixed.h"
#include "kmeans.h"
#include <cmath>
#include <Eigen/Dense>
#include "outlierlogic.h"
#include "fft.h"
#include <iomanip>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <limits>
#include "rawprocessing.h"
#include "patternrecog.h"
#include "utils.h"
#include "vis_clusters.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <filesystem>
#include "cppumap.h"
#include "preservetopo.h"


using namespace FFT;

double dotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2);

double cosineSimilarityCPP(const std::vector<double>& vec1, const std::vector<double>& vec2);

double magnitude(const std::vector<double>& vec);

bool checkForNaN(const std::vector<double>& data);

std::vector <PolarCoordinates> makePolar(std::vector <double>, std::vector<double>);

void transposePolarSignalData(std::vector<std::vector<std::vector<PolarCoordinates>>>& data);

void transposeInPlace(std::vector<std::vector<double>>& matrix);

void print3dPolar(std::vector<std::vector<std::vector<PolarCoordinates>>> data);

void logTransformAndScale(std::vector<double>& mahalanobisdistances, double minRange, double maxRange);

std::vector<std::vector<double>> transposeEntropies(const std::vector<Entropy>& entropies);

std::vector<std::vector<double>> generatePoints(int numVectors, int vectorSize);

std::vector<std::vector<PolarCoordinates>> slidingWindows(const std::vector<PolarCoordinates> &data, 
                                           size_t windowSize, 
                                           size_t stride = 1);

void dumpPolarCSV(const std::string &filename,
                  const std::vector<std::vector<double>> &rows,
                  int numClusters,
                  const std::vector<std::string> &cluster_names);

void dumpToCSV(const std::vector<Entropy>& entropies, std::vector<TimeChunking>& stream_chunks, const std::string& filename, const std::vector <std::string> &streams);

std::vector<double> FourierExpansion(double x, int d, int repeat_count);

std::vector <double> getCosSim(const std::vector<std::vector<double>>& centroids, std::vector<double> &p);

struct Timer
{

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<float> duration;

    Timer(){
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer(){
        end = std::chrono::high_resolution_clock::now();

        duration = end - start;

        float ms = duration.count() * 1000.0f;
        std::cout << "Timer took: " << ms << "ms" << std::endl;
    }
    
};

struct PCAdata{
    std::vector<std::vector<double>> pcarawvecs;
    std::vector<std::vector<double>> pcaentropyvecs;
};

struct SinCos{
    double sinval;
    double cosval;
};

int main() {

    //g++ main.cpp RDS.cpp timechunking.cpp entropywindow.cpp outlierlogic.cpp kmeans.cpp rawprocessing.cpp patternrecog.cpp utils.cpp vis_clusters.cpp -o myprogram

    srand(time(0));

    Timer timer;

    std::vector <std::string> files_normalx = 
    {
        "C:/Users/H/Desktop/cpp/.csv files/normal sensor data/tempstable.csv", //temp
        "C:/Users/H/Desktop/cpp/.csv files/normal sensor data/humidity(stable).csv",   //hum
        "C:/Users/H/Desktop/cpp/.csv files/normal sensor data/voltage(stable).csv",    //v
        "C:/Users/H/Desktop/cpp/.csv files/normal sensor data/current(stable).csv",    //curr
        "C:/Users/H/Desktop/cpp/.csv files/normal sensor data/power(stable).csv",      //w
        "C:/Users/H/Desktop/cpp/.csv files/normal sensor data/vibration(stable).csv",   //freqs
        "C:/Users/H/Desktop/cpp/.csv files/normal sensor data/stable.csv"       //stable
    };

    std::vector <std::string> files_normal = 
    {
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS1/x1.csv", //multivariate higher res
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS1/x2.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS1/x3.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS1/x4.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS1/x5.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS1/x6.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS1/x7.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS1/x8.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS1/x9.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS1/x10.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS1/x11.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS1/x12.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS1/x13.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS1/y.csv" 
    };

    std::vector <std::string> files_normaly = 
    {
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS3/x1.csv", //multivariate higher res
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS3/x2.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS3/x3.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS3/x4.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS3/x5.csv",
        "C:/Users/H/Desktop/cpp/.csv files/MultiVar Data/MDS3/x6.csv"
    };

    // ENTROPY PROCESSING UNIT

    std::vector<SensorData> streams;   // one object per file

    for (const auto& file : files_normal){
        SensorData stream;           
        if(stream.readFile(file)){  
            streams.push_back(stream); //object that has been processed (holds data)
        }
        else{
            std::cout << "Failed to open file" << file;
        }
    }

    



     std::vector<double> alphas = {0.55, 0.6, 0.7, 0.8}; // Increased the span for alphas
     std::vector<int> ks = {2,3,4,8}; // Expanded range for ks       this is good for sensor data

     double best_score = -1.0;
     double best_alpha = 0.0;  //pre set
     int best_k = 0;

     double best_score_hx = -1.0;
     double best_alpha_hx = 0.0;  //pre set
     int best_k_hx = 0;

     double best_mean_r = 0.0;
     double best_mean_rs = 0.0;

     double bestscoreraw = 0.0;
     double bestscoreenttropy = 0.0;


     std::vector<std::vector<double>> best_normal_raw;
     std::vector<std::vector<double>> best_normal_entropy;

     std::vector<std::vector<std::vector<double>>> cluster_check_raw;
     std::vector<std::vector<std::vector<double>>> cluster_check_entropy;
     //put these in a vector for loop acces
     std::vector<std::vector<double>> centroid_check_raw;
     std::vector<std::vector<double>> centroid_check_entropy;

     std::vector<double> preservation_ratio_raw;
     std::vector<double> preservation_ratio_entropy;

     bool done = false;

for(int h = 0; h <= 1; h++){

for (double alpha : alphas) {
    std::cout << "Testing alpha = " << alpha << std::endl;

    RawProcessing RawProcessor; //this resets everytime
    OnlineEntropy EntropyProcesser(files_normal.size(), 18, 7); //adjust for optimal entropy 
    std::vector <std::vector<double>> raw_normal;
    std::vector <std::vector<double>> entropy_normal;
    std::vector <std::vector<double>> actual_raw_data;

    std::vector <double> pearsoncorrelations;
    std::vector <double> spearmancorrelations;

    std::vector<std::ifstream> fileStreams;
 
    // Open all files and skip header
    for (const auto& file : files_normal) {
       std::ifstream stream(file);
         if (!stream.is_open()) {
           std::cerr << "Failed to open file: " << file << std::endl;
           continue;
         }
    
    std::string header;
    std::getline(stream, header); // Skip header line
    fileStreams.push_back(std::move(stream));

    }
    
    // Read line-by-line interleaved
    bool stillReading = true;
    while (stillReading) { //this is the val that makes getline keep going

        std::vector<double> row;
        stillReading = false;

        for (auto& stream : fileStreams) { //every stream

           if (!stream.eof()) { //if its NOT ended

            std::string line;  //place line in here

                if (std::getline(stream, line)) { //open the stream, get the line, and place it in line

                  if (line.empty()) continue;  // skip blank lines
                  double val;
                  val = SensorData::myProcessor(line);  //process it (just turn it into a double)
                  row.push_back(val); //push the lines val into a vector
                  stillReading = true; // Keep going if at least one stream has data

                }
            }
        }

        //PROCESSING WINDOW ---------------------------------

        if (!row.empty()){

            std::vector<double> raw_vec = RawProcessor.EMA_Online_Normalized(row, alpha); //optimize alpha and k's for raw first, that also decides for entropy

            if(h == 1 && !raw_vec.empty()){
            std::vector<double> entropy_vec = EntropyProcesser.update(raw_vec); //extra processing
               if(!entropy_vec.empty()){
                  entropy_normal.push_back(entropy_vec);
               }
            }
            // Only push if vectors are also non-empty
            else if(h == 0 && !raw_vec.empty()) { //needs to be else if
               raw_normal.push_back(raw_vec); //normal state of system with this alpha ^
               //preservation_ratio_raw.push_back(sampleVariance(raw_vec) / sampleVariance(row)); 
            }                 
            else {
               std::cerr << "Warning: Skipped pushing empty processed vectors.\n";
            }
        }  


    } // ----------------------------------- 

    //optmize with checking pearson and spearman

    auto transposed_raw_normal = transposeData(raw_normal);
    auto transposed_actual_raw_data = transposeData(actual_raw_data);

    if(transposed_actual_raw_data.size() != transposed_raw_normal.size() || transposed_actual_raw_data[0].size() != transposed_raw_normal[0].size()){
        throw std::invalid_argument("Raw normal and OG data are not the same size");
    }
    
    for(int i = 0; i < transposed_actual_raw_data.size(); i++){ //iter over each column
        double r = pearsonCorrelation(transposed_actual_raw_data[i], transposed_raw_normal[i]); //how well the proportional relationship was preserved across each dimension
        pearsoncorrelations.push_back(r);
        double rs = spearmanCorrelation(transposed_actual_raw_data[i], transposed_raw_normal[i]); //how well the topological structure was preserved across each dimension
        spearmancorrelations.push_back(rs);
        //if n is 14. i get 14 rs's and 14 r's
    }

    double mean_rs = mean(spearmancorrelations);
    double median_rs = median(spearmancorrelations);

    double mean_r = mean(pearsoncorrelations);
    double median_r = median(pearsoncorrelations);

    //PCA here add a weight

    double currentscore_ = score(mean_r, median_r, mean_rs, median_rs);

        if(h == 0){

             std::cout << " PCA score = " << std::endl;

              if (currentscore_ > bestscoreraw) {
                bestscoreraw = currentscore_;
                best_alpha = alpha;          
         
                best_normal_raw = raw_normal;
    
                std::cout << "New Best: alpha = " << best_alpha
                      << ", k = " << best_k
                      << ", score = " << best_score << std::endl;
                }

            std::cout << std::endl;

        }
        else if(h == 1){

             std::cout << " PCA score = " << std::endl;

              if (currentscore_ > best_score_hx) { //i have a seperate score since were doing pca too, pca should define spearman and pearson value? lets set weights
                 best_score_hx = currentscore_;
                 best_alpha_hx = alpha;           
          
            
                 best_normal_entropy = entropy_normal;
            

                 std::cout << "New Best: alpha = " << best_alpha_hx
                      << ", k = " << best_k_hx
                      << ", score = " << best_score_hx << std::endl;
                 }

            std::cout << std::endl;

        }
    





    /*
    for (int k : ks) { //for k's
        std::cout << "  Trying k = " << k << std::endl;

        // Your existing kmeans function should return silhouette score
        if(h == 0){

             auto [clusters, centroids, silhouette_score] = kmeans(raw_normal, k, 200); 
             std::cout << "    Silhouette Score = " << silhouette_score << std::endl;

              if (silhouette_score > best_score) {
                best_score = silhouette_score;
                best_alpha = alpha;
                best_k = k;
            
         
                best_normal_raw = raw_normal;
                cluster_check_raw = clusters;
                centroid_check_raw = centroids;
        
           
                std::cout << "New Best: alpha = " << best_alpha
                      << ", k = " << best_k
                      << ", score = " << best_score << std::endl;
              }

            std::cout << std::endl;

        }
        else if(h == 1){

             auto [clusters, centroids, silhouette_score] = kmeans(entropy_normal, k, 200); 
             std::cout << "    Silhouette Score = " << silhouette_score << ". ST: " << h << std::endl;

              if (silhouette_score > best_score_hx) { 
                 best_score_hx = silhouette_score;
                 best_alpha_hx = alpha;
                 best_k_hx = k;
            
          
            
               best_normal_entropy = entropy_normal;
               cluster_check_entropy = clusters;
               centroid_check_entropy = centroids;
            

            std::cout << "New Best: alpha = " << best_alpha_hx
                      << ", k = " << best_k_hx
                      << ", score = " << best_score_hx << std::endl;
            }

            std::cout << std::endl;

        }
    }
    */

    // Close files after use
    for (auto& fs : fileStreams) {
        if (fs.is_open()) fs.close();
    }


}//for alpha

  done = true;

}

std::cout << "\n Best Parameters Found for RAW:\n";
std::cout << "Alpha: " << best_alpha << ", K: " << best_k << ", Silhouette Score: " << best_score << "\n"; 

  for(auto cluster : cluster_check_raw){
        std::cout << cluster.size() << " ";
    }
    std::cout << std::endl;

if(done){
    std::cout << "\n Best Parameters Found for ENTROPY: " << "\n";
    std::cout << "Alpha: " << best_alpha_hx << ", K: " << best_k_hx << ", Silhouette Score: " << best_score_hx << "\n";

    for(auto cluster : cluster_check_entropy){
        std::cout << cluster.size() << " ";
    }

}
else{
    std::cout << "entropy not processed" << std::endl;
}
    std::cout << std::endl;
    
    vis::write_svg(cluster_check_raw, "C:/Users/H/Desktop/C++ Tryharding/clustering/2/clusters21(kmeansM + pre + umap)raw.svg", 900, 700, 3.0);
    vis::write_svg(cluster_check_entropy, "C:/Users/H/Desktop/C++ Tryharding/clustering/2/clusters21(kmeansM + pre + umap)entropy.svg", 900, 700, 3.0);

    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> covmatrices_raw = CovarianceMatrix(cluster_check_raw, centroid_check_raw, best_normal_raw); 
    std::cout << "covs matrices raw:" << covmatrices_raw.size() << std::endl;

    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> covmatrices_entropy = CovarianceMatrix(cluster_check_entropy, centroid_check_entropy, best_normal_entropy); 
    std::cout << "cov matrices entropy: " << covmatrices_entropy.size() << std::endl;

    int c = 1;
    std::cout << "Sanity check for covmatrices raw"<< std::endl;
    for(auto x : covmatrices_raw){
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(x);
    double log_det = es.eigenvalues().array().log().sum();
    double det = exp(log_det);
    std::cout << "Cov matrix (k =) "<< c << " eigenvalues: " << es.eigenvalues().transpose() << '\n' << "Cov matrix "<< c <<" determinant: " <<det << std::endl;  //larger det is more spread out, each cluster
    std::cout << std::endl;
    c++;
    }

    std::cout<< std::endl;  //larger det is more spread out, each cluster

    c = 1;
    std::cout << "Sanity check for covmatrices entropy"<< std::endl;
    for(auto x : covmatrices_entropy){
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(x);
    double log_det = es.eigenvalues().array().log().sum();
    double det = exp(log_det);
    std::cout << "Cov matrix (k =)"<< c << " eigenvalues: " << es.eigenvalues().transpose() << '\n' <<"Cov matrix "<< c <<" determinant: " << det << std::endl;  //larger det is more spread out, each cluster
    std::cout << std::endl;
    c++;
    }
    c=1;

    std::cout << "umap 2d output: " << std::endl;
for(auto vector : best_normal_raw){
    std::cout << "(";
    for(int i = 0; i < vector.size(); i++){
        std::cout << vector[i];
        if(i == 0){
            std::cout << ", ";
        }
    }
    std::cout << ")";
    std::cout << std::endl;
}


/*    
    //alignement check for microclusters raw
     std::vector <int> validsizes;
     validsizes.resize(cluster_check_raw.size()); //reserve only does memory

     int clustercount = 0;
     for (auto& cluster : cluster_check_raw) {
        if (cluster.size() < 10) { // example threshold
            validsizes[clustercount] = 1;
        }
        else{
            validsizes[clustercount] = 0;
        }
        clustercount++;
    }

    clustercount = 0;
    int minicentroidcount = 0;
    
  for(auto minicentroid : centroid_check_raw){
     if(validsizes[minicentroidcount] == 1){ //IF WE HIT A MINI CENTROID 
          for (int i = 0; i < centroid_check_raw.size(); i++) { //COMPARE EACH CENTROID TO THAT MINI CENTROID

           double csim = cosineSimilarityCPP(minicentroid, centroid_check_raw[i]); //compare to each other cluster
           double mdist = mahalanobisDistanceSingle(toEigenVector(minicentroid), toEigenVector(centroid_check_raw[i]), covmatrices_raw[i]);

           std::cout << "Mini centroid "<< minicentroidcount<<" compared to centroid "<< clustercount << ". CosSim: " << csim << ". Mdist: "<< mdist << std::endl;

           clustercount++;
           }
         clustercount = 0;
       }
       minicentroidcount++;
    }

     std::cout<<std::endl;
    for (auto& size : validsizes) {
             std::cout << size << " ";  
      }
    std::cout <<std::endl;
*///i think we can mark zones instead of mini clusters
                  
    //FOR OUTLIER LOGIC

    std::vector <std::string> files_live =  
    {
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/temperature.csv", //temp
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/humidity.csv",   //hum
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/voltage.csv",    //v
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/current.csv",    //curr
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/power.csv",      //w
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/vibration.csv",   //freqs
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/stable.csv"       //stable
    };

    std::vector <std::string> files_livex =  
    {
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/MultiVar Data/MDS1/x1.csv", //multivariate higher res
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/MultiVar Data/MDS1/x2.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/MultiVar Data/MDS1/x3.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/MultiVar Data/MDS1/x4.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/MultiVar Data/MDS1/x5.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/MultiVar Data/MDS1/x6.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/MultiVar Data/MDS1/x7.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/MultiVar Data/MDS1/x8.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/MultiVar Data/MDS1/x9.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/MultiVar Data/MDS1/x10.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/MultiVar Data/MDS1/x11.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/MultiVar Data/MDS1/x12.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/MultiVar Data/MDS1/x13.csv",
        "C:/Users/H/Desktop/C++ Tryharding/.csv files/MultiVar Data/MDS1/y.csv" 
    };

    

    //std::vector<std::vector<double>> newpoints = generatePoints(500, 7); //new entropy values
                                                            //amount of vectors, dimensions //these are still reconstruction errors tho. feed in the actual entropy
   

    RawProcessing RawProcessorX; //this resets everytime its called
    OnlineEntropy EntropyProcesserX(6, 18, 7);

    std::vector<std::ifstream> fileStreams;
 
    // Open all files and skip header
    for (const auto& file : files_live) {
       std::ifstream stream(file);
         if (!stream.is_open()) {
           std::cerr << "Failed to open file: " << file << std::endl;
           continue;
         }
    
    std::string header;
    std::getline(stream, header); // Skip header line
    fileStreams.push_back(std::move(stream));

    }

    static int vector_stack = 0;
    static int iteration = 0;

    std::vector<std::vector<std::vector<PolarCoordinates>>> polarSignalDataMulti;

    Regime trajectoryAnalysis;

    // Read line-by-line interleaved
    bool stillReading = true;
    while (stillReading) { //this is the val that makes getline keep going
        std::vector<double> row;
        stillReading = false;

        for (auto& stream : fileStreams) { //every stream

           if (!stream.eof()) { //if its NOT ended

            std::string line;  //place line in here

                if (std::getline(stream, line)) { //open the stream, get the line, and place it in line

                if (line.empty()) continue;  // skip blank lines
                double val;
                val = SensorData::myProcessor(line);  //process it (just turn it into a double)
                row.push_back(val); //push the lines val into a vector
                stillReading = true; // Keep going if at least one stream has data

                }
            }
        }
        
        //LIVE PROCESSING WINDOW ------------------------------------------------------------------------------------- (potential kmeans hybrid integration)

        if (!row.empty()){
            //g++ main.cpp RDS.cpp timechunking.cpp entropywindow.cpp outlierlogic.cpp kmeans.cpp rawprocessing.cpp patternrecog.cpp utils.cpp -o myprogram

            std::vector<double> raw_vec = RawProcessorX.EMA_Online_Normalized(row, 0.60); //optimize alpha and k's for raw first, that also decides for entropy          
            std::vector<double> entropy_vec = EntropyProcesserX.update(raw_vec);

            if(checkForNaN(raw_vec) || checkForNaN(entropy_vec)){
                std::cout << "Raw Vec or Entropy Vec Empty/Full of Zero's or NaN's!" << std::endl;
                for(int i = 0; i < raw_vec.size(); i++){
                    std::cout << "R[" << i <<"]: " << raw_vec[i] << "  E[" << i <<"]: " << entropy_vec[i] << std::endl;
                }
                std::cout << "Index: " << iteration << std::endl;
            }

            //OUTLIER LOGIC

            std::vector<double> cossims_raw;
            std::vector<double> mahalanobisdistances_raw; 

            std::vector<double> cossims_entropy;
            std::vector<double> mahalanobisdistances_entropy; //its all online now

            
            mahalanobisdistances_raw = MahalanobisDistMulti(raw_vec, centroid_check_raw, covmatrices_raw); //takes each point cross references it to the centroid. needs covmatrix
            cossims_raw = getCosSim(centroid_check_raw, raw_vec);   //each vector is the size of the amount of clusters. and gives the info

            mahalanobisdistances_entropy = MahalanobisDistMulti(entropy_vec, centroid_check_entropy, covmatrices_entropy); //covmatrices already pre calculated
            cossims_entropy = getCosSim(centroid_check_entropy, entropy_vec);

            //everytime get CosSim is called it compares the incoming vec to all clusters evertime

            //logTransformAndScale(mahalanobisdistances_raw, 0.0, 100.0); //100 is worst, log smoothed heavy shit out dont worry
            //logTransformAndScale(mahalanobisdistances_entropy, 0.0, 100.0);   ONLY USE DOWNSTREAM

            std::vector<PolarCoordinates> polarData_raw = makePolar(mahalanobisdistances_raw, cossims_raw);
            std::vector<PolarCoordinates> polarData_entropy = makePolar(mahalanobisdistances_entropy, cossims_entropy); //each object is data from a cluster

            

            
           
            std::vector<std::vector<PolarCoordinates>> polarSignalData = {polarData_raw, polarData_entropy}; //each row is the vectors metrics across all clusters, both entropy and raw

            polarSignalDataMulti.push_back(polarSignalData);

        
            vector_stack++; //reset this each if
            

            if(vector_stack == 5){
                
                std::cout << "Trajectory Window of " << vector_stack << " vectors." << std::endl;
                print3dPolar(polarSignalDataMulti); //Multi = multiple points

                for(auto vector : polarSignalDataMulti){
                   
                  trajectoryAnalysis.stateAnalysisPerCluster2(vector);

                }

                polarSignalDataMulti.clear();
            }

            //should be 2 rows and k columns

            //T and N are for the explanations

            //take each polar metric, check the regime and if its severe. see regime x severity grid
            // for explanation on orthogonal see the labels too

            



                




























         
           iteration++;

        } // ----------------------------------------------------------------------------------------------------

    }


































   

// AUXILIARY UNIT THAT SELF HEALS THE SYSTEM
/*  --------------------------------------------------------------------------------------------------------
    SensorData epochs;
    epochs.readFile("C:/Users/H/Desktop/C++ Tryharding/.csv files/RE/epochs.csv");   //for aux baseline drift checkin

    std::vector<std::vector<double>> stressvectors;  //are vectors (points)
    FIX
    while(epochs.hasNext()){
        std::vector <double> x = FourierExpansion(epochs.getNext(), 2, 3); //expands it so kmeans has multidimensional data
        stressvectors.push_back(x);
    }



    //will return a d * 2 dimensional vector of the r_error cuz im sin + cos mapping

    std::cout << "Encoded vectors:\n" << stressvectors.size() << std::endl;


    for (auto vector : stressvectors) { 
        for(auto val : vector){

            std::cout << val << " ";
            std::cout << std::endl;

        }
        std::cout << std::endl;
        std::cout << std::endl;
    }
*///--------------------------------------------------------------------------------------------------------
   







    // PULL THIS INTO LIVE PROCESSING WINDOW
    
    std::cout << "Cosine similarity/Mahalanobis Dist to all clusters from given poins:";    
    std::cout << std::endl;

    int count = 1;

    std::vector<std::vector<double>> cossims;
    std::vector<std::vector<double>> mahalanobisdistances;
/* XXXXXXXXXXXXXXXXXX
    for(auto point : newpoints){ 
           std::vector<double> mahalabonisdist_buffer = MahalanobisDistMulti(point, centroid_check_raw, covmatrices); //takes each point cross references it to the centroid. needs covmatrix
           std::vector<double> cossims_buffer = getCosSim(centroid_check_raw, point);   //each vector is the size of the amount of clusters. and gives the info

           mahalanobisdistances.push_back(mahalabonisdist_buffer);
           cossims.push_back(cossims_buffer);  //each row in this vector is the data of each point. each point has data towards each cluster
    }
*/
    //look at how how mahalanobis distances look like

    //do a for loop to check

    //BOUNDED + LOG. SO THEYRE COMPARABLE with eachother.
    //logTransformAndScale(mahalanobisdistances, 0.0, 100.0); //10 is worst, log smoothed it out dont worry XXXXXXXXXXX
//DELETE COMMENTS -----------------------------------------------
/*
    for(auto vector : mahalanobisdistances){
        for(auto val : vector){
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

*/
   

/*
    for(auto row : mahalanobisdistances){
        for(int x = 0; x < row.size(); x++){
        std::cout << "point " << count << " Mdist to cluster "<< (x + 1) << ": " << row[x] << " " << std::endl;
        }
       count++;
    }
    std::cout << std::endl;
    count = 1;
    for(auto row: cossims ){
        for(int x = 0; x < row.size(); x++){
        std::cout<<"point " << count << " cossim to cluster "<< (x + 1) <<": " << row[x] << " " << std::endl;
        }
        count++;
    }
*/



    

   
    //this is purely to get the r,theta layout


    std::vector<std::vector<double>> polarRows;
    std::vector <double> polarBuffer; //dont forget to set clean after each iteration, move does it
    //each row = [r1, theta1, r2, theta2]

    count = 1;

    for(int i = 0; i < mahalanobisdistances.size(); i++){ //going over each point
        for(int j = 0; j < centroid_check_raw.size(); j++){  //going over each cluster

        polarBuffer.push_back(mahalanobisdistances[i][j]); //each point, and each cluster data
        polarBuffer.push_back(cossims[i][j]);
        //so polar buffer orders the data like this per point [r, cossim, r, cossim... rn, cossimn] so each row is this
        }
        polarRows.push_back(std::move(polarBuffer)); //each row is a point.  // one full row done → stash it
        polarBuffer.clear(); //one row in polarRow = [r, cossim, r, cossim], twice cuz we have to centroids, clusters
    }

    std::vector<std::vector<std::complex<double>>> complex_data;
    std::vector<double> complex_buffer;

/*
    //point > cluster id > data
    std::cout << "points not sorted" <<std::endl;
    std::cout << std::endl;
    for(auto vec : polarRows){  //each vec is a point. each val is the [mdist1, cossim1, mdist2, cosim2]
        for(int i = 0; i < vec.size(); i++){

            std::cout << vec[i] << " ";
        }
       std::cout << std::endl;
    }
*/
   

   std::cout << "commencing sort. structure is now 2d vector where each row is a point, each column is k cluster object containing polar metrics" <<std::endl;
   std::vector<std::vector<PolarCoordinates>> pointssorted;
   pointssorted.reserve(polarRows.size()); //points sorted contains same layout as polarrows but now tagged PER cluster, row = point, columns = what cluster

    for (auto &vec : polarRows) { //each point
    std::vector<PolarCoordinates> polardatasorted(centroid_check_raw.size());

    // Process pairs [r, θ]
        for (size_t cluster = 0; cluster < centroid_check_raw.size(); ++cluster) { //each cluster
        double r     = vec[2*cluster];  //even 
        double theta = std::acos(std::clamp(vec[2*cluster + 1], -1.0, 1.0)); //uneven + safety

        polardatasorted[cluster].r      = r;
        polardatasorted[cluster].cos_t  = std::cos(theta); //embed
        polardatasorted[cluster].sin_t  = std::sin(theta);
        //each vector is an object that stores the data per cluster
        }
           if(!polardatasorted.empty()){
                pointssorted.push_back(std::move(polardatasorted)); //faster and cleans it for reuse
           }
    }

    // so each row of points sorted should contain each point. and within those rows we have another vector that stores the polar stress metrics per clusters (can be any size), to choose we just increment

    //now we need to make a function that takes in: function(std::vector <double> cosine_similarity, std::vector <double> mahalanobis distances);
   
    // BUILD:

    // 1: Polar stress trajectory with sliding windows (the path/shape how it evolves)
    // 2: If trajectory looks like anomaly type a b or c, get spectral fingerprint. save it and/or cross reference it

    /*
    Prototype: Log the raw trajectory (sequence of r, θ over sliding window).

    Later: Derive spectral fingerprints from those trajectories for scalable storage/search.
    */

    /*

    Workflow:

    Detect anomaly → store trajectory.

    Human labels anomaly and therefore the trajectory.

    Encode trajectory into spectral fingerprint → save that fingerprint as the reusable signature.
    
    */

    //IMPORTANT: θ = arccos(cos_sim)
    //Now (r, θ) lives in legit polar space, and trajectories make geometric sense. (60 degrees is interpretable)

    /*

    Best practice (for embeddings & stability)

    Don’t stop at arccos. Use sin/cos encoding of θ:

θ = arccos(cos_sim)
embed = (cos θ, sin θ)


That kills discontinuity at wrap-around, keeps it smooth for Fourier/NNs.

So final representation per timestep:

(r, cos θ, sin θ)


instead of (r, θ) or (r, cos_sim).
    
    
    */

    //so do this: θ = arccos(cos_sim) then: cos(theta) and sin(theta) for proper geometrical continuity
    // so timesteps look like: [r, cos(theta), sin(theta)]

    /*

    Using just cos_sim: anomaly looks like a boring linear fall.

Using (cos θ, sin θ): anomaly looks like a rotation on a circle, which is exactly what the angle is supposed to mean.

When you later feed trajectories into Fourier/autoencoders, the system sees real rotational structure, not a warped squashed 1D line.

So the payoff:

cos_sim → misleading, collapses angle info.

θ → correct but brittle at wraparound.

(cos θ, sin θ) → smooth, faithful, no wrap issues.
    s
    */

  
/*

     SensorData polarcoord;//this is for csv reading
     polarcoord.readFile("C:/Users/H/Desktop/C++ Tryharding/.csv files/synthetic data/polarcoordinates.csv");
     std::vector <std::vector <PolarCoordinates>> polarcoords = polar_csv_slidingwindow_fixed(polarcoord, 10, 16);
     //multiple points = trajectory (window)
*/
/*
     for(auto window : polarcoords){
        for(auto p : window){
            std::cout << "radius: " << p.r << " cos: "<< p.cos_t << " sin: " << p.sin_t << std::endl;
        }
        std::cout << std::endl;
     }  
*/    
      // [window][points][cluster]

       std::cout << "now i have to transform this into re + im space so the fft understands the trajectories. but first i need to do a sliding window over this" <<std::endl;

       std::cout << std::endl;

       int cluster_id = 2; // 2 clusters is 0 and 1 idx

       std::vector<PolarCoordinates> clusterTrajectory_entropy; //holds polar stress metrics of points in one cluster
       for (auto &row : pointssorted) {  //every point.
        clusterTrajectory_entropy.push_back(row[cluster_id]); // one cluster across all points
        //pushes back the first or second or third object with data. each object is a cluster we can acces with p.r p.cos etc
       } 

       auto windows = slidingWindows(clusterTrajectory_entropy, 16, 8); //data, windowsize, stride. good for j attribution
      //data looks like this [p1,p2,p3,p4...pn]. each p we can acces for its corresponding polar data
     


      std::vector<std::vector<std::complex<double>>> fftWindows; //points per cluster. so we have the metrics, of certain clusters now

      for (auto &win : windows) { //each window in windows is a list of points with polar data

         std::vector<std::complex<double>> buffer;
         buffer.reserve(win.size());

            if(win.size() < 16){
               buffer.resize(16, std::complex<double>(0.0, 0.0)); //zero padding
            }
            else if(win.size() > 16){
               win.resize(16); // Truncate to 16 elements
            }

            for (auto &p : win) {
                 buffer.emplace_back(p.r * p.cos_t, p.r * p.sin_t); //for each point in the window we make it a cartesian coordinate
                 // buffer.emplace_back(/*p.r8 */p.cos_t,/*p.r */p.sin_t); //this for unit cirlce coordinates (cos is x, sin is y, without severity r)
            }

          fftWindows.push_back(std::move(buffer));
          //same windows as before but now in cartesian so fft understands it
          //example window: [10.76 + 23.32i... re + im] window size
      }
//DELETE COMMENTS -----------------------------------------------
/*
       std::cout << "fft trajectoy window(s): " << std::endl;

       for (auto vector : fftWindows){
          for(auto y : vector){ //every window in complex data
       
             std::cout << "("<< y.real() << ", "<< y.imag() << ")" << std::endl; //y needs to be i (complex plane)
       
          }
           std::cout<<std::endl;
        }
*/
       count = 0;

       //so these windows are a bunch of metrics from different points (evolving in cartesian space) we can use them to see whats acting up

    
       //HOW TO READ THE DATA, EMBED THEM, PREPARE FOR FFT ETC
















    std::cout << "Starting FFT sanity test" << std::endl;

    // Complex FFT / IFFT sanity check
    vcd a = { {1,0}, {2,0}, {3,0}, {4,0} };  //x[a]1  x[a]2 .. x[a]4
    fft(a, false); //time domain to freqs doman
    fft(a, true);  //frequency domain back to time domain
    for (auto &x : a) std::cout << x.real() << " " << x.imag() << "\n";

    // when we see 1234 in the output. thats the reconstructed vector

    // the zeros after = sanity check: “hey, your supposedly real signal didn’t accidentally come back with a big imaginary ghost.”
    /*

    For most “spectral fingerprint” use cases (music analysis, sensor fingerprinting, etc.), magnitude is what you care about. 
    Phase only matters if you want to rebuild the signal.

    Those imaginary “zeros” are basically the checksum. If the inverse FFT math stitched your cosines and sines back together correctly, the imaginary part vanishes. If you screwed up the algorithm, you’d see big, ugly imaginary numbers polluting the output.

        So:

          Small imag ≈ 0 → everything added up properly, round-trip success.

          Large imag → you botched the FFT/IFFT implementation.

          That’s the only reason we print them: to prove the machine didn’t drop a term when reconstructing the original signal.
    
    */

    //spectral fingerprint

    std::cout << "Starting real FFT for complex data and geting spectral fingerprint for " << fftWindows.size() << " windows." << std::endl;
    std::vector<std::vector<double>> spectralfingerprints;

    std::cout << std::endl;

    for(auto &window : fftWindows){// 60 vectors

        fft(window, false); //forward transform if a window with complex points [re + im] or [re + 0]

        //std::cout << "window " << count << std::endl;

        std::vector <double> mags(window.size());
 
        for (size_t k = 0; k < window.size(); ++k) {
            mags[k] = std::abs(window[k]);
            //std::cout << "bin " << k << ": " << std::abs(window[k]) << "\n";  
            //takes b (that was changed, and takes the absolute value of each number)
        }

        double norm = 0;
        for(auto m : mags){
            norm += m*m;
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
        for (double &m : mags) m /= norm;
        }

        count++;
        
        //std::cout << std::endl;

        spectralfingerprints.push_back(std::move(mags)); //data base storage is now consistent

    }

    count = 1;

    //vcd b = { {10.0}, {20.0}, {30.0}, {40.0} };  // aka: [10 + 0i, 20 + 0i, etc...] . what the ffts expects

    
    //L2 Normalize and discard duplicates for spectral fingerprints

    //in my polar batch, each sample is truly complex

    /*

    So can FFT handle 2 features?

    Yes, because FFT input is literally defined as a vector of complex numbers:

       [x0, x1, x2, …, xN-1]


       and each xn is a pair of floats: (Re, Im).

    Real-only signal = second float is always 0.

    Polar-derived signal = second float is nonzero.
    */

    //so my vector is: [0.700+0.714i, 1.049+1.072i, 1.398+1.430i, … ] perfectly valid

    /*
    Re = radius * cos θ
    Im = radius * sin θ
    */

    //each window is a trajectory of polar coordinates






//DELETE COMMENTS -----------------------------------------------
/*
    for(auto vector : spectralfingerprints){
        std::cout << "spectralfingerprint " << count << ":  ";
        for(auto val : vector){
            std::cout << val << "  ";
        }
        count++;
        std::cout << std::endl;
    }
*/
    


    //i need to sort it to a 2d vector where each row is the timestep

    

       

       
        
    



    

    

    
    

    



   


     

     

     

    






















     
     








    

    return 0;
}

std::vector <double> getCosSim(const std::vector<std::vector<double>>& centroids, std::vector<double> &point){

    std::vector <double> cossims;

     for(auto c : centroids){

        if(c.size() != point.size()){
            std::cout << "dimension mismatch!" << std::endl;
        }

        Eigen::Map<Eigen::VectorXd> eigen_vector_A(c.data(), c.size());                    //convert 2 vectors in eigen vectors
        Eigen::Map<Eigen::VectorXd> eigen_vector_B(point.data(), point.size());            //ofc you got one, it was the same vector
        cossims.push_back(cosineSimilarity(eigen_vector_A, eigen_vector_B));
    
    }

    return cossims;

}

void dumpToCSV(const std::vector<Entropy>& entropies, std::vector<TimeChunking>& stream_chunks, const std::string& filename, const std::vector <std::string> &streams) {

    // Open the output file in write mode
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    // Write the header row (representing streams as features)
    file << "Row";
    for(int i = 1; i <= streams.size(); i++){
        file << ",stream" << i;  //use commas, csv reader chokes
    }
    file << '\n';

    // Loop through the entropy values (columns)
    for (int x = 0; x < entropies[0].getEntropies().size(); x++) {  // Loop through each time step
        file << x;
        // Loop through the streams (rows)
        for (int y = 0; y < entropies.size(); y++) {
            file << "," << entropies[y].getEntropies()[x];  // Write entropy value for the stream at this time step
        }

        file << std::endl;  // Move to the next line after printing a time step
    }

    file.close();  // Close the file after writing
    std::cout << "Data dumped to " << filename << std::endl;

}

//file << stream_chunks[0].getTime(x);// << "ms";  // First column is the time step, numerical for now   //IT WAS READING AN EMPTY VECTOR. blew the whole fucking program up  bro

// cluster_names is optional, e.g. {"cluster1","cluster2",...}
// rows = list of points, each point has (r1,theta1,r2,theta2,...)
void dumpPolarCSV(const std::string &filename,
                  const std::vector<std::vector<double>> &rows,
                  int numClusters,
                  const std::vector<std::string> &cluster_names)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // Header row
    if (cluster_names.empty()) {
        for (int c = 0; c < numClusters; ++c) {
            file << "r_cluster" << (c+1) << ",theta_cluster" << (c+1);
            if (c != numClusters-1) file << ",";
        }
    } else {
        for (int c = 0; c < numClusters; ++c) {
            file << "r_" << cluster_names[c] << ",theta_" << cluster_names[c];
            if (c != numClusters-1) file << ",";
        }
    }
    file << "\n";

    // Each row of data
    for (const auto &point : rows) {
        for (size_t i = 0; i < point.size(); ++i) {
            file << point[i];
            if (i != point.size()-1) file << ",";
        }
        file << "\n";
    }

    file.close();
}

std::vector<std::vector<std::complex<double>>> readPolarCSV(const std::string &filepath, int numClusters) {
    std::ifstream infile(filepath);
    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    std::string line;
    std::getline(infile, line); // skip header

    // one trajectory per cluster
    std::vector<std::vector<std::complex<double>>> clusterTrajectories(numClusters);

    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> values;

        while (std::getline(ss, cell, ',')) {
            values.push_back(std::stod(cell));
        }

        if (values.size() != 2 * numClusters) {
            throw std::runtime_error("Row length does not match 2*numClusters");
        }

        for (int j = 0; j < numClusters; j++) {
            double r = values[2*j];
            double theta = values[2*j+1];
            double re = r * std::cos(theta);
            double im = r * std::sin(theta);
            clusterTrajectories[j].emplace_back(re, im);
        }
    }

    return clusterTrajectories;
}

std::vector<std::vector<PolarCoordinates>> slidingWindows(const std::vector<PolarCoordinates> &data, size_t windowSize, size_t stride){

    //data is a vector that holds a bunch of points with polar data.

    std::vector<std::vector<PolarCoordinates>> windows; //holds windows [window][data (metric of clusters)]
    if (windowSize == 0 || data.size() < windowSize) return windows;

    for (size_t start = 0; start + windowSize <= data.size(); start += stride) {
        windows.emplace_back(data.begin() + start, data.begin() + start + windowSize);
    }
    return windows; // slid window over all points/objects in data, each row is the window

}

std::vector<std::vector<double>> generatePoints(int numVectors, int vectorSize) {
    std::vector<std::vector<double>> points;

    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<double> noiseDist(-0.005, 0.005); // subtle noise
    std::uniform_real_distribution<double> stepDist(0.001, 0.01);    // very small steps

    double value = 0.0;           // persistent value across vectors
    bool trendingUp = true;       // start trending up
    int trendLength = 50;         // how long to keep a trend direction
    int trendCounter = 0;         // track steps in current trend

    for (int i = 0; i < numVectors; ++i) {
        std::vector<double> newVector;

        for (int j = 0; j < vectorSize; ++j) {
            double step = stepDist(rng);
            double noise = noiseDist(rng);

            if (trendingUp) {
                value += step + noise;
            } else {
                value -= step + noise;
            }

            // Clamp between -1 and 1
            value = std::clamp(value, -1.0, 1.0);

            newVector.push_back(value);
        }

        points.push_back(newVector);

        // Update trend counter and possibly reverse direction
        trendCounter++;
        if (trendCounter >= trendLength) {
            trendingUp = !trendingUp;              // flip trend direction
            trendCounter = 0;                      // reset counter
            trendLength = rng() % 40 + 40;         // randomize next trend length (40–80 vectors)
        }
    }

    return points;
}

std::vector<std::vector<double>> transposeEntropies(const std::vector<Entropy>& entropies) {
    std::vector<std::vector<double>> result;

    if (entropies.empty()) return result;

    size_t numStreams = entropies.size();
    size_t numWindows = entropies[0].getEntropies().size();

    result.resize(numWindows, std::vector<double>(numStreams));

    for (size_t streamIdx = 0; streamIdx < numStreams; ++streamIdx) {
        const std::vector<float>& streamEnts = entropies[streamIdx].getEntropies();

        // Optional: Add bounds checking if streams could differ in size
        for (size_t windowIdx = 0; windowIdx < numWindows; ++windowIdx) {
            result[windowIdx][streamIdx] = streamEnts[windowIdx];
        }
    }

    return result;
}

std::vector<PolarCoordinates> makePolar(std::vector <double> mdist, std::vector<double> csims){ //(R, THETA)

    if (mdist.size() != csims.size()) {
        std::cerr << "[ERROR] makePolar: mdist and csims size mismatch!" << std::endl;
        return {};
    }

    std::vector<PolarCoordinates> polarData; //per CLUSTER
    polarData.resize(csims.size()); //amount of clusters, should be the same as mdist.  

     for(int i = 0; i < csims.size(); i++){

        double r     = mdist[i];   
        double theta = std::clamp(csims[i], -1.0, 1.0); //uneven + safety  + I DELETED ACOS

        polarData[i].r      = r;
        polarData[i].cos_t  = std::cos(theta); //embed to cartesian
        polarData[i].sin_t  = std::sin(theta);
       
     }

   return polarData; //result is k*2 vector, or mdist/csisms * 2 size
}

void logTransformAndScale(std::vector<double>& mahalanobisdistances, double minRange, double maxRange) {
    // Step 1: Log transform all distances first
    for (auto& dist : mahalanobisdistances) {
        dist = std::log(dist + 1);
    }

    // Step 2: Find min and max in log domain
    double globalMin = *std::min_element(mahalanobisdistances.begin(), mahalanobisdistances.end());
    double globalMax = *std::max_element(mahalanobisdistances.begin(), mahalanobisdistances.end());

    // Step 3: Handle edge case if min == max
    if (globalMax == globalMin) {
        for (auto& dist : mahalanobisdistances) {
            dist = minRange;  // or (minRange + maxRange)/2
        }
        return;
    }

    // Step 4: Min-max scale log distances to [minRange, maxRange]
    for (auto& dist : mahalanobisdistances) {
        dist = (dist - globalMin) / (globalMax - globalMin) * (maxRange - minRange) + minRange;
    }
}

std::vector<double> FourierExpansion(double x, int d, int repeat_count) {

    double r_error = x;
    double w = 0;

    std::vector <double> features;

    double *freqs;
    freqs = new double[d];

   

    for(int idx = 0; idx < d; idx++){
        int i = idx / repeat_count;
        freqs[idx] = exp2(i); 

        //fill repeated values, int gets rounded. im doing this so multiple dimensions get same frequencies. just to see what happens
        // same as: * (freqs + x)

    }   

    for(int i = 0; i < d; i++){           //INCREMENT OVER FREQEUNCY CHANGES
         w = freqs[i];                    //MAKE 10000 DYNAMIC SO ITS A sFOURIER FEATURE ENCODER. make it an array with freqs
         double sinval = sin(r_error*w);  //hardcoded 10000 = argument becomes to tiny the wave function collapses
         double cosval = cos(r_error*w);  //true fourier not divisions
         features.push_back(sinval);
         features.push_back(cosval);
    }

    delete[] freqs;
    freqs = nullptr;

    return features;  //keep in mind output = d * 2; sin + cos
}

void transposeInPlace(std::vector<std::vector<double>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) {
        matrix.clear();
        return;
    }

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::vector<std::vector<double>> result(cols, std::vector<double>(rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    matrix = std::move(result);  // replace the orignal matrix to T matrix. faster
}

// Function to transpose the 3D vector
void transposePolarSignalData(std::vector<std::vector<std::vector<PolarCoordinates>>>& data) {

    size_t xDim = data.size();
    size_t yDim = data[0].size();
    size_t zDim = data[0][0].size();

    std::vector<std::vector<std::vector<PolarCoordinates>>> transposedData(
        zDim, std::vector<std::vector<PolarCoordinates>>(yDim, std::vector<PolarCoordinates>(xDim))
    );

    for (size_t i = 0; i < xDim; ++i) {
        for (size_t j = 0; j < yDim; ++j) {
            for (size_t k = 0; k < zDim; ++k) {
                transposedData[k][j][i] = data[i][j][k];
            }
        }
    }

}

void print3dPolar(std::vector<std::vector<std::vector<PolarCoordinates>>> data){
    for(auto point : data){ //outer vector stores points          == these 2 dont change, its constant
        std::cout << "Next Point" << std::endl;
        for(auto signaltype : point){  //each point has 2 signals ==
            for(auto metric : signaltype){ //this changes, right now EACH signaltype vector stores metrics of different clusters. but we want metrics of the same cluster
                 std::cout <<"r: "<< metric.r << " cos: " << metric.cos_t << " sin: " << metric.sin_t << " Original Theta: " << atan2(metric.sin_t, metric.cos_t) << std::endl;
            }
             std::cout << std::endl;
        }
          std::cout << std::endl;
       std::cout << std::endl;
     std::cout << std::endl;
    }
     
}

bool checkForNaN(const std::vector<double>& data) {
    for (size_t i = 0; i < data.size(); ++i) {
        if (std::isnan(data[i] || data[i] == 0)) {
            return true; 
        }
    }
    return false;
}

double dotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    double result = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// Function to calculate the magnitude of a vector
double magnitude(const std::vector<double>& vec) {
    double sumOfSquares = 0.0;
    for (size_t i = 0; i < vec.size(); ++i) {
        sumOfSquares += vec[i] * vec[i];
    }
    return std::sqrt(sumOfSquares);
}

// Function to calculate cosine similarity
double cosineSimilarityCPP(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    double dot = dotProduct(vec1, vec2);
    double mag1 = magnitude(vec1);
    double mag2 = magnitude(vec2);
    return dot / (mag1 * mag2);
}

