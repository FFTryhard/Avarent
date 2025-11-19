#include "timechunking.h"
#include "RDS.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>

TimeChunking::TimeChunking(SensorData& streamRef , int stride, int windowSize)  //syncing
    : stream(streamRef), stride(stride), windowSize(windowSize) { }     

void TimeChunking::print(std::vector<std::vector<float>> chunks){
    for(size_t y = 0; y < chunks.size();y++){
        for(size_t x = 0; x < chunks[y].size(); x++){
            std::cout << chunks[y][x] << " ";
        }
        std::cout << "time: " << timer[y] << " ";
        std::cout << std::endl;        
    }
    std::cout << std::endl;
}

const std::vector<std::vector<float>>& TimeChunking::getChunks() const{
    return chunks;
}

std::vector <std::vector <float>> TimeChunking::z_score(std::vector<std::vector<float>>& chunks) {

    float sum = 0; 
    int values = 0;

    std::vector <float> means;

    for(size_t y = 0; y < chunks.size();y++){
        for(size_t x = 0; x < chunks[y].size(); x++){

           sum = sum + chunks[y][x];
           values++;
           int size = chunks[y].size();
           
        }

        float mean = sum / values;

        means.push_back(mean);

        values = 0;
        sum = 0;
    }


    float difference = 0;
    float difsum = 0;
    float variance = 0;

    std::vector <float> stddev;

    for(int y = 0; y < chunks.size(); y++){
        for (int x = 0; x < chunks[y].size(); x++){   

        difference = chunks[y][x] - means[y];
        difsum = difsum + (difference * difference);
       

        }

        variance = difsum / (float)chunks[y].size();

        float standarddeviation;

        if(variance == 0.0f){
            standarddeviation = 1.0f;
        }
        else{
            standarddeviation = sqrt(variance);   
        }   

        stddev.push_back(standarddeviation);

        difsum = 0;
  
    }

    float z_score_val = 0;
    float val = 0;

    std::vector <float> z_score_temp;
    std::vector <std::vector <float>> z_score;

    for(int y = 0; y < chunks.size(); y++){
        for(int x = 0; x < chunks[y].size(); x++){

            val = chunks[y][x];

            z_score_val = (val - means[y]) / stddev[y]; 

            z_score_temp.push_back(z_score_val);

        }
        z_score.push_back(z_score_temp);
        z_score_temp.clear();
    } 
    

  return z_score;
}

void TimeChunking::run_slidingwindow_discrete() {

    currentChunk.clear();  

    while (stream.hasNext()) {
        float val = stream.getNext();
        currentChunk.push_back(val);

        if (currentChunk.size() == windowSize) {
            chunks.push_back(currentChunk);

            // Slide forward
            if (stride < windowSize) {
                currentChunk.erase(currentChunk.begin(), currentChunk.begin() + stride); //erases middle too
            } else {
                currentChunk.clear();
            }
        }
    }

    // Add leftover values at the end
    if (!currentChunk.empty()) {
        chunks.push_back(currentChunk);
    }
}

    










   





