#pragma once
#include "timechunking.h"
#include "RDS.h"
#include <chrono>
#include <vector>



class TimeChunking{

    public:

    TimeChunking(SensorData& stream, int stride, int windowSize);   //initialze private fields

    //void run();

    void run_slidingwindow_discrete();

    const std::vector<std::vector<float>>& getChunks() const;  //were returning by reference

    void print(std::vector<std::vector<float>> chunks);

    float getTime(int);

    std::vector <std::vector <float>> z_score(std::vector<std::vector<float>>&);



    
    private:

    SensorData& stream;
    std::vector<float> currentChunk;  //stores values in this chunk
    int stride;   //initialized by contstructor
    int windowSize;
    std::vector<std::vector<float>> chunks;
    std::vector <float> timer;



};







