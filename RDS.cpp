#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "RDS.h"
#include <sstream>


bool SensorData::readFile(const std::string &filepath) {
    
    std::ifstream infile(filepath);

    if (!infile.is_open()) {
        std::cout << "failed to open file" << std::endl;
        return false;
    }

    std::string header;
    std::getline(infile, header); // skip the "temp" header

    double data;
    while (infile >> data) {
        tempdata.push_back(data);
    }

    return !tempdata.empty(); //true if at least one number is read
}

double SensorData::getNext(){
    return tempdata[index++]; //post-increment
}

bool SensorData::hasNext(){
    if(tempdata.size() == index){
        return false;
    }
    else{
        return true;
    }
}

std::vector<double> SensorData::getStreamData(){
    return tempdata;
}


void SensorData::readFileLineByLine(const std::string& filename,void (*processLine)(const std::string&)) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    std::string line;
  
    //Read the rest line by line
    std::getline(file, line);
    processLine(line);
    
}


double SensorData::myProcessor(const std::string& line){ //static member function dont require an object to be called btw

    double val;
   
    try { 
        val = std::stod(line);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << e.what() << '\n';
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range: " << e.what() << '\n';
    }


    return val;
}
/*
double SensorData::getNext(){
    return datavalue; 
}
*/


