#pragma once
#include <iostream>
#include <vector>

class SensorData {

private:
    double datavalue;  // internal storage of all sensor readings       X
    size_t index = 0;           // keeps track of next value to stream              X
    int rowToReadIndex = 1; // Start from 1 (assuming 0 is the header row)
    std::vector <double> tempdata;
    bool invoked = false;
public:
    std::vector<double> getStreamData();             
    bool readFile(const std::string& filepath); // fills 'data' internally   
    
    bool hasNext();         // returns true if index < tempdata.size()          X
    double getNext();       // returns data[index++] safely                     X
    
    void readFileLineByLine(const std::string& filename, void (*processLine)(const std::string&));
    static double myProcessor(const std::string& line);

};