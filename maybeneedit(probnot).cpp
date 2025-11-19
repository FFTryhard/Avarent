#pragma once

     // ENTROPY UNIT (STILL OFFLINE)
    
    std::vector <TimeChunking> stream_chunks;                  
    for(auto& stream : streams){                               
        stream_chunks.push_back(TimeChunking(stream, 5, 30));  
    }
  
    std::vector < std::vector < std::vector <float>>> z_chunked_data;
    for(auto& stream : stream_chunks){ 
        stream.run_slidingwindow_discrete(); // Sliding window [50, 5]
        std::vector<std::vector<float>> chunks_forzscore = stream.getChunks(); 

        stream.z_score(chunks_forzscore);  
        z_chunked_data.push_back(chunks_forzscore);  
    }

    std::vector <Entropy> entropies;  //each entropy object return a vector of floats. each entropy object is a stream
    entropies.reserve(z_chunked_data.size());
    for (auto& z_chunks : z_chunked_data){  
        entropies.emplace_back(Entropy(z_chunks, true));   
    }

    // Output entropy values with timestamps
    //dumpToCSV(entropies, stream_chunks, "C:/Users/H/Desktop/C++ Tryharding/.csv files/data.csv", files);

    std::vector <std::vector<double>> transposedentropies = transposeEntropies(entropies);

-------------------------------------------------------------------------------------------------------