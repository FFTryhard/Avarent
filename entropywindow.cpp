#include "entropywindow.h"
#include "timechunking.h"
#include <vector>
#include <algorithm>
#include <cmath>

/*

Before, your “reference” was the window itself: every window got squashed to its own min/max, so differences between windows disappeared. That’s why all your entropies looked the same. You were basically measuring “how flat is this window relative to itself,” which is useless if you want to compare windows.

With global min/max:

Every window is evaluated against the same scale.

Now, a window that spreads values across the global range → high entropy.

A window that’s clumped → low entropy.

Entropy now actually tells you which windows are more unpredictable or variable in a way that makes sense across your dataset.

So yeah, using a global reference gives your entropy meaning.



*/

//this is literally all offline

Entropy::Entropy(std::vector<std::vector<float>>& chunks_z_scored, bool normalize) { 

    double GLOBAL_MIN = 0.0;
    double GLOBAL_MAX = 0.0;

    bool any_value = false;

    for (const auto& window : chunks_z_scored) {
        if (window.empty()) continue;

        float local_min = *std::min_element(window.begin(), window.end());               
        float local_max = *std::max_element(window.begin(), window.end());

        if (!any_value) {
            GLOBAL_MIN = local_min;
            GLOBAL_MAX = local_max;
            any_value = true;
        } else {
            if (local_min < GLOBAL_MIN) GLOBAL_MIN = local_min;
            if (local_max > GLOBAL_MAX) GLOBAL_MAX = local_max;
        }

    }

    if (!any_value) return;

    double range = GLOBAL_MAX - GLOBAL_MIN;

    if (range == 0.0f) {   //if global range is constant, everything is 0
        entropies.assign(chunks_z_scored.size(), 0.0);
        return;
    }

    entropies.reserve(chunks_z_scored.size());  //the vector that holds the entropy values    
    
    for(auto window : chunks_z_scored){

         if(window.empty()){
                entropies.push_back(0.0);
                continue; // first window empty i guess
            }

        int N = window.size();
        int bins = std::max(2, (int)std::ceil(4 * std::sqrt(N)));
        double bin_width = (GLOBAL_MAX - GLOBAL_MIN) / bins;
        
        std::vector<int> counts(bins, 0); //will use to push back entropy

        for(float v : window) { //inner loop, each float in a window
            float scaled_v = v * 1.02f;    //OPTIONAL. ITS AN AMPLIFIER. SLIGHTY BIASED BUT NEGLIGABLE
            int bin_idx = (int)((v - GLOBAL_MIN) / bin_width);
            if(bin_idx < 0) bin_idx = 0;
            if (bin_idx == bins) bin_idx = bins - 1; // max edge fix
            counts[bin_idx]++;  // bin_idx tells us what bin. and counts keep track of the values that fit into each bin

        }

        // Laplace smoothing
        for (int i = 0; i < bins; i++){
            counts[i] += 1;             // add 1 to every bin
        }

        float sum = 0; //thats our sigma
        int occupied_bins = 0;
        for (int c : counts) {
            if(c > 0) occupied_bins++;
            if(c == 0) continue;
            float p = (float)c / (window.size() + bins); // + fake ones  
            if(p == 0)  continue;
            sum = sum + p * log2(1/p); 
        }   

        //compute the entropy for each probabilty. sum the entropy. and thats the entropy for each window

        double max_entropy = (occupied_bins > 1) ? std::log2((double)occupied_bins) : 1.0;

        if (normalize) {
            if (max_entropy > 0.0) entropies.push_back(sum / max_entropy);
            else entropies.push_back(0.0);
        }else {
            entropies.push_back(sum);
        }

    }
    
}

std::vector<float> Entropy::getEntropies() const{
    return entropies;
}






