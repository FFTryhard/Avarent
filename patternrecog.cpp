#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "patternrecog.h"
#include "outlierlogic.h"   

    //FIND A WAY TO QUANITFY IN REAL TIME WHAT TO DO, WITH NUMBERS IN A DECISION MATRIX

    //std::vector<std::vector<double>> decision_matrix; //use later
    
    //DATA IS A 2 X N vector. 2 signal types, raw and entropy. n clusters per signal type.
    // so we analyze per point, since its online

    /*

    +1 region: the river’s main flow — I measure it occasionally.

     0 region: the turbulent eddies — I instrument this zone heavily.

    –1 region: the backflow — I record and classify known disasters.


    Aligned = steady state. Acting here is noise. Only capture rare extreme amplitudes.

    Orthogonal = change state. This is where you learn and intervene.

    Opposite = flipped state. You don’t need gradients—you’re post-mortem; use FFT to recognize or log.

    */

    //add debounce and hysteris

void Regime::printLog(const std::string& message) {
    std::cout << message << std::endl;
}

int Regime::classifyRegime(double theta) {
    // Classifying regimes based on cosine similarity (theta)
    if (std::abs(theta) < 0.25) {
        return 0; // Orthogonal (vectors are perpendicular)
    } 
    // Weakly Aligned: When theta is between 0.25 and 0.6 (moderate alignment)
    else if (theta >= 0.25 && theta < 0.6) {
        return 2; // Weakly Aligned (moderately similar vectors)
    } 
    else if (theta > 0.6) {
        return 1; // Aligned (vectors are in the same direction)
    } 
    // Opposite: When theta is between -1 and -0.6
    else if (theta < -0.6) {
        return -1; // Opposite (vectors are in opposite directions)
    } 
    // Weakly Opposite: When theta is between -0.25 and -0.6
    else if (theta >= -0.6 && theta < -0.25) {
        return -2; // Weakly Opposite (moderately opposite vectors)
    }
    // Orthogonal: By default, for values close to zero or near the perpendicular direction
    return 0; // Orthogonal (default case for when theta is between -0.25 and 0.25)
}

int Regime::stateAnalysisPerCluster(std::vector<std::vector<PolarCoordinates>> data) {
    // This will track regime stability over multiple samples (debounce)
    for (int signalTypeIndex = 0; signalTypeIndex < data.size(); ++signalTypeIndex) {
        std::vector<PolarCoordinates>& signalType = data[signalTypeIndex];

        // Sort the signal data based on 'r' values for quantiles
        std::vector<double> rValues;
        for (const auto& clusterMetric : signalType) {
            rValues.push_back(clusterMetric.r);
        }
        std::sort(rValues.begin(), rValues.end());

        // Calculate quantiles
        int n = rValues.size();
        double q33 = rValues[static_cast<int>(n * 0.33)];
        double q66 = rValues[static_cast<int>(n * 0.66)];
        double q90 = rValues[static_cast<int>(n * 0.90)];

        // Process each data point in the cluster
        for (auto& clusterMetric : signalType) {
            double theta = atan2(clusterMetric.sin_t, clusterMetric.cos_t);

            // Classify the regime based on theta
            int newRegime = classifyRegime(theta);

            // Debounce: Only switch regimes if we have stable samples
            if (newRegime != currentRegime) {
                stableSamples = 1;  // Reset debounce counter
            } else {
                stableSamples++;
            }

            // If stable enough (debounce threshold reached), apply state transition
            if (stableSamples >= debounceThreshold) {
                currentRegime = newRegime;
                printLog("Stable in regime " + std::to_string(currentRegime) + ", triggering state transition.");
            }

            // Hysteresis logic: Only update if the change is large enough
            if (std::abs(theta - lastTheta) > hysteresisMargin) {
                currentRegime = newRegime;
            }

            lastTheta = theta;

            // Now apply specific logic based on the current regime
            std::string signalTypeStr = (signalTypeIndex == 0) ? "[RAW]" : "[ENTROPY]";

            // Handle each regime state
            if (currentRegime == 1) {  // Aligned
                if (clusterMetric.r < q33) {
                    printLog("Aligned - Low activity: " + signalTypeStr);
                } else if (clusterMetric.r < q66) {
                    printLog("Aligned - Moderate activity: " + signalTypeStr);
                } else {
                    printLog("Aligned - High activity: " + signalTypeStr);
                }
            } else if (currentRegime == -1) {  // Opposite
                if (clusterMetric.r < q33) {
                    printLog("Opposite - Low activity: " + signalTypeStr);
                } else if (clusterMetric.r < q66) {
                    printLog("Opposite - Moderate activity: " + signalTypeStr);
                } else {
                    printLog("Opposite - High activity: " + signalTypeStr);
                }
            } else if (currentRegime == 0) {  // Orthogonal
                if (clusterMetric.r < q33) {
                    printLog("Orthogonal - Low activity: " + signalTypeStr);
                } else if (clusterMetric.r < q66) {
                    printLog("Orthogonal - Moderate activity: " + signalTypeStr);
                } else {
                    printLog("Orthogonal - High activity: " + signalTypeStr);
                }
            } else if (currentRegime == 2) {  // Weakly Aligned
                if (clusterMetric.r < q33) {
                    printLog("Weakly Aligned - Low activity: " + signalTypeStr);
                } else if (clusterMetric.r < q66) {
                    printLog("Weakly Aligned - Moderate activity: " + signalTypeStr);
                } else {
                    printLog("Weakly Aligned - High activity: " + signalTypeStr);
                }
            } else if (currentRegime == -2) {  // Weakly Opposite
                if (clusterMetric.r < q33) {
                    printLog("Weakly Opposite - Low activity: " + signalTypeStr);
                } else if (clusterMetric.r < q66) {
                    printLog("Weakly Opposite - Moderate activity: " + signalTypeStr);
                } else {
                    printLog("Weakly Opposite - High activity: " + signalTypeStr);  
                }
            }
        }
    }
    return 0;
}

//Nice, so right now we output how each cluster responds to the input vector (right now i have 10 vectors as input)
//Lets print a trajectory (look at one cluster)

int Regime::stateAnalysisPerCluster2(std::vector<std::vector<PolarCoordinates>> data) {
    // Store results for printing later in a table format
    std::vector<std::vector<std::string>> outputData;  // Rows will represent activity levels
    std::vector<std::string> headers;  // Column headers: cluster1(raw), cluster1(entropy), etc.

    // This will track regime stability over multiple samples (debounce)
    int clusterIndex = 1;  // Start cluster index at 1
    
    // Loop through each signal type (raw and entropy)
    for (int signalTypeIndex = 0; signalTypeIndex < data.size(); ++signalTypeIndex) {
        std::vector<PolarCoordinates>& signalType = data[signalTypeIndex];

        // Sort the signal data based on 'r' values for quantiles
        std::vector<double> rValues;
        for (const auto& clusterMetric : signalType) {
            rValues.push_back(clusterMetric.r);
        }
        std::sort(rValues.begin(), rValues.end());

        // Calculate quantiles
        int n = rValues.size();
        double q33 = rValues[static_cast<int>(n * 0.33)];
        double q66 = rValues[static_cast<int>(n * 0.66)];
        double q90 = rValues[static_cast<int>(n * 0.90)];

        // Store the activity for this signal type (raw/entropy) in the row
        std::vector<std::string> rowData;

        // Process each cluster in the signal type
        for (auto& clusterMetric : signalType) {
            double theta = atan2(clusterMetric.sin_t, clusterMetric.cos_t);

            // Classify the regime based on theta
            int newRegime = classifyRegime(theta);

            // Debounce: Only switch regimes if we have stable samples
            if (newRegime != currentRegime) {
                stableSamples = 1;  // Reset debounce counter
            } else {
                stableSamples++;
            }

            // If stable enough (debounce threshold reached), apply state transition
            if (stableSamples >= debounceThreshold) {
                currentRegime = newRegime;
                printLog("Stable in regime " + std::to_string(currentRegime) + ", triggering state transition.");
            }

            // Hysteresis logic: Only update if the change is large enough
            if (std::abs(theta - lastTheta) > hysteresisMargin) {
                currentRegime = newRegime;
            }

            lastTheta = theta;

            // Handle each regime state
            std::string signalTypeStr = (signalTypeIndex == 0) ? "[RAW]" : "[ENTROPY]";

            // ASSIGNING VECTOR ACTIVITY LEVEL TO EACH CLUSTER METRIC

            std::string activityLevel;

            if (currentRegime == 1) {  // Aligned
                if (clusterMetric.r < q33) {
                    activityLevel = "A - Low";
                } else if (clusterMetric.r < q66) {
                    activityLevel = "A - Moderate";
                } else {
                    activityLevel = "A - High";
                }
            } else if (currentRegime == -1) {  // Opposite
                if (clusterMetric.r < q33) {
                    activityLevel = "OP - Low";
                } else if (clusterMetric.r < q66) {
                    activityLevel = "OP - Moderate";
                } else {
                    activityLevel = "OP - High";
                }
            } else if (currentRegime == 0) {  // Orthogonal
                if (clusterMetric.r < q33) {
                    activityLevel = "O - Low";
                } else if (clusterMetric.r < q66) {
                    activityLevel = "O - Moderate";
                } else {
                    activityLevel = "O - High";
                }
            } else if (currentRegime == 2) {  // Weakly Aligned
                if (clusterMetric.r < q33) {
                    activityLevel = "WA - Low";
                } else if (clusterMetric.r < q66) {
                    activityLevel = "WA - Moderate";
                } else {
                    activityLevel = "WA - High";
                }
            } else if (currentRegime == -2) {  // Weakly Opposite
                if (clusterMetric.r < q33) {
                    activityLevel = "WOP - Low";
                } else if (clusterMetric.r < q66) {
                    activityLevel = "WOP - Moderate";
                } else {
                    activityLevel = "WOP - High";
                }
            }

            // Add the activity level for this cluster and signal type
            rowData.push_back(activityLevel);
        }

        // After processing the signal type (raw or entropy), store the data
        outputData.push_back(rowData); // Each row has data of each cluster

        // Add headers for each signal type (cluster1(raw), cluster1(entropy), etc.)
        if (signalTypeIndex == 0) {  // For the first signal type (raw)
            for (int i = 1; i <= signalType.size(); ++i) {
                headers.push_back("cluster" + std::to_string(i) + "(raw)");
            }
        } else {  // For the second signal type (entropy)
            for (int i = 1; i <= signalType.size(); ++i) {
                headers.push_back("cluster" + std::to_string(i) + "(entropy)");
            }
        }
    }

    // Print the table header
    for (const auto& header : headers) {
        std::cout << header << "\t";
    }
    std::cout << std::endl;

    // Print the rows of activity data
    for (const auto& row : outputData) {
        for (const auto& activity : row) {
            std::cout << activity << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}
    