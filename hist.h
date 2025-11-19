#pragma once
#include <iostream>
#include "utils.h"
#include <vector>


void histogramAnalysis(const std::vector<double> &data);

double FreedmanDiaconisRuleIQR(const std::vector<double> &vec, int numBins);

double ScottsNormalReferenceRule(const std::vector<double>& data);

int calculateNumBins(double maxValue, double minValue, double binWidth, int n);

int binIndex(double val, int num_bins);

//int binIndex(double val, int num_bins, double minValue, double maxValue);