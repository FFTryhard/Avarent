#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <random>
#include <filesystem>
#include "utilspy.h"

namespace py = pybind11;

std::vector<std::vector<double>> UMAP(std::vector<std::vector<double>> data_target);