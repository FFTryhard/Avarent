#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <stdexcept>

// --- Minimal Tensor abstraction ---
struct Tensor {
    std::vector<float> data;
    std::vector<int> shape; // e.g. {batch, time, features}

    Tensor() {}
    Tensor(std::vector<int> shape_) : shape(shape_) {
        int total = 1;
        for (int s : shape_) total *= s;
        data.resize(total);
    }

    float& operator()(int i, int j, int k) {
        // Assume 3D: [A, B, C]
        int idx = i * (shape[1] * shape[2]) + j * (shape[2]) + k;
        return data[idx];
    }

    const float& operator()(int i, int j, int k) const {
        int idx = i * (shape[1] * shape[2]) + j * (shape[2]) + k;
        return data[idx];
    }
};

// --- Load CSV into [rows, features] tensor ---
Tensor readCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open file");

    std::string line;
    bool headerSkipped = false;
    std::vector<std::vector<float>> rows;

    while (std::getline(file, line)) {
        if (!headerSkipped) { headerSkipped = true; continue; }

        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        bool firstCol = true;

        while (std::getline(ss, cell, ',')) {
            if (firstCol) { firstCol = false; continue; } // drop first column
            row.push_back(std::stof(cell));
        }
        rows.push_back(row);
    }

    int nRows = rows.size();
    int nCols = rows[0].size();

    Tensor X({nRows, nCols, 1}); // cheat: store as 3D with singleton 3rd dim
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            X(i, j, 0) = rows[i][j];
        }
    }
    return X;
}

// --- Make sliding windows [num_sequences, window, features] ---
Tensor makeSequences(const Tensor& X, int window_size, int stride) {
    int timesteps = X.shape[0];
    int features  = X.shape[1];
    int num_seq   = (timesteps - window_size) / stride + 1;

    Tensor seqs({num_seq, window_size, features});
    for (int s = 0; s < num_seq; s++) {
        int start = s * stride;
        for (int t = 0; t < window_size; t++) {
            for (int f = 0; f < features; f++) {
                seqs(s, t, f) = X(start + t, f, 0);
            }
        }
    }
    return seqs;
}

// --- Make batches: [num_batches, batch_size, window, features] ---
std::vector<Tensor> makeBatches(const Tensor& seqs, int batch_size) {
    int num_seq  = seqs.shape[0];
    int window   = seqs.shape[1];
    int features = seqs.shape[2];

    std::vector<int> indices(num_seq);
    for (int i = 0; i < num_seq; i++) indices[i] = i;

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<Tensor> batches;
    for (int i = 0; i < num_seq; i += batch_size) {
        int actual_size = std::min(batch_size, num_seq - i);
        Tensor batch({actual_size, window, features});

        for (int b = 0; b < actual_size; b++) {
            int seq_idx = indices[i + b];
            for (int t = 0; t < window; t++) {
                for (int f = 0; f < features; f++) {
                    batch(b, t, f) = seqs(seq_idx, t, f);
                }
            }
        }
        batches.push_back(batch);
    }
    return batches;
}

int main() {
    try {
        // 1. Load CSV into [timesteps, features, 1]
        Tensor data = readCSV("C:/Users/H/Desktop/C++ Tryharding/.csv files/data.csv");
        std::cout << "CSV loaded. Shape: [" 
                  << data.shape[0] << ", " << data.shape[1] << "]\n";

        // 2. Slice into sequences
        int window_size = 30, stride = 5;
        Tensor seqs = makeSequences(data, window_size, stride);
        std::cout << "Sequences shape: [" 
                  << seqs.shape[0] << ", " << seqs.shape[1] 
                  << ", " << seqs.shape[2] << "]\n";

        // 3. Batch them
        int batch_size = 4;
        auto batches = makeBatches(seqs, batch_size);
        std::cout << "Batches: " << batches.size() << "\n";

        // 4. Inspect one batch
        if (!batches.empty()) {
            Tensor& b = batches[0];
            std::cout << "One batch shape: [" 
                      << b.shape[0] << ", " << b.shape[1] 
                      << ", " << b.shape[2] << "]\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}