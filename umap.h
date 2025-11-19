#pragma once
#include <random>
#include <vector>
#include <C:\Users\H\Desktop\C++ Tryharding\eigen-3.4.0\eigen-3.4.0\Eigen\Dense>

using namespace std;
using namespace Eigen;

double euclidean_distance(const vector<double>& a, const vector<double>& b);
MatrixXd compute_distance_matrix(const vector<vector<double>>& data);
vector<vector<int>> knn_indices(const MatrixXd& dist, int k);
MatrixXd compute_fuzzy_graph(const MatrixXd& dist, const vector<vector<int>>& knn, double sigma = 1.0);
MatrixXd random_init(int n, int dim = 2);
void optimize_embedding(MatrixXd& emb, const MatrixXd& graph, double lr = 1.0, int epochs = 500, double repulsion = 1.0);
MatrixXd umap_from_2d_vector(const vector<vector<double>>& data, int n_neighbors = 10, int n_epochs = 500);