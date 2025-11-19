#include <C:\Users\H\Desktop\C++ Tryharding\eigen-3.4.0\eigen-3.4.0\Eigen\Dense>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <limits>

using namespace Eigen;
using namespace std;

// ========== Utility Functions ==========

// Compute squared Euclidean distance matrix (n x n)
MatrixXd pairwiseDist2(const MatrixXd &X) {
    int n = X.rows();
    MatrixXd sum_X = X.rowwise().squaredNorm();
    MatrixXd D = sum_X.replicate(1, n) + sum_X.transpose().replicate(n, 1) - 2 * X * X.transpose();
    D = D.cwiseMax(0.0); // numerical stability
    return D;
}

// Compute P affinities with binary search for each point
MatrixXd computeP(const MatrixXd &D, double perplexity = 30.0, double tol = 1e-5) {
    int n = D.rows();
    MatrixXd P = MatrixXd::Zero(n, n);
    double logU = log(perplexity);

    for (int i = 0; i < n; ++i) {
        double beta = 1.0;
        double betamin = -numeric_limits<double>::infinity();
        double betamax = numeric_limits<double>::infinity();
        VectorXd Di = D.row(i);
        Di(i) = numeric_limits<double>::infinity();

        // binary search for correct beta (precision)
        VectorXd Pi(n);
        bool found = false;
        for (int iter = 0; iter < 50; ++iter) {
            VectorXd Pj = (-Di * beta).array().exp();
            Pj(i) = 0.0;
            double sumPj = Pj.sum();
            double H = log(sumPj) + beta * (Di.cwiseProduct(Pj)).sum() / sumPj;
            double Hdiff = H - logU;

            if (fabs(Hdiff) < tol) { found = true; break; }

            if (Hdiff > 0) {
                betamin = beta;
                if (isinf(betamax)) beta *= 2.0; else beta = (beta + betamax) / 2.0;
            } else {
                betamax = beta;
                if (isinf(betamin)) beta /= 2.0; else beta = (beta + betamin) / 2.0;
            }
        }

        Pi = (-Di * beta).array().exp();
        Pi(i) = 0.0;
        Pi /= Pi.sum();
        P.row(i) = Pi.transpose();
    }

    // Symmetrize and normalize
    MatrixXd Psym = (P + P.transpose()) / (2.0 * n);
    return Psym;
}

// ========== t-SNE Core ==========

MatrixXd tsne(const MatrixXd &X, int no_dims = -1, double perplexity = 30.0, int max_iter = 1000,
              double learning_rate = 200.0, double momentum = 0.8) {

    int n = X.rows();
    int orig_dims = X.cols();
    if (no_dims <= 0 || no_dims > 3) no_dims = std::min(3, orig_dims);

    cout << "t-SNE: Reducing from " << orig_dims << "D to " << no_dims << "D" << endl;

    MatrixXd D = pairwiseDist2(X);
    MatrixXd P = computeP(D, perplexity);
    P = (P.array() < 1e-12).select(1e-12, P);

    // Initialize low-dimensional map Y
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1e-4);
    MatrixXd Y = MatrixXd::NullaryExpr(n, no_dims, [&]() { return dist(gen); });
    MatrixXd dY = MatrixXd::Zero(n, no_dims);
    MatrixXd iY = MatrixXd::Zero(n, no_dims);

    double initial_momentum = 0.5;
    double final_momentum = momentum;
    int mom_switch_iter = 250;
    int stop_lying_iter = 100;
    double exaggeration = 12.0;

    MatrixXd Pexag = P * exaggeration;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute pairwise affinities in low-d space (Student-t kernel)
        MatrixXd sum_Y = Y.rowwise().squaredNorm();
        MatrixXd num = (sum_Y.replicate(1, n) + sum_Y.transpose().replicate(n, 1) - 2 * Y * Y.transpose());
        num = (1.0 + num.array()).inverse();
        num.diagonal().setZero();
        double sum_num = num.sum();
        MatrixXd Q = num / sum_num;
        Q = (Q.array() < 1e-12).select(1e-12, Q);

        // Gradient computation
        for (int i = 0; i < n; ++i) {
            RowVectorXd grad = RowVectorXd::Zero(no_dims);
            for (int j = 0; j < n; ++j) {
                double pqdiff = (iter < stop_lying_iter ? Pexag(i, j) : P(i, j)) - Q(i, j);
                grad += pqdiff * num(i, j) * (Y.row(i) - Y.row(j));
            }
            dY.row(i) = 4.0 * grad;
        }

        // Update Y
        double cur_mom = (iter < mom_switch_iter) ? initial_momentum : final_momentum;
        iY = cur_mom * iY - learning_rate * dY;
        Y += iY;

        // zero-mean
        RowVectorXd meanY = Y.colwise().mean();
        Y.rowwise() -= meanY;

        if ((iter + 1) % 100 == 0)
            cout << "Iteration " << iter + 1 << " / " << max_iter << endl;
    }

    return Y;
}
/*
int main() {
    // Example: create a simple 3D dataset (e.g., swiss roll)
    int n = 500;
    MatrixXd X(n, 3);
    for (int i = 0; i < n; ++i) {
        double t = 1.5 * M_PI * (1 + 2.0 * i / n);
        double h = 21.0 * ((double) i / n);
        X(i, 0) = t * cos(t);
        X(i, 1) = h;
        X(i, 2) = t * sin(t);
    }
                        //-1 = adaptive, else is fixed //30 is neighbours // 500 = iters // 200 is gradient descent steps // 0.8 = Momentum term for smoother updates — helps prevent oscillations during optimization.
    MatrixXd Y = tsne(X, -1, 30.0, 500, 200.0, 0.8);
    cout << "Reduced data (first 5 rows):\n" << Y.topRows(5) << endl;

    return 0;
}
*/