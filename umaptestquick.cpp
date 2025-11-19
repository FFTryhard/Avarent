#include <Eigen/Dense>
#include <knncolle/knncolle.hpp>
#include <umappp/umappp.hpp>
#include <vector>
#include <memory>
#include <iostream>

int main() {
    const int ndim = 8, nobs = 300, out_dim = 2;

    std::vector<double> data(ndim * nobs);
    for (auto& x : data) x = (double)rand() / RAND_MAX;

    knncolle::VptreeBuilder<int, double, double> vp_builder(
        std::make_shared<knncolle::EuclideanDistance<double, double>>()
    );

    umappp::Options opt;
    opt.num_neighbors = 10;
    opt.num_threads   = 4;

    std::vector<double> embedding(nobs * out_dim);

    auto status = umappp::initialize(
        ndim, nobs, data.data(),
        vp_builder,
        out_dim, embedding.data(),
        opt
    );
    status.run(embedding.data());

    std::cout << "ok " << "\n";
    return 0;
}
