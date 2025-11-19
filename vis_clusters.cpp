#include "vis_clusters.h"
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace vis {

// ---------- Small linear algebra helpers (no external deps) ----------

static std::vector<double> mean_vector(const std::vector<std::vector<double>>& X) {
    if (X.empty()) return {};
    size_t d = X[0].size();
    std::vector<double> mu(d, 0.0);
    for (const auto& v : X) {
        for (size_t j = 0; j < d; ++j) mu[j] += v[j];
    }
    double invN = 1.0 / static_cast<double>(X.size());
    for (size_t j = 0; j < d; ++j) mu[j] *= invN;
    return mu;
}

static void center_in_place(std::vector<std::vector<double>>& X, const std::vector<double>& mu) {
    if (X.empty()) return;
    for (auto& v : X) {
        for (size_t j = 0; j < v.size(); ++j) v[j] -= mu[j];
    }
}

static std::vector<std::vector<double>> covariance(const std::vector<std::vector<double>>& X) {
    // X is centered: rows are observations
    size_t n = X.size();
    if (n == 0) return {};
    size_t d = X[0].size();
    std::vector<std::vector<double>> C(d, std::vector<double>(d, 0.0));
    if (n == 1) return C; // all zeros
    for (const auto& v : X) {
        for (size_t i = 0; i < d; ++i) {
            double vi = v[i];
            for (size_t j = i; j < d; ++j) {
                C[i][j] += vi * v[j];
            }
        }
    }
    double scale = 1.0 / static_cast<double>(n - 1);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = i; j < d; ++j) {
            C[i][j] *= scale;
            C[j][i] = C[i][j];
        }
    }
    return C;
}

static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static double norm2(const std::vector<double>& v) {
    return std::sqrt(dot(v, v));
}

static std::vector<double> matvec(const std::vector<std::vector<double>>& A, const std::vector<double>& x) {
    size_t n = A.size();
    std::vector<double> y(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < n; ++j) s += A[i][j] * x[j];
        y[i] = s;
    }
    return y;
}

// Power iteration for symmetric matrix: returns (eigenvector, eigenvalue)
static std::pair<std::vector<double>, double>
power_iteration(const std::vector<std::vector<double>>& A, int max_iters = 1000, double tol = 1e-9) {
    size_t n = A.size();
    std::vector<double> b(n, 0.0);
    // simple deterministic init: [1, 1, ..., 1]
    for (size_t i = 0; i < n; ++i) b[i] = 1.0;

    double lambda = 0.0;
    for (int it = 0; it < max_iters; ++it) {
        std::vector<double> Ab = matvec(A, b);
        double nb = norm2(Ab);
        if (nb < 1e-15) break;
        for (size_t i = 0; i < n; ++i) Ab[i] /= nb;
        // Rayleigh quotient for eigenvalue estimate
        double new_lambda = dot(Ab, matvec(A, Ab));
        double diff = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double di = Ab[i] - b[i];
            diff += di * di;
        }
        b = Ab;
        lambda = new_lambda;
        if (diff < tol * tol) break;
    }
    return {b, lambda};
}

static std::vector<std::vector<double>>
deflate(const std::vector<std::vector<double>>& A, const std::vector<double>& v, double lambda) {
    size_t n = A.size();
    std::vector<std::vector<double>> B = A;
    // B = A - lambda * v v^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            B[i][j] -= lambda * v[i] * v[j];
        }
    }
    return B;
}

static std::vector<std::string> default_palette() {
    // 20 distinct-ish hues
    return {
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#393b79","#637939","#8c6d31","#843c39","#7b4173",
        "#3182bd","#e6550d","#31a354","#756bb1","#9c9ede"
    };
}

// ---------- Public API ----------

void write_svg(const std::vector<std::vector<std::vector<double>>>& clusters,
               const std::string& filename,
               int width,
               int height,
               double point_radius) {
    // Validate and flatten
    size_t total_points = 0;
    size_t dim = 0;
    bool dim_set = false;
    for (const auto& cl : clusters) {
        for (const auto& p : cl) {
            if (!dim_set) { dim = p.size(); dim_set = true; }
            else if (p.size() != dim)
                throw std::invalid_argument("All points must have the same dimensionality.");
            total_points++;
        }
    }
    if (total_points == 0) throw std::invalid_argument("No points to plot.");
    if (dim == 0) throw std::invalid_argument("Points must have dimension >= 1.");

    // Gather all points with cluster labels
    std::vector<std::vector<double>> X;
    X.reserve(total_points);
    std::vector<int> y; // cluster id per point
    y.reserve(total_points);

    for (size_t c = 0; c < clusters.size(); ++c) {
        for (const auto& p : clusters[c]) {
            X.push_back(p);
            y.push_back(static_cast<int>(c));
        }
    }

    // Center and compute covariance
    std::vector<double> mu = mean_vector(X);
    center_in_place(X, mu);
    std::vector<std::vector<double>> C = covariance(X); // dim x dim

    // If dim == 1, fabricate a second axis of zeros
    std::vector<double> u1, u2;
    double l1 = 0.0, l2 = 0.0;

    if (dim == 1) {
        u1 = {1.0};
        u2 = {0.0};
    } else {
        // First eigenvector
        auto [v1, lambda1] = power_iteration(C);
        double n1 = norm2(v1);
        if (n1 > 1e-15) for (auto& t : v1) t /= n1;
        u1 = v1;
        l1 = lambda1;

        // Deflate and get second
        std::vector<std::vector<double>> C2 = deflate(C, u1, l1);
        auto [v2, lambda2] = power_iteration(C2);
        // Orthonormalize v2 against u1 in case of numeric leakage
        double proj = dot(v2, u1);
        for (size_t i = 0; i < v2.size(); ++i) v2[i] -= proj * u1[i];
        double n2 = norm2(v2);
        if (n2 < 1e-12) {
            // Fall back: build an arbitrary orthonormal vector
            v2 = std::vector<double>(dim, 0.0);
            v2[0] = -u1.size() > 1 ? u1[1] : 0.0;
            if (dim > 1) v2[1] = u1[0];
            n2 = norm2(v2);
            if (n2 < 1e-15) v2[0] = 1.0, n2 = 1.0;
        }
        for (auto& t : v2) t /= n2;
        u2 = v2;
        l2 = lambda2;
    }

    // Project to 2D
    std::vector<ProjectedPoint> P;
    P.reserve(total_points);
    double xmin = std::numeric_limits<double>::infinity();
    double xmax = -xmin;
    double ymin = std::numeric_limits<double>::infinity();
    double ymax = -ymin;

    for (size_t i = 0; i < X.size(); ++i) {
        const auto& v = X[i];
        double px = dot(v, u1);
        double py = (dim == 1) ? 0.0 : dot(v, u2);
        xmin = std::min(xmin, px);
        xmax = std::max(xmax, px);
        ymin = std::min(ymin, py);
        ymax = std::max(ymax, py);
        P.push_back({px, py, y[i]});
    }

    // If degenerate spread, pad a bit to avoid divide-by-zero
    if (std::abs(xmax - xmin) < 1e-12) { xmax = xmin + 1.0; xmin -= 1.0; }
    if (std::abs(ymax - ymin) < 1e-12) { ymax = ymin + 1.0; ymin -= 1.0; }

    // Map to SVG coords with margins
    const double margin = 40.0;
    const double W = static_cast<double>(width);
    const double H = static_cast<double>(height);

    auto sx = (W - 2.0 * margin) / (xmax - xmin);
    auto sy = (H - 2.0 * margin) / (ymax - ymin);

    auto to_svg_x = [&](double x) {
        return margin + (x - xmin) * sx;
    };
    // SVG y grows downwards, so invert
    auto to_svg_y = [&](double y) {
        return H - margin - (y - ymin) * sy;
    };

    // Colors
    std::vector<std::string> palette = default_palette();

    // Write SVG
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Failed to open output file: " + filename);

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";

    // Background
    out << "  <rect x=\"0\" y=\"0\" width=\"" << width << "\" height=\"" << height
        << "\" fill=\"white\"/>\n";

    // Axes
    out << "  <g stroke=\"#cccccc\" stroke-width=\"1\">\n";
    out << "    <line x1=\"" << margin << "\" y1=\"" << H - margin
        << "\" x2=\"" << W - margin << "\" y2=\"" << H - margin << "\" />\n";
    out << "    <line x1=\"" << margin << "\" y1=\"" << margin
        << "\" x2=\"" << margin << "\" y2=\"" << H - margin << "\" />\n";
    out << "  </g>\n";

    // Points
    out << "  <g>\n";
    for (const auto& p : P) {
        double cx = to_svg_x(p.x);
        double cy = to_svg_y(p.y);
        const std::string& color = palette[p.cluster % palette.size()];
        out << "    <circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << point_radius
            << "\" fill=\"" << color << "\" fill-opacity=\"0.85\" stroke=\"#222222\" stroke-width=\"0.5\" />\n";
    }
    out << "  </g>\n";

    // Legend
    out << "  <g font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\">\n";
    double lx = W - margin + 10.0;
    double ly = margin + 5.0;
    for (size_t c = 0; c < clusters.size(); ++c) {
        const std::string& color = palette[c % palette.size()];
        double yrow = ly + static_cast<double>(c) * 18.0;
        out << "    <circle cx=\"" << lx << "\" cy=\"" << yrow
            << "\" r=\"5\" fill=\"" << color << "\" stroke=\"#222222\" stroke-width=\"0.5\" />\n";
        out << "    <text x=\"" << (lx + 10.0) << "\" y=\"" << (yrow + 4.0)
            << "\" fill=\"#000000\">Cluster " << c << " (" << clusters[c].size() << ")</text>\n";
    }
    out << "  </g>\n";

    out << "</svg>\n";
    out.close();
}

} // namespace vis
