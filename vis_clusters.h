#pragma once
#include <vector>
#include <string>

namespace vis {

struct ProjectedPoint {
    double x;
    double y;
    int cluster;
};

/// Visualize clusters (cluster_id -> points -> coordinates) to an SVG file.
/// - clusters: shape [num_clusters][num_points_in_cluster][dim], dim can be any N>=1
/// - filename: output SVG path
/// - width, height: SVG canvas size in pixels
/// - point_radius: circle radius in pixels
/// Notes:
/// - Uses PCA (top 2 components via power-iteration + deflation) to project N-D -> 2D
/// - Colors cycle through a vivid palette per cluster
void write_svg(const std::vector<std::vector<std::vector<double>>>& clusters,
               const std::string& filename,
               int width = 800,
               int height = 600,
               double point_radius = 3.0);

} // namespace vis