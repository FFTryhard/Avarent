#include <iostream>
#include "utils.h"
#include <vector>

double median(const std::vector<double> &data){

    std::vector<double> vec = data;
    std::sort(vec.begin(), vec.end());  

    if(data.size() % 2 == 0){

        return 0.5 * (vec[vec.size()/2 - 1] + vec[vec.size()/2]);
    }else{

        return vec[vec.size()/2];
    }
}

double mean(const std::vector<double> &data){

    double sum = 0.0;

    for(auto val : data){
        sum = sum + val;
    }
    

    return sum / data.size();
}

std::vector<std::vector<double>> transposeData(const std::vector<std::vector<double>>& matrix) {

    if (matrix.empty() || matrix[0].empty()) {
       throw std::invalid_argument("[transposeData] Matrix invalid");
    }

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::vector<std::vector<double>> result(cols, std::vector<double>(rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}

std::vector<std::vector<double>> matrixToVector2D(const Eigen::MatrixXd& mat) {
    std::vector<std::vector<double>> result(mat.rows(), std::vector<double>(mat.cols()));

    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            result[i][j] = mat(i, j);
        }
    }

    return result;
}

double avgClusterSize(const std::vector<std::vector<std::vector<double>>>& clusters){
    double sum= 0;
    for(auto cluster : clusters){
        sum = sum + cluster.size();
    }
    return sum / clusters.size();
}

double distanceBlend(int iter, bool trigger, int startIter, int rampLength) {
 

    if (!trigger) return 0.0;               // fully Euclidean
    if (iter >= startIter + rampLength) return 1.0; // fully Mahalanobis after past this amount of iterations
    return double(iter - startIter) / rampLength;   // returns the actual blend value
}

double adaptiveAlpha(int clusterSize) {

    int minSize = 10;
    int maxSize = 500;
    double alphaMin = 0.05;
    double alphaMax = 0.3;
    // Clamp cluster size
    clusterSize = std::max(minSize, std::min(clusterSize, maxSize));
    // Linear mapping clusterSize -> alpha in [alphaMax, alphaMin]
    double alpha = alphaMax - (clusterSize - minSize) * (alphaMax - alphaMin) / (maxSize - minSize);
    return alpha;
}

bool allClustersHaveEnoughPoints(
    const std::vector<std::vector<std::vector<double>>>& clusters,
    size_t featureDim,
    size_t safetyFactor
) {
    if (clusters.empty()) return false;

    for (const auto& cluster : clusters) {
        if (cluster.size() < safetyFactor * featureDim) {
            return false;
        }
    }
    return true;
}

Eigen::VectorXd toEigenVector(const std::vector<double>& vec) {
    Eigen::VectorXd eigenVec(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        eigenVec(i) = vec[i];
    }
    return eigenVec;
}

Eigen::MatrixXd vectorToEigenMatrix(const std::vector<std::vector<double>>& data) {

    if (data.empty()) {
        std::cout << "return empty matrix" << std::endl;
        return Eigen::MatrixXd(); // return empty matrix
    }

    size_t rows = data.size();
    size_t cols = data[0].size();

    // Verify all rows have the same size
    for (const auto& row : data) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows in the 2D vector must have the same length.");
        }
    }

    Eigen::MatrixXd mat(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = data[i][j];
        }
    }

    return mat;
}

bool choleskyDecompLLT(const Eigen::MatrixXd &cov){

    if (cov.rows() != cov.cols() || cov.rows() == 0)
    return false;

    Eigen::MatrixXd sym = 0.5 * (cov + cov.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt(sym);
    return llt.info() == Eigen::Success; //better than numerical issue, checks more

}

bool choleskyDecompLDLT(const Eigen::MatrixXd &cov){

    if (cov.rows() != cov.cols() || cov.rows() == 0)
    return false;

    Eigen::MatrixXd sym = 0.5 * (cov + cov.transpose());
    Eigen::LDLT<Eigen::MatrixXd> llt(sym);
    return llt.info() == Eigen::Success;

}

Eigen::MatrixXd getInvCov(const Eigen::MatrixXd &data){
        
        Eigen::MatrixXd cov = data; 

        cov = 0.5 * (cov + cov.transpose()); //exacty symmetric

        // Cholesky decomposition
        Eigen::LLT<Eigen::MatrixXd> llt(cov);
        Eigen::MatrixXd cov_inv;

        // Check if the matrix is positive definite
        if (llt.info() == Eigen::NumericalIssue) {
           std::cerr << "[getInvCov] Matrix is not positive definite! Using Pseudo-Inverse!" << std::endl;
           cov += Eigen::MatrixXd::Identity(cov.rows(), cov.cols()) * 1e-3; 
           cov_inv = cov.inverse();  // slower but safer fallback
        } else {
        // Compute the inverse using the Cholesky factor
           cov_inv = llt.solve(Eigen::MatrixXd::Identity(cov.rows(), cov.cols())); 
        }
    
   return cov_inv;
}

std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> CovarianceMatrix(std::vector<std::vector<std::vector<double>>>& clusters, const std::vector<std::vector<double>>& centroids, const std::vector<std::vector<double>>& data) {
    
    int d = centroids[0].size();
    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> results; 
        
    //small clusters + high-dim = guaranteed numerical chaos.
    
    // Iterate over all clusters
    for (size_t z = 0; z < clusters.size(); ++z) {
        Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(d, d);
        
        // Iterate over each vector inside one cluster
        for (size_t i = 0; i < clusters[z].size(); ++i) {
            Eigen::VectorXd centered(d); 
            
            // Center the vector by subtracting the centroid
            for (size_t j = 0; j < clusters[z][i].size(); ++j) {
                centered(j) = clusters[z][i][j] - centroids[z][j];
            }

            // Add the outer product of the centered vector to the covariance matrix
            cov += centered * centered.transpose();
        }

        int n = static_cast<int>(clusters[z].size());
        if (n <= 1) {
         results.push_back(Eigen::MatrixXd::Identity(d, d) * 1e-1);
         continue; //When a cluster has fewer than 2 points, (n – 1) blows up or I end up with garbage covariance.
        }

        Eigen::MatrixXd global_cov = computeRobustCovarianceMAD(data); //global covariance of data

        // Divide by (n-1) for unbiased covariance estimate
        cov /= (clusters[z].size() - 1);

        cov += Eigen::MatrixXd::Identity(d, d) * 1e-4;  // Add small regularization. affect off diagonals

        //Eigen::Matrix<double, 14, 14> I = Eigen::Matrix<double, 14, 14>::Identity();
        double lambda = 5e-4;  // or adaptive: lambda = 0.1 * cov.trace() / 14.0;

        //double lambda = 1e-4 + 1e-3 * (1.0 / sqrt(clusters[z].size())); //adaptive? smaller clusters get more pull
 
        //Eigen::Matrix<double, 14, 14> T = (cov.trace() / 14) * I;
        cov = (1 - lambda) * cov + lambda * global_cov;

        cov *= (d / cov.trace()); // normalize average variance ~ 1.0. global cov step // or do det based

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
        auto evals = es.eigenvalues();
        bool bad = (evals.minCoeff() <= 0);
        if (bad)
           cov += Eigen::MatrixXd::Identity(d, d) * 1e-2;
        //Numerical noise in high-dim data can produce slightly negative eigenvalues even when it “shouldn’t.”


        // Store the covariance matrix (7x7) for the cluster
        results.push_back(cov);
    }

    return results;  
}

std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> CovarianceMatrixGPT(
    const std::vector<std::vector<std::vector<double>>>& clusters,
    const std::vector<std::vector<double>>& centroids,
    const std::vector<std::vector<double>>& data)
{
    int d = centroids[0].size();
    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> results;

    // Compute global covariance once for shrinkage
    Eigen::MatrixXd global_cov = computeRobustCovarianceMAD(data); // user-defined
    double global_trace = global_cov.trace();

    for (size_t z = 0; z < clusters.size(); ++z) {
        int n = static_cast<int>(clusters[z].size());

        // Small clusters -> fallback
        if (n <= 1) {
            results.push_back(Eigen::MatrixXd::Identity(d, d) * 1e-4);
            continue;
        }

        // Compute centered data matrix for cluster (d x n)
        Eigen::MatrixXd centered(d, n);
        for (int i = 0; i < n; ++i) {
            centered.col(i) = Eigen::Map<const Eigen::VectorXd>(clusters[z][i].data(), d)
                              - Eigen::Map<const Eigen::VectorXd>(centroids[z].data(), d);
        }

        // Unbiased covariance
        Eigen::MatrixXd cov = (centered * centered.transpose()) / (n - 1);

        // Regularization: small diagonal scaled by average variance
        double diag_eps = std::max(1e-6, 1e-4 * cov.trace() / d); //so its not too weak
        cov += Eigen::MatrixXd::Identity(d, d) * diag_eps;

        // Adaptive shrinkage toward global covariance
        double lambda = 1e-4; //+ 1e-2 * (1.0 / std::sqrt(n)); // smaller clusters get more shrinkage
        cov = (1.0 - lambda) * cov + lambda * global_cov;

        // Normalize average variance ~1 for numerical stability
        //cov *= d / cov.trace(); might kill variance

        // Eigenvalue check for numerical issues
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
        Eigen::VectorXd evals = es.eigenvalues();
        double min_eval = evals.minCoeff();

        double eps_threshold = 1e-8;  // lower threshhold
        double adjust_eps = std::max(1e-6, 1e-4 * cov.trace() / d); //adaptive guard 

        if (min_eval <= eps_threshold) {
            cov += Eigen::MatrixXd::Identity(d, d) * adjust_eps;
        }

        results.push_back(cov);
    }

    return results;
}

double mahalanobisDistanceSingle(const Eigen::VectorXd& point, const Eigen::VectorXd& centroid, const Eigen::MatrixXd& covariance) {
    

    //Eigen::Matrix<double, 14, 14> cov_reg = covariance;
    //cov_reg += 1e-3 * Eigen::Matrix<double, 14, 14>::Identity();   //pretty sure we dont need this since we already did shrinking in the covariancematrix function

    // Defensive copy to avoid modifying the original matrix
    Eigen::MatrixXd cov = covariance;

    cov = 0.5 * (cov + cov.transpose()); //exacty symmetric

        // Cholesky decomposition
        Eigen::LLT<Eigen::MatrixXd> llt(covariance);
        Eigen::MatrixXd cov_inv;

        // Check if the matrix is positive definite
        if (llt.info() == Eigen::NumericalIssue) {
           std::cerr << "[MdistSingle]Matrix is not positive definite! Using Pseudo-Inverse!" << std::endl;
           cov += Eigen::MatrixXd::Identity(cov.rows(), cov.cols()) * 1e-3; 
           cov_inv = cov.inverse();  // slower but safer fallback
        } else {
        // Compute the inverse using the Cholesky factor
           cov_inv = llt.solve(Eigen::MatrixXd::Identity(covariance.rows(), covariance.cols())); //redeclaration fucked me up
        }
    
    // Calculate the difference between the point and the mean
    Eigen::VectorXd diff = point - centroid;

    if (diff.size() != cov_inv.rows()) {
        std::cerr << "[MdistSingle] Dimension mismatch in Mahalanobis: diff=" << diff.size()
                  << " cov=" << cov_inv.rows() << std::endl;
    }
    
    // Compute Mahalanobis distance
    double distance = std::sqrt(diff.transpose() * cov_inv * diff); //someone said to do this, idk why lol

    if (distance < 0) distance = 0.0;
    
    return distance;
}

std::vector<double> MahalanobisDistMulti(
    const std::vector<double>& point,
    const std::vector<std::vector<double>>& meanvecs,
    const std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>& covmatrices
) {

    // Convert input point into Eigen
    Eigen::VectorXd x = toEigenVector(point);

    std::vector<double> distances;
    distances.reserve(meanvecs.size());

    for (size_t c = 0; c < meanvecs.size(); ++c) {
        // Convert mean
        
        Eigen::VectorXd mu = toEigenVector(meanvecs[c]);

        // Regularize covariance to avoid singularities
        Eigen::MatrixXd cov_reg = covmatrices[c];
        
        //cov_reg += 1e-3 * Eigen::Matrix<double, 14, 14>::Identity();
        cov_reg = 0.5 * (cov_reg + cov_reg.transpose());   
       
        // Inverse covariance
        Eigen::MatrixXd cov_inv;
        // Cholesky decomposition
        Eigen::LLT<Eigen::MatrixXd> llt(cov_reg);

        // Check if the matrix is positive definite
        if (llt.info() == Eigen::NumericalIssue) {
        std::cerr << "[MdistMulti] Matrix is not positive definite!" << std::endl;
        } else {
         // Compute the inverse using the Cholesky factor
        Eigen::MatrixXd cov_inv = llt.solve(Eigen::MatrixXd::Identity(cov_reg.rows(), cov_reg.cols()));
        }


        // Compute diff
        Eigen::VectorXd diff = x - mu;

        // Mahalanobis distance
        double dist_sq = (diff.transpose() * cov_inv * diff)(0, 0); //That ensures im not silently storing a matrix object with garbage or misaligned memory.
        distances.push_back(std::sqrt(dist_sq));
    }

    return distances;
}

double cosineSimilarity(Eigen::Map<Eigen::VectorXd> &A, Eigen::Map<Eigen::VectorXd> &B){ //takes 2 vectors

    float epsilon = 1e-10f;

    //i think a is comparing it a point thats not there?

    double dot_product = A.dot(B);  // A · B  //does A . B
    double norm_A = A.norm();       // ||A||
    double norm_B = B.norm();       // ||B||  //length vector from origin. "length" of a vector. // why is B 0 

    if (norm_A < epsilon || norm_B < epsilon) {
      //if shits too small
      return 0.0;
    }

    //dividing by the norm allows you to focus on the direction of the vectors (the angle between them), rather than their magnitudes.
    
    return dot_product / (norm_A * norm_B);
}

double mahalanobisDistanceSingleFast(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::MatrixXd& invCov){
    
    Eigen::VectorXd diff = x - y;
    double distsquared = std::sqrt((diff.transpose() * invCov * diff).value());
    double dist = std::sqrt(std::max(distsquared, 0.0));
    if(dist < 0) dist = 0;

    if (diff.size() != invCov.rows()) {
        std::cerr << "[MdistSingleFast] Dimension mismatch: diff=" << diff.size()
                  << " cov=" << invCov.rows() << std::endl;
    }

    return dist;
    
}

Eigen::MatrixXd computeGlobalCovariance(const std::vector<std::vector<double>>& data) {
    size_t n = data.size();
    size_t d = data[0].size();
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
    for (auto& x : data)
        for (size_t j = 0; j < d; ++j)
            mean(j) += x[j];
    mean /= n;

    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(d, d);
    for (auto& x : data) {
        Eigen::VectorXd diff(d);
        for (size_t j = 0; j < d; ++j)
            diff(j) = x[j] - mean(j);
        cov += diff * diff.transpose();
    }

    cov /= (n - 1);
    return cov;
}

Eigen::MatrixXd computeRobustCovarianceMAD(const std::vector<std::vector<double>>& data) {
    size_t n = data.size();
    if (n < 2) {
        throw std::invalid_argument("[ComputeRCovMAD] At least two data points are required to compute covariance.");
    }

    size_t d = data[0].size();
    for (const auto& x : data) {
        if (x.size() != d) {
            throw std::invalid_argument("[ComputeRCovMAD] Inconsistent row size detected.");
        }
    }

    // Step 1: Compute the mean of each dimension
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
    for (const auto& x : data) {
        for (size_t j = 0; j < d; ++j) {
            mean(j) += x[j];
        }
    }
    mean /= n;

    // Step 2: Compute the MAD for each dimension
    Eigen::MatrixXd madMatrix = Eigen::MatrixXd::Zero(d, d);
    
    // For each dimension, compute the MAD
    for (size_t j = 0; j < d; ++j) {
        std::vector<double> columnValues;
        for (const auto& x : data) {
            columnValues.push_back(x[j]);
        }

        double mad = computeMAD(columnValues);

        // The "robust covariance" is simply MAD applied to each dimension for now
        madMatrix(j, j) = mad * mad;  // Using MAD^2 as a measure of spread for variance
    }

    return madMatrix;
}

double computeMAD(const std::vector<double>& values) {
    std::vector<double> absDeviations(values.size());
    double median = 0;

    // Compute the median of the values
    std::vector<double> sortedValues = values;
    std::sort(sortedValues.begin(), sortedValues.end());

    size_t n = sortedValues.size();
    if (n % 2 == 0) {
        median = (sortedValues[n / 2 - 1] + sortedValues[n / 2]) / 2.0;
    } else {
        median = sortedValues[n / 2];
    }

    // Calculate the absolute deviations from the median
    for (size_t i = 0; i < values.size(); ++i) {
        absDeviations[i] = std::abs(values[i] - median);
    }

    // Compute the median of the absolute deviations
    std::sort(absDeviations.begin(), absDeviations.end());
    double mad = 0;
    if (n % 2 == 0) {
        mad = (absDeviations[n / 2 - 1] + absDeviations[n / 2]) / 2.0;
    } else {
        mad = absDeviations[n / 2];
    }

    return mad;
}

std::vector<std::vector<double>> PCA(const Eigen::MatrixXd& data, double varianceThreshold){

    if(data.rows() == 0 || data.cols() == 0) {
      throw std::runtime_error("Input data matrix is empty!");
    }

    if(data.rows() == 1) {
      throw std::runtime_error("PCA requires at least 2 rows of data.");
    }

    varianceThreshold = std::clamp(varianceThreshold, 0.0, 1.0);
    // Step 1: Mean-center the data
    const Eigen::MatrixXd centered = data.rowwise() - data.colwise().mean();

    // Step 2: Compute covariance matrix
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(data.rows() - 1);

    // Step 3: Eigen decomposition (Eigen gives ascending order)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);

    if (eig.info() != Eigen::Success) {
      throw std::runtime_error("Eigen decomposition failed.");
    }  

    Eigen::VectorXd eigenValues = eig.eigenvalues();
    Eigen::MatrixXd eigenVectors = eig.eigenvectors();

    // Step 4: Reverse to get descending order (largest eigenvalue first)
    Eigen::VectorXd eigenValuesDesc = eigenValues.reverse();

    Eigen::MatrixXd eigenVectorsDesc(eigenVectors.rows(), eigenVectors.cols());
    for (int i = 0; i < eigenVectors.cols(); ++i) {
        eigenVectorsDesc.col(i) = eigenVectors.col(eigenVectors.cols() - 1 - i);
    }

    // Step 5: Determine number of components to keep based on variance threshold
    double totalVariance = eigenValuesDesc.sum();
    double runningSum = 0.0;
    int k = eigenValuesDesc.size();
    for (int i = 0; i < eigenValuesDesc.size(); ++i) {
        runningSum += eigenValuesDesc(i);
        if (runningSum / totalVariance >= varianceThreshold) {
            k = i + 1;
            break;
        }
    }

    std::cout << "Keeping " << k << " principal components.\n";

    // Step 6: Project data onto top-k components
    Eigen::MatrixXd reduced = centered * eigenVectorsDesc.leftCols(k);

    // Step 7: Convert Eigen matrix → std::vector<std::vector<double>>
    std::vector<std::vector<double>> result(reduced.rows(), std::vector<double>(reduced.cols()));
    for (int i = 0; i < reduced.rows(); ++i)
        for (int j = 0; j < reduced.cols(); ++j)
            result[i][j] = reduced(i, j);

    return result;
}

double sampleVariance(const std::vector<double>& data) {
    if (data.size() <= 1) {
        throw std::invalid_argument("Sample size must be greater than 1 for variance calculation.");
    }

    double mean = 0.0;
    double sum_of_squares = 0.0;

    // Compute the mean
    for (const auto& val : data) {
        mean += val;
    }
    mean /= data.size();

    // Calculate the sum of squared deviations from the mean
    for (const auto& val : data) {
        sum_of_squares += (val - mean) * (val - mean);
    }

    // Return sample variance (divide by n-1)
    return sum_of_squares / (data.size() - 1);
}

double euclideanDist(const std::vector<double>& a, const std::vector<double>& b) {

    //takes in vector
    if (a.size() != b.size()) {
        std::cerr << "[euclideanDist] Dimension mismatch: a.size() = " << a.size()
                  << ", b.size() = " << b.size() << std::endl;
        exit(1); // Or th3row exception
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }

    //sums up the squared differences of each single dimension
    //measures how far theyre apart in high dimensional space
    return sqrt(sum);
}




