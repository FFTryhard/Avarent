import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def fit_and_match(gec_data, previous_means):
    # Ensure inputs are numpy arrays
    gec_data = np.asarray(gec_data, dtype=np.float64)
    previous_means = np.asarray(previous_means, dtype=np.float64)

    K = previous_means.shape[0]

    # Fit K-Means
    kmeans = KMeans(
        n_clusters=K,
        init=previous_means,
        n_init=1,          # Required when supplying initial centers
        max_iter=100,
        random_state=42,
    )

    kmeans.fit(gec_data)

    new_means = kmeans.cluster_centers_

    # Compute Euclidean distance matrix
    cost_matrix = cdist(new_means, previous_means, metric="euclidean")

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Reorder means
    matched_means = np.zeros_like(previous_means)
    for r, c in zip(row_ind, col_ind):
        matched_means[c] = new_means[r]

    print("KMEANS MODEL RETURNING MEANS")

    return matched_means

"""
def fit_and_match(gec_data, previous_means):

    # Ensure inputs are numpy arrays
    gec_data = np.asarray(gec_data, dtype=np.float64)
    previous_means = np.asarray(previous_means, dtype=np.float64)

    K = previous_means.shape[0]

    # Fit the GMM
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="diag",
        means_init=previous_means,
        max_iter=100,
        random_state=42,
    )

    gmm.fit(gec_data)

    new_means = gmm.means_

    # Compute Euclidean distance matrix
    cost_matrix = cdist(new_means, previous_means, metric="euclidean")

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Reorder means
    matched_means = np.zeros_like(previous_means)
    for r, c in zip(row_ind, col_ind):
        matched_means[c] = new_means[r]

    print("GMM MODEL RETURNING MEANS")

    return matched_means
"""

"""
import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def fit_and_match(gec_data, previous_means):
 
    # Ensure inputs are numpy arrays
    gec_data = np.asarray(gec_data, dtype=np.float64)
    previous_means = np.asarray(previous_means, dtype=np.float64)
    
    K = previous_means.shape[0]
    
    # Fit the HMM
    # We use a small number of iterations since we run this continuously on rolling windows
    hmm = GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state=42)
    
    # Optional: We could initialize hmm.means_ with previous_means here 
    # to speed up convergence and stabilize it, but for purity we'll let it fit 
    # from the data and just match labels afterwards.
    hmm.fit(gec_data)
    
    new_means = hmm.means_
    
    # Compute Euclidean distance matrix between new means and previous means
    # cost_matrix[i, j] = distance from new_means[i] to previous_means[j]
    cost_matrix = cdist(new_means, previous_means, metric='euclidean')
    
    # Hungarian assignment to find the optimal 1-to-1 mapping
    # row_ind: indices of new_means
    # col_ind: indices of previous_means they are matched to
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Re-order new_means so the i-th row corresponds to the i-th previous_mean
    matched_means = np.zeros_like(previous_means)
    for r, c in zip(row_ind, col_ind):
        matched_means[c] = new_means[r]

    print("HMM MODEL RETURNING MEANS")
        
    return matched_means
"""