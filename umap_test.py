import umap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def run_umap(data, output_plot):

    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    X_umap = umap_model.fit_transform(data)

    # Plot
    plt.scatter(X_umap[:, 0], X_umap[:, 1])
    plt.title('UMAP projection of the data')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig(output_plot)
    plt.close()


    return {
        "reduced_data": np.array(X_umap) # Keep it as a NumPy array
    }
