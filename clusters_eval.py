import spatialcluster as sp
import matplotlib.pyplot as plt
import numpy as np

def clusters_eval(data, locs, clusters, cluster_names, srch, nn, title, outfl):
    fig, axs = plt.subplots(1, 1, figsize=(8,8))

    for idx, label in enumerate(cluster_names):
        m = sp.cluster_metrics_single(data, locs, clusters[idx], nnears=nn, searchparams=srch)
        axs.scatter(m[1], m[0], label=label)

    m = sp.cluster_metrics_single(data ,locs , np.random.randint(0, 2, len(data)), nnears=nn, searchparams=srch)
    axs.scatter(m[1], m[0], label='Random')

    axs.grid()
    axs.legend()
    axs.set_ylabel('Multivariate delineation (WCSS)')
    axs.set_xlabel('Spatal contiguity (Entropy)')

    plt.savefig(outfl, bbox_inches='tight', facecolor='white')