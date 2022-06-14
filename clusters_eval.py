from numpy.random import pareto
import spatialcluster as sp
import matplotlib.pyplot as plt
import numpy as np

def is_pareto(costs, maximise=False):
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient

def clusters_eval(data, locs, clusters, cluster_names, srch, nn, title, outfl, pareto):
    fig, axs = plt.subplots(1, 1, figsize=(8,8))

    costs =[]

    for idx, label in enumerate(cluster_names):
        m = sp.cluster_metrics_single(data, locs, clusters[idx], nnears=nn, searchparams=srch)
        axs.scatter(m[1], m[0], label=label)
        costs.append([m[1], m[0]])

    m = sp.cluster_metrics_single(data ,locs , np.random.randint(0, 2, len(data)), nnears=nn, searchparams=srch)
    axs.scatter(m[1], m[0], label='Random')

    pareto_mask = is_pareto(np.array(costs))
    costs_y = costs[pareto_mask]

    if pareto:
        axs.scatter(costs_y.T[0], costs_y.T[1], color='red', label='Pareto optimal')

    axs.grid()
    axs.legend()
    axs.set_ylabel('Multivariate delineation (WCSS)')
    axs.set_xlabel('Spatal contiguity (Entropy)')
    plt.set_title(title)

    plt.savefig(outfl, bbox_inches='tight', facecolor='white')