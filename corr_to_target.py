import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def correlations_to_target(vars_array, vars_names_array, target, target_name, cat, title, outfl):
    n_var = len(vars_array)
    n_lines = math.ceil(n_var/3)
    fig, axs = plt.subplots(n_lines, 3, figsize=(15,15))

    axs = axs.flatten()

    unique_cats = np.unique(cat)
    cat_map = {}
    c = 0
    for i in unique_cats:
        cat_map[i] = c
        c = c+1
    inv_map = {v: k for k, v in cat_map.items()}

    cat_num = [cat_map[i] for i in cat]

    n = len(unique_cats)
    ht = (n-1)/n*1/2

    for i, v in enumerate(vars_array):

        cmap = cm.get_cmap('jet', len(unique_cats))
        rho = np.corrcoef(v, target)
        
        sc = axs[i].scatter(v, target, c=cat_num, s=2, cmap=cmap)
        axs[i].set_xlabel(vars_names_array[i])
        axs[i].set_ylabel(target_name)
        axs[i].grid()
        axs[i].annotate('rho: {}'.format(rho[1][0].round(2)), xy=(0.7, 0.05), xycoords='axes fraction', color='black')
        
        cbar = fig.colorbar(sc, ax=axs[i])
        tick_locs = np.linspace(ht, (n-1) - ht, n)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels([inv_map[i] for i in np.arange(len(unique_cats))])

    iidx = n_lines*3-(n_lines*3-len(vars_array))
    axs_to_remove = np.arange(iidx, n_lines*3)
    for i in axs_to_remove:
        axs[i].set_visible(False)


    fig.suptitle(title)

    plt.tight_layout()


    plt.savefig(outfl, bbox_inches='tight', facecolor='white')