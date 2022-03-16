import math
import numpy as np 
import matplotlib.pyplot as plt

def proportional_effect(vars_array, var_names_array, cat, title, outfl):
    n_var = len(vars_array)
    n_lines = math.ceil(n_var/3)
    fig, axs = plt.subplots(n_lines, 3, figsize=(15,15))

    axs = axs.flatten()

    unique_cats = np.unique(cat)

    for i, v in enumerate(vars_array):
        for j, d in enumerate(unique_cats):

            catf = cat == d

            mean = np.nanmean(v[catf])
            std = np.nanstd(v[catf])

            axs[i].scatter([mean], [std], label=d, s=30, marker='s')

        axs[i].set_title(var_names_array[i])
        axs[i].grid()
        axs[i].legend()
        axs[i].set_xlabel('Mean')
        axs[i].set_ylabel('Std dev')

    iidx = n_lines*3-(n_lines*3-len(vars_array))
    axs_to_remove = np.arange(iidx, n_lines*3)
    for i in axs_to_remove:
        axs[i].set_visible(False)

    fig.suptitle(title, y=1.0)

    plt.tight_layout()

    plt.savefig(outfl, bbox_inches='tight', facecolor='white')