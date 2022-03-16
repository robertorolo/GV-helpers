import math
import numpy as np
import matplotlib.pyplot as plt

def boxplots(vars_array, vars_names_array, cat, title, outfl):
    n_var = len(vars_array)
    n_lines = math.ceil(n_var/3)
    fig, axs = plt.subplots(n_lines, 3, figsize=(15,15))

    axs = axs.flatten()

    unique_cats = np.unique(cat)

    for i, v in enumerate(vars_array):
        print('Processing {}'.format(vars_names_array[i]))

        fdef = np.isfinite(v)
        v = v[fdef]
        cat_aux = cat[fdef]
        
        axs[i].boxplot([v[cat_aux==c] for c in unique_cats], labels=unique_cats)
        axs[i].set_title(vars_names_array[i])
        axs[i].grid()

    iidx = n_lines*3-(n_lines*3-len(vars_array))
    axs_to_remove = np.arange(iidx, n_lines*3)
    for i in axs_to_remove:
        axs[i].set_visible(False)

    fig.suptitle(title)

    plt.tight_layout()

    plt.savefig(outfl, bbox_inches='tight', facecolor='white')