import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def boxplots(vars_array, vars_names_array, cat, title, outfl, cmap='jet'):
    n_var = len(vars_array)
    n_lines = math.ceil(n_var/3)
    fig, axs = plt.subplots(n_lines, 3, figsize=(15,15))

    axs = axs.flatten()

    unique_cats = np.unique(cat)
    cmap = matplotlib.cm.get_cmap(cmap, len(unique_cats))
    colors = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]

    for i, v in enumerate(vars_array):
        print('Processing {}'.format(vars_names_array[i]))

        fdef = np.isfinite(v)
        v = v[fdef]
        cat_aux = cat[fdef]
        
        bplot = axs[i].boxplot([v[cat_aux==c] for c in unique_cats], labels=unique_cats, patch_artist=True, notch=True)
        
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        
        axs[i].set_title(vars_names_array[i])
        axs[i].grid()

    iidx = n_lines*3-(n_lines*3-len(vars_array))
    axs_to_remove = np.arange(iidx, n_lines*3)
    for i in axs_to_remove:
        axs[i].set_visible(False)

    fig.suptitle(title, y=1.0)

    plt.tight_layout()

    plt.savefig(outfl, bbox_inches='tight', facecolor='white')