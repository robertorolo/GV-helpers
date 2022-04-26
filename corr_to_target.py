import math
import numpy as np
import matplotlib.pyplot as plt

def discrete_cmap(N, base_cmap='jet'):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def correlations_to_target(vars_array, vars_names_array, target, target_name, cat, nmax, figsize, title, outfl):
    n_var = len(vars_array)
    n_lines = math.ceil(n_var/3)
    fig, axs = plt.subplots(n_lines, 3, figsize=figsize)

    if nmax is not None:
        idxs = np.arange(len(vars_array[0]))
        ridxs = np.random.choice(idxs, size=nmax, replace=False)
        vars_array = [i[ridxs] for i in vars_array]
        cat = np.array(cat)[ridxs]
        target = target[ridxs]
    else:
        pass

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
    tick_dif = (n-1)/(2*n)

    for i, v in enumerate(vars_array):

        tdef = np.isfinite(target)
        vdef = np.isfinite(v)
        bdef = np.logical_and(tdef, vdef)
        rho = np.corrcoef(v[bdef], target[bdef])
        
        cmap = discrete_cmap(N=n, base_cmap='jet')
        sc = axs[i].scatter(v, target, c=cat_num, s=2, cmap=cmap)
        axs[i].set_xlabel(vars_names_array[i])
        axs[i].set_ylabel(target_name)
        axs[i].grid()
        #axs[i].annotate('rho: {}'.format(rho[1][0].round(2)), xy=(0.7, 0.05), xycoords='axes fraction', color='black')
        axs[i].set_title('rho: {}'.format(rho[1][0].round(2)))
        
        cbar = fig.colorbar(sc, ax=axs[i])
        loc = np.linspace(tick_dif,(n-1)-tick_dif,n)
        cbar.set_ticks(loc)
        cbar.set_ticklabels([inv_map[i] for i in np.arange(n)])

    iidx = n_lines*3-(n_lines*3-len(vars_array))
    axs_to_remove = np.arange(iidx, n_lines*3)
    for i in axs_to_remove:
        axs[i].set_visible(False)


    #fig.suptitle(title)

    plt.tight_layout()

    plt.savefig(outfl, bbox_inches='tight', facecolor='white')