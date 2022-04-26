import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def scatter_plot(x, y, ax, c=None):
    xdef = np.isfinite(x)
    ydef = np.isfinite(y)
    bothdef = np.logical_and(xdef, ydef)
    x, y = x[bothdef], y[bothdef]
    if c is None:
        xy = np.vstack([x, y])
        c = gaussian_kde(xy)(xy)

    corr = round(np.corrcoef([x, y])[0,1], 2)

    ax.scatter(x, y, c=c, s=1, cmap='jet')
    ax.set_title('rho: {}'.format(corr))


def scatter_matrix(vars_array, vars_name, figsize, outfl, nmax=None, cat=None):
    if nmax is not None:
        idxs = np.arange(len(vars_array[0]))
        ridxs = np.random.choice(idxs, size=nmax, replace=False)
        vars_array = [i[ridxs] for i in vars_array]
    else:
        pass

    data = np.array(vars_array)
    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=figsize)

    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:

            scatter_plot(data[x], data[y], axes[x,y], cat)
            axes[x,y].grid()
            axes[x,y].set_xlabel(vars_name[x])
            axes[x,y].set_ylabel(vars_name[y])

    for i, label in enumerate(vars_name):
        axes[i,i].hist(data[i], color='green')
        axes[i,i].set_title(label)


    plt.tight_layout()
    plt.savefig(outfl, facecolor='white', bbox_inches='tight')