from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
 
def tdscatter(x, y, z, c, zex, figsize, title, outfl):

    cdef = np.isfinite(c)
    c = c[cdef]
    x, y, z = x[cdef], y[cdef], z[cdef]
    k = len(np.unique(c))
    cmap = plt.cm.get_cmap('jet', k)
 
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_box_aspect((np.ptp(x), np.ptp(y), zex*np.ptp(z)))  # aspect ratio is 1:1:1 in data space

    #ts = ax.scatter3D(x, y, z, c = c, s=2, cmap=cmap)
    ts = ax.scatter(x, y, z, c=c, s=2, cmap=cmap)
    
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_zlabel('Elevation (m)')

    ax.set_title(title)

    tick_locs = (np.arange(k) + 0.5)*(k-1)/k
    cbar = fig.colorbar(ts, ax=ax)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(np.arange(k))
    
    fig.tight_layout()
    fig.savefig(outfl, bbox_inches='tight', facecolor='white')