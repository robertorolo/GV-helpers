import matplotlib.pyplot as plt
import numpy as np
 
def tdscatter(x, y, z, c, zex, outfl):

    cdef = np.isfinite(c)
    c = c[cdef]
    x, y, z = x[cdef], y[cdef], z[cdef]
 
    fig = plt.figure(figsize = (15, 15))
    ax = plt.axes(projection ="3d")
    
    ax.set_box_aspect((np.ptp(x), np.ptp(y), zex*np.ptp(z)))  # aspect ratio is 1:1:1 in data space

    ax.scatter3D(x, y, z, c = c, s=2, cmap='jet')

    plt.tight_layout()
    
    plt.savefig(outfl, bbox_inches='tight', facecolor='white')