import matplotlib.pyplot as plt
import numpy as np
 
def tdscatter(x, y, z, c, zex, figsize, title, outfl):

    cdef = np.isfinite(c)
    c = c[cdef]
    x, y, z = x[cdef], y[cdef], z[cdef]
 
    fig = plt.figure(figsize = figsize)
    ax = plt.axes(projection ="3d")
    
    ax.set_box_aspect((np.ptp(x), np.ptp(y), zex*np.ptp(z)))  # aspect ratio is 1:1:1 in data space

    ax.scatter3D(x, y, z, c = c, s=2, cmap='jet')

    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_zlabel('Elevation (m)')

    ax.set_title(title)
    
    plt.savefig(outfl, bbox_inches='tight', facecolor='white')