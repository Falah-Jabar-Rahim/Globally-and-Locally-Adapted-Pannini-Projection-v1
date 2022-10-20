import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


def mesh_plt(x, y, name):

    plt.plot(x, y,
             color='k',  # All points are set to red
             marker='.',  # The shape of the dot is a dot
             linestyle='')  # The line type is empty, that is, there is no line connection between points
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(name)


    segs1 = np.stack((x[:,[0,-1]],y[:,[0,-1]]), axis=2)
    segs2 = np.stack((x[[0,-1],:].T,y[[0,-1],:].T), axis=2)
    plt.gca().add_collection(LineCollection(np.concatenate((segs1, segs2))))
    plt.autoscale()
    plt.show()

