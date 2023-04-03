"""
===============
Simple Colorbar
===============

"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import numpy as np

ax = plt.subplot()
im = ax.imshow(np.linspace(0, 1, 100).reshape((10, 10)), cmap=cm.inferno)

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.05)

plt.colorbar(im, cax=cax, orientation='horizontal')

plt.show()
