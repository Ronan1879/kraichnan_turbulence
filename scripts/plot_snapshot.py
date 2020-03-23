"""
Create png image of simulation snapshots.

Type in command line the name of folder containing the snapshot data

Example : python plot_slices.py corrected_simulation/snapshots/snapshots_s1.h5

Usage:
    plot_slices.py <file>
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from parameters import *
import os
import sys

filename=sys.argv[1]

with h5py.File(str(filename), mode='r') as file:

	w = file['tasks']['wz']#ωz
	t = file['scales']['sim_time']
	w_data = np.array(w)[0]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

img = ax.imshow(w_data,cmap='jet',extent=(Bx[0],Bx[1],By[0],By[1]))
plt.colorbar(img)

ax.set_xlabel("x",fontsize=14)
ax.set_ylabel("y",fontsize=14)
ax.set_title("vorticity field",fontsize=14)

plt.savefig('snapshot.png',dpi=300)


