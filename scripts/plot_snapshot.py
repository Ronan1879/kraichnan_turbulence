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
import matplotlib as mpl
from matplotlib.animation import FuncAnimation

from parameters import *
import os
import sys

#plt.style.use('science')
mpl.rcParams['axes.linewidth'] = 1.5
plt.rcParams["font.family"] = "serif"
filename=sys.argv[1]

with h5py.File(str(filename), mode='r') as file:

	w = file['tasks']['wz']#ωz
	t = file['scales']['sim_time']
	w_data = np.array(w)[0]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

ax.tick_params(direction='out',width=2,length=5,labelsize = 20)
img = ax.imshow(w_data,cmap='jet',extent=(Bx[0],Bx[1],By[0],By[1]))
img.set_clim(vmin=-10,vmax=10)
clb = plt.colorbar(img)
clb.ax.tick_params(labelsize=20)

ax.set_xlabel("x",fontsize=20)
ax.set_ylabel("y",fontsize=20)
ax.set_title("Corrected t = 2 s",fontsize=22,y=1.03)

plt.savefig('snapshot.png',dpi=300,bbox_inches='tight')


