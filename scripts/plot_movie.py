"""
Create movie using simulation snapshots.

Type in command the name of folder containing snapshots

Example : python plot_slices.py corrected_simulation/snapshots

Usage:
    plot_slices.py <data_folder>
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from parameters import *
import os
import sys

vorticity = []

data_folder=sys.argv[1]

i = 1
while os.path.exists(str(data_folder) + "/snapshots_s%s.h5" % i):
    i += 1

n_sets = i
for j in range(1,n_sets):
	with h5py.File(str(data_folder) + "/snapshots_s{}.h5".format(j), mode='r') as file:

		w = file['tasks']['wz']#ωz
		t = file['scales']['sim_time']
		vorticity.append(np.array(w)[0])


vorticity = np.array(vorticity)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

img = ax.imshow(vorticity[0,:,:],cmap='jet',extent=(Bx[0],Bx[1],By[0],By[1]))
plt.colorbar(img)

ax.set_xlabel("x",fontsize=14)
ax.set_ylabel("y",fontsize=14)
ax.set_title("vorticity field",fontsize=14)

def animate(i):
	img.set_array(vorticity[i,:,:])
	img.set_clim(vmin=np.min(vorticity[i,:,:])/2,vmax=np.max(vorticity[i,:,:])/2)

anim = FuncAnimation(fig, animate, interval=100, frames=n_sets-1, repeat=True,blit=False,save_count=n_sets-1)

anim.save('simulation_movie.mp4')


