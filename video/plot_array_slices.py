"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.ioff()
from dedalus.extras import plot_tools
from dedalus.tools import post
import os
import sys

vorticity = np.load("sim_data.npy")
print(vorticity.shape)

n_sets = vorticity.shape[0]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

x_array = np.linspace(-1,1,vorticity.shape[1])
y_array = np.linspace(-1,1,vorticity.shape[2])
x,y = np.meshgrid(x_array,y_array)
quadmesh = ax.pcolormesh(x_array,y_array,vorticity[0,:,:],cmap='jet')
plt.colorbar(quadmesh)

ax.set_xlabel("x",fontsize=14)
ax.set_ylabel("y",fontsize=14)
ax.set_title("vorticity field",fontsize=14)

def animate(i):
	quadmesh.set_array(vorticity[i,:,:].ravel())
	quadmesh.set_clim(vmin=np.min(vorticity[i,:,:])/2,vmax=np.max(vorticity[i,:,:])/2)

anim = FuncAnimation(fig, animate, interval=100, frames=n_sets-1, repeat=True,blit=False,save_count=n_sets-1)


anim.save('simulation_13.mp4')


