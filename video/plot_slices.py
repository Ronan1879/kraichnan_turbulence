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

vorticity = []

#data_folder=sys.argv[1]
# Go to path with snapshots of simulation 
#path = "../../data/dns/"+str(data_folder)
# Merge the processes from all sets of data
#post.merge_process_files("snapshots", cleanup=True)


#os.chdir(path)
i = 40
while os.path.exists("./snapshots/snapshots_s%s.h5" % i):
    i += 1

n_sets = i
print(n_sets)
for j in range(40,n_sets):
	with h5py.File("snapshots/snapshots_s{}.h5".format(j), mode='r') as file:

		w = file['tasks']['w']
		t = file['scales']['sim_time']
		vorticity.append(np.array(w)[0])

vorticity = np.array(vorticity)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

x_array = np.linspace(-1,1,vorticity.shape[1])
y_array = np.linspace(-1,1,vorticity.shape[2])
x,y = np.meshgrid(x_array,y_array)
quadmesh = ax.pcolormesh(x_array,y_array,vorticity[0,:,:],cmap='jet')
plt.colorbar(quadmesh)
#cax = ax.imshow(vorticity[0,:,:],cmap='jet')
#ax.colorbar(cax)

ax.set_xlabel("x",fontsize=14)
ax.set_ylabel("y",fontsize=14)
ax.set_title("vorticity field",fontsize=14)

def animate(i):
	quadmesh.set_array(vorticity[i,:,:].ravel())
	quadmesh.set_clim(vmin=np.min(vorticity[i,:,:])/2,vmax=np.max(vorticity[i,:,:])/2)

anim = FuncAnimation(fig, animate, interval=100, frames=n_sets-1, repeat=True,blit=False,save_count=n_sets-1)


anim.save('simulation.mp4')


