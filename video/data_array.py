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
i = 1
while os.path.exists("./snapshots/snapshots_s%s.h5" % i):
    i += 1

n_sets = i
print(n_sets)
for j in range(1,n_sets):
	with h5py.File("snapshots/snapshots_s{}.h5".format(j), mode='r') as file:

		w = file['tasks']['w']
		t = file['scales']['sim_time']
		vorticity.append(np.array(w)[0])

vorticity = np.array(vorticity)
np.save("sim_data.npy",vorticity)


