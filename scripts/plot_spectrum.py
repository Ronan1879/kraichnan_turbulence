"""
Plots power spectrum of fluid simulation

Type in command the name of snapshot file to plot power spectrum

Ex : python spectrum.py snapshots/snapshots_s1.h5

Usage:
    python spectrum.py <file>

"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import post as post_tools
import parameters as params

filename = str(sys.argv[1])

threshold = 10**(-30)
domain = post_tools.build_domain(params, comm=None)

kx = domain.elements(0)
ky = domain.elements(1)

ux = post_tools.load_field(filename, domain, 'ux', 0)
uy = post_tools.load_field(filename, domain, 'uy', 0)
ux_coef = np.abs(ux['c'])
uy_coef = np.abs(uy['c'])

k = np.array(np.meshgrid(ky,kx))
k = np.sqrt(np.abs(k[0])**2 + np.abs(k[1])**2)

pspec = (np.abs(ux_coef)**2 + np.abs(uy_coef)**2)/2

n_count,bins = np.histogram(k.flatten(),bins=params.N)

n, bins = np.histogram(k.flatten(),bins=params.N,weights=pspec.flatten())

# Remove data points where the power is below threshold. This arises from plotting issues
bad_indices = np.where(n < threshold)[0]
n = np.delete(n,bad_indices)
n_count = np.delete(n_count,bad_indices)
bins = np.delete(bins,bad_indices)

# Save power spectrum data
np.save('pspec_data.npy',n)

plt.plot(bins[:-1],n/n_count)
plt.ylabel("$E(k)$",fontsize=14)
plt.xlabel("$k$",fontsize=14)
plt.xscale("log")
plt.yscale("log")
plt.savefig("spectrum.png",dpi=300)