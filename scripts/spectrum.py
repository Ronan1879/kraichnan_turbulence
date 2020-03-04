"""
Generate power spectrum of fluid simulation

Usage:
    spectrum.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]
"""
"""
Generate power spectrum of the kinetic energy of fluid simulation
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import post as post_tools
import sim_parameters as params
import filter


load_num = np.arange(1,40)

domain = post_tools.build_domain(params, comm=None)

kx = domain.elements(0)
ky = domain.elements(1)

dim_kx = kx.shape[0]
dim_ky = ky.shape[1]

# Define arrays to store fields in order to compute average
fields_ux = np.zeros((dim_kx,dim_ky,len(load_num)))
fields_uy = np.zeros((dim_kx,dim_ky,len(load_num)))
#fields_ux_filt = np.zeros((dim_kx,dim_ky,len(load_num)))
#fields_uy_filt = np.zeros((dim_kx,dim_ky,len(load_num)))



# Set fourier cutoff (i.e N = 256 is equivalent to reducing the fields to 256 by 256 pixels)
N = 256
threshold = 10**(-30)
#filt = filter.build_filter(domain, N)

for i in load_num:
	# Load fields
	filename = "simulation_14/snapshots/snapshots_s{}.h5".format(i)
	temp_ux = post_tools.load_field(filename, domain, 'ux', 0)
	temp_uy = post_tools.load_field(filename, domain, 'uy', 0)

	# Apply filtering on fields
	#filt_ux = filt(temp_ux).evaluate()
	#filt_uy = filt(temp_uy).evaluate()

	#fields_ux_filt[:,:,i-np.min(load_num)] = np.abs(filt_ux['c'])
	#fields_uy_filt[:,:,i-np.min(load_num)] = np.abs(filt_uy['c'])
	fields_ux[:,:,i-np.min(load_num)] = np.abs(temp_ux['c'])
	fields_uy[:,:,i-np.min(load_num)] = np.abs(temp_uy['c'])

# Take the mean of the coefficient spaces snapshots taken at different times
ux_coef = np.mean(fields_ux,axis=2)
uy_coef = np.mean(fields_uy,axis=2)
#ux_coef_filt = np.mean(fields_ux_filt,axis=2)
#uy_coef_filt = np.mean(fields_uy_filt,axis=2)




k = np.array(np.meshgrid(ky,kx))#[:,:,:(ky.shape[1]//2 + 1)]
#print(k.shape)
k = np.sqrt(np.abs(k[0])**2 + np.abs(k[1])**2)

ps = (np.abs(ux_coef)**2 + np.abs(uy_coef)**2)/2
#ps_filt = (np.abs(ux_coef_filt)**2 + np.abs(uy_coef_filt)**2)/2

n_count,bins = np.histogram(k.flatten(),bins=1000*params.N)

n, bins = np.histogram(k.flatten(),bins=1000*params.N,weights=ps.flatten())
#n_filt, bins_filt = np.histogram(k.flatten(),bins=1000*params.N,weights=ps_filt.flatten())

ind = np.where(n < threshold)
n = np.delete(n,ind)
bins = np.delete(bins,ind)
n_counts = np.delete(n_count,ind)

#ind_filt = np.where(n_filt < threshold)
#n_filt = np.delete(n_filt,ind_filt)
#bins_filt = np.delete(bins_filt,ind_filt)
#n_counts_filt = np.delete(n_count,ind_filt)

fig = plt.figure()
ax = plt.gca()

#np.save('filt_ps.npy',en_list_filt)
np.save('ps_'+str(params.N)+'x'+str(params.N)+'.npy',n/n_counts)

ax.plot(bins[:-1],n/n_counts)
#ax.plot(bins_filt[:-1],n_filt/n_counts_filt)
#ax.plot(k_list_filt[::100],en_list_filt[::100])
ax.set_ylabel(r'$E(k)$')
ax.set_xlabel(r'$k$')
ax.set_xscale('log')
ax.set_yscale('log')
#plt.savefig("test_E_PS_mean.png",dpi=300)
plt.show()

