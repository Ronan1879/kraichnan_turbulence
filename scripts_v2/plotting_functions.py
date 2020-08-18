import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from parameters import *

mpl.rcParams['axes.linewidth'] = 1.5
plt.rcParams["font.family"] = "serif"

c_scale = 0.7
#data_name = 'wz'
ls = 20
cmap = 'RdBu_r'


def plot_snapshot(filename,data_name):

	fig, ax = plt.subplots(1,1,figsize=(10,8))
	ax.tick_params(direction='out',width=2,length=5,labelsize = 20)
	ax.set_title(data_name + " prediction",fontsize=ls+6,y=1.02)
    ax.tick_params(direction='out',width=2,length=5,labelsize = ls)
    ax.set_xlabel("x",fontsize=ls)
    ax.set_ylabel("y",fontsize=ls)

	with h5py.File(str(filename), mode='r') as file:
		data = file['tasks'][data_name]
		data = np.array(data)[0]

	max_value = np.max(np.abs(data))

    im = ax.imshow(data,cmap=cmap,extent=(Bx[0],Bx[1],By[0],By[1]))
    clb = fig.colorbar(im, cax=cax, orientation='vertical')

    im.set_clim(vmin=c_scale*min_value,vmax=c_scale*max_value)
    clb.ax.tick_params(labelsize=ls)


def plot_snapshot_movie(snapshot_folder,data_name):

	fig, ax = plt.subplots(1,1,figsize=(10,8))
	ax.tick_params(direction='out',width=2,length=5,labelsize = 20)
	ax.set_title(data_name + " prediction",fontsize=ls+6,y=1.02)
    ax.tick_params(direction='out',width=2,length=5,labelsize = ls)
    ax.set_xlabel("x",fontsize=ls)
    ax.set_ylabel("y",fontsize=ls)

	data_set = []
	Nsnapshots = stop_iteration/snapshots_iter
	filename = "/snapshots_s{}.h5"

	for i in range(Nsnapshots)

		with h5py.File(snapshot_folder + filename.format(i), mode='r') as file:
			data = file['tasks'][data_name]
			data_set.append(np.array(data)[0])
		
		#data = xarray.open_dataset(snapshot_folder + filename.format(i))
		#data_set.append(data[data_name].data)


	data_set = np.array(data_set)
	max_value = np.max(np.abs(data[0]))

    im = ax.imshow(data,cmap=cmap,extent=(Bx[0],Bx[1],By[0],By[1]))

    clb = fig.colorbar(im, cax=cax, orientation='vertical')
    im.set_clim(vmin=c_scale*min_value,vmax=c_scale*max_value)
    clb.ax.tick_params(labelsize=ls)

	def animate(i):
		img.set_array(data_set[i,:,:])

	anim = FuncAnimation(fig, animate, interval=100, frames=Nsnapshots, repeat=True,blit=False,save_count=Nsnapshots)
	anim.save('simulation_movie.mp4',codec='mpeg4',bitrate=800000)

def plot_ene_pspec(k,ux,uy):

	threshold = 1e-25

	pspec = (np.abs(ux['c'])**2 + np.abs(uy['c'])**2)/2

	n_count,bins = np.histogram(k.flatten(),bins=2*N)

	n, bins = np.histogram(k.flatten(),bins=2*N,weights=pspec.flatten())

	# Remove data points where the power is below threshold. This arises from plotting issues
	bad_indices = np.where(n < threshold)[0]
	n = np.delete(n,bad_indices)
	n_count = np.delete(n_count,bad_indices)
	bins = np.delete(bins,bad_indices)

	# Save power spectrum data
	#np.save('pspec_data.npy',n)

	fig, ax = plt.subplots(1,1,figsize=(10,8))
	ax.plot(bins[:-1],n/n_count)
	ax.set_ylabel("$E(k)$",fontsize=14)
	ax.set_xlabel("$k$",fontsize=14)
	ax.set_xscale("log")
	ax.set_yscale("log")
	plt.savefig("spectrum.png",dpi=300)


