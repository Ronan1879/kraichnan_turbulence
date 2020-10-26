"""Load weights and model of neural net and computes closure term for coarsed grain simulation"""

import unet
import numpy as np
import tensorflow as tf
from parameters import *
from dedalus import public as de
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from misc_functions import deviatoric_part
from misc_functions import array_of_tf_components
import xarray
import fluid_functions as ff


def load_data(domain,savenum):

    ux_field = domain.new_field(name = 'ux')
    uy_field = domain.new_field(name = 'uy')
    
    filename = 'filtered/snapshots_s%i.nc' %savenum
    dataset = xarray.open_dataset(filename)
    comps = ['xx', 'yy','xy']
    ux = dataset['ux'].data
    uy = dataset['uy'].data
    
    ux_field['g'] = ux
    uy_field['g'] = uy

    # Compute inputs from velocity fields
    du_field = ff.uxuy_derivatives(domain,ux_field,uy_field)

    # Velocity field derivatives
    du = [du['g'] for du in du_field]
    # Implicit subgrid stress
    tau = [dataset['im_t'+c].data for c in comps]
    # Deviatoric subgrid stress
    tr_tau = tau[0] + tau[1]
    tau[0] = tau[0] - tr_tau/2
    tau[1] = tau[1] - tr_tau/2
    # Reshape as (batch, *shape, channels)
    input_set = np.moveaxis(np.array(du), 0, -1)[None]
    label_set = np.moveaxis(np.array(tau), 0, -1)[None]

    return input_set, label_set 

model = unet.Unet(stacks,stack_width,filters_base,output_channels, **unet_kw)
checkpoint = tf.train.Checkpoint(net=model)

checkpoint.restore(checkpoint_path + '/best_epoch')#.assert_consumed()

x_basis = de.Fourier('x', N_filter, interval=Bx, dealias=3/2)
y_basis = de.Fourier('y', N_filter, interval=By, dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64, comm=None)

dx = domain.bases[0].Differentiate
dy = domain.bases[1].Differentiate

kx = domain.elements(0)
ky = domain.elements(1)
k = np.sqrt(kx**2 + ky**2)

input_set, label_set = load_data(domain,66)

tf_input = [tf.cast(input_set,datatype)]
tf_label = tf.cast(label_set,datatype)
tf_output = model.call(tf_input)

tau_pred = deviatoric_part(array_of_tf_components(tf_output))
tau_true = deviatoric_part(array_of_tf_components(tf_label))

fig, ax = plt.subplots(3,4,figsize=(13*4,11*3))

pred_list = [tau_pred[0,0][0],tau_pred[1,1][0],tau_pred[1,0][0]]
true_list = [tau_true[0,0][0],tau_true[1,1][0],tau_true[1,0][0]]

names = [r'$\tau_{xx}$',r'$\tau_{yy}$',r'$\tau_{xy}$']

mpl.rcParams['axes.linewidth'] = 3
plt.rcParams["font.family"] = "serif"

ls = 26
scale = 0.70
for a in ax.flatten():
    for axis in ['top','bottom','left','right']:
        a.spines[axis].set_linewidth(3)
for i in range(3):

    name = names[i]
    results = [pred_list[i], true_list[i], pred_list[i] - true_list[i], np.abs(pred_list[i] - true_list[i])**2]

    for j in range(4):
        cax = make_axes_locatable(ax[i,j]).append_axes('right', size='5%', pad=0.1)
        im1 = ax[i,j].imshow(results[j],cmap = 'RdBu_r',extent = (Bx[0],Bx[1],By[0],By[1]))
        clb = fig.colorbar(im1, cax = cax, orientation = 'vertical')
        ax[i,j].set_title(name + " prediction",fontsize = ls + 6, y = 1.02)
        ax[i,j].tick_params(direction = 'out',width = 2,length = 5,labelsize = ls)
        im1.set_clim(vmin = -scale*np.max(results[j]),vmax = scale*np.max(results[j]))
        clb.ax.tick_params(labelsize=ls)
        ax[i,j].set_xlabel("x",fontsize=ls)
        ax[i,j].set_ylabel("y",fontsize=ls)

plt.savefig('sst_im.png',bbox_inches='tight')


