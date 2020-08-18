"""Load weights and model of neural net and computes closure term for coarsed grain simulation"""

import unet
import numpy as np
import tensorflow as tf
from parameters import *

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

from misc_functions import deviatoric_part
from misc_functions import array_of_tf_components


model = unet.Unet(stacks,stack_width,filters_base,output_channels, **unet_kw)
checkpoint = tf.train.Checkpoint(net=model)

train_path = 'training_costs.npy'
output_path = 'subgrid_stress_field.png'

training_costs = np.load(train_path)

best_epoch = np.where(training_costs == np.min(training_costs))[0][0]

best_epoch -= best_epoch % save_interval

restore_path = f"{checkpoint_path}-{best_epoch//save_interval}"
checkpoint.restore(restore_path)#.assert_consumed()
print('Restored from {}'.format(restore_path))

inputs = np.load("inputs.npy")[2]
labels = np.load("labels.npy")[2]

inputs = inputs[np.newaxis,...]
labels = labels[np.newaxis,...]

tf_inputs = [tf.cast(inputs,datatype)]
tf_labels = tf.cast(labels,datatype)
tf_outputs = model.call(tf_inputs)

tau_pred = deviatoric_part(array_of_tf_components(tf_outputs))
tau_true = deviatoric_part(array_of_tf_components(tf_labels))

fig, ax = plt.subplots(3,4,figsize=(13*4,11*3))

pred_list = [tau_pred[0,0],tau_pred[1,1],tau_pred[1,0]]
true_list = [tau_true[0,0],tau_true[1,1],tau_true[1,0]]

names = [r'$\tau_{xx}$',r'$\tau_{yy}$',r'$\tau_{xy}$']

mpl.rcParams['axes.linewidth'] = 3
plt.rcParams["font.family"] = "serif"

ls = 26
scale = 0.70
for a in ax.flatten():
    for axis in ['top','bottom','left','right']:
        a.spines[axis].set_linewidth(3)
for i in range(3):
    divider1 = make_axes_locatable(ax[i,0])
    cax1 = divider1.append_axes('right', size='5%', pad=0.1)

    divider2 = make_axes_locatable(ax[i,1])
    cax2 = divider2.append_axes('right', size='5%', pad=0.1)

    divider3 = make_axes_locatable(ax[i,2])
    cax3 = divider3.append_axes('right', size='5%', pad=0.1)

    divider4 = make_axes_locatable(ax[i,3])
    cax4 = divider4.append_axes('right', size='5%', pad=0.1)

    name = names[i]

    pred_grid = pred_list[i][0]
    true_grid = true_list[i][0]
    res_grid = pred_grid - true_grid
    res_sq_grid = np.abs(res_grid)**2

    max_value = np.max(np.abs(true_grid))
    min_value = -max_value

    im1 = ax[i,0].imshow(pred_grid,cmap='RdBu_r',extent=(Bx[0],Bx[1],By[0],By[1]))
    clb1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
    ax[i,0].set_title(name+" prediction",fontsize=ls+6,y=1.02)
    ax[i,0].tick_params(direction='out',width=2,length=5,labelsize = ls)
    im1.set_clim(vmin=scale*min_value,vmax=scale*max_value)
    clb1.ax.tick_params(labelsize=ls)
    ax[i,0].set_xlabel("x",fontsize=ls)
    ax[i,0].set_ylabel("y",fontsize=ls)

    im2 = ax[i,1].imshow(true_grid,cmap='RdBu_r',extent=(Bx[0],Bx[1],By[0],By[1]))
    clb2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
    ax[i,1].set_title(name+" truth",fontsize=ls+6,y=1.02)
    ax[i,1].tick_params(direction='out',width=2,length=5,labelsize = ls)
    im2.set_clim(vmin=scale*min_value,vmax=scale*max_value)
    clb2.ax.tick_params(labelsize=ls)
    ax[i,1].set_xlabel("x",fontsize=ls)
    ax[i,1].set_ylabel("y",fontsize=ls)

    im3 = ax[i,2].imshow(res_grid,cmap='RdBu_r',extent=(Bx[0],Bx[1],By[0],By[1]))
    clb3 = fig.colorbar(im3, cax=cax3, orientation='vertical')
    ax[i,2].set_title(name+" residuals",fontsize=ls+6,y=1.02)
    ax[i,2].tick_params(direction='out',width=2,length=5,labelsize = ls)
    im3.set_clim(vmin=-scale*np.max(np.abs(res_grid)),vmax=scale*np.max(np.abs(res_grid)))
    clb3.ax.tick_params(labelsize=ls)
    ax[i,2].set_xlabel("x",fontsize=ls)
    ax[i,2].set_ylabel("y",fontsize=ls)

    im4 = ax[i,3].imshow(res_sq_grid,cmap='RdBu_r',extent=(Bx[0],Bx[1],By[0],By[1]))
    clb4 = fig.colorbar(im4, cax=cax4, orientation='vertical')
    ax[i,3].set_title(name+" residuals squared",fontsize=ls+6,y=1.02)
    ax[i,3].tick_params(direction='out',width=2,length=5,labelsize = ls)
    im4.set_clim(vmin=0,vmax=scale*np.max(np.abs(res_sq_grid)))
    clb4.ax.tick_params(labelsize=ls)
    ax[i,3].set_xlabel("x",fontsize=ls)
    ax[i,3].set_ylabel("y",fontsize=ls)

plt.savefig(output_path,bbox_inches='tight',dpi = 200)
plt.clf()


