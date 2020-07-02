"""Train u-net."""

import numpy as np
import tensorflow as tf
import leith_experiment_model_unet as unet
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dedalus import public as de
from parameters import *

def snapshot_generator(Nsnapshots):

    inputs = np.zeros((Nsnapshots,Nx,Ny,1))
    labels = np.zeros((Nsnapshots,Nx,Ny,output_channels))
    # Create bases and domain
    x_basis = de.Fourier('x', Nx, interval = Bx, dealias=3/2)
    y_basis = de.Fourier('y', Ny, interval = By, dealias=3/2)
    domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

    x = domain.grid(0)
    y = domain.grid(1)
    kx = domain.elements(0)
    ky = domain.elements(1)
    k = np.array(np.meshgrid(ky,kx))
    k = np.sqrt(k[0]**2 + k[1]**2)
    
    wz = domain.new_field(name='wz')
    norm_grad_wz = domain.new_field(name='wz')

    
    for s in np.arange(Nsnapshots):
        
        # Generate random normal field with fourier scaling that dampens higher k modes
        wz['g'] = np.random.normal(0,1,(x.shape[0],y.shape[1]))
        # The + 1 term is to avoid division by zero
        wz['c'] = wz['c']*(k + 1)**k_scaling
        wz['g'] = (wz['g'] - np.mean(wz['g']))/np.std(wz['g'])

        # Set higher grid scale to perform differentiation
        wz.set_scales(3/2)
        dx = domain.bases[0].Differentiate
        dy = domain.bases[1].Differentiate
        norm_grad_wz = np.sqrt(dx(wz)**2 + dy(wz)**2).evaluate()

        # Set grid scale back to original size
        wz.set_scales(1)
        norm_grad_wz.set_scales(1)
        inputs[s,:,:,0] = wz['g']
        labels[s,:,:,0] = norm_grad_wz['g']

    return inputs, labels

def cost_function(outputs,labels):
    cost = tf.math.reduce_sum(tf.math.reduce_sum((outputs-labels)**2,axis=(1,2)),axis=0)

    return cost

checkpoint_path = "unet_checkpoints/"

rand = np.random.RandomState(seed=perm_seed)

# Build network and optimizer

#tf.random.set_seed(tf_seed)
model = unet.Unet(stacks,stack_width,filters_base,output_channels, **unet_kw)
optimizer = tf.keras.optimizers.Adam(learning_rate)

model.load_weights(checkpoint_path + "checkpoint_990")

# Create plot of field of prediction vs truth

valid_input, valid_label= snapshot_generator(Nsnapshots = 1)

valid_predict = model.call([valid_input])

fig, ax = plt.subplots(1,3,figsize=(8*3,6))

ax[0].scatter(valid_predict,valid_label,color='r',alpha=0.02)
ax[0].set_ylabel("True",fontsize=14)
ax[0].set_xlabel("Predictions",fontsize=14)

divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes('right', size='5%', pad=0.05)

divider2 = make_axes_locatable(ax[2])
cax2 = divider2.append_axes('right', size='5%', pad=0.05)

im1 = ax[1].pcolor(valid_predict[0,:,:,0])

fig.colorbar(im1, cax=cax1, orientation='vertical')

ax[1].set_title("Prediction",fontsize=14)

ax[2].set_title("True",fontsize=14)
im2 = ax[2].pcolor(valid_label[0,:,:,0])

fig.colorbar(im2, cax=cax2, orientation='vertical')

plt.savefig("unet_fields.png",dpi=200,bbox_inches='tight')