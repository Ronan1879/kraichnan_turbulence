"""Train u-net."""

import numpy as np
import xarray
import tensorflow as tf
import leith_experiment_model_unet as unet
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dedalus import public as de
from parameters import *
import time
import os

def snapshot_generator(Nsnapshots):

    inputs = np.zeros((Nsnapshots,Nx,Ny,input_channels))
    labels = np.zeros((Nsnapshots,Nx,Ny,output_channels))

    # Create bases and domain
    x_basis = de.Fourier('x', Nx, interval = Bx, dealias=3/2)
    y_basis = de.Fourier('y', Ny, interval = By, dealias=3/2)
    domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

    x = domain.grid(0)
    y = domain.grid(1)
    kx = domain.elements(0)
    ky = domain.elements(1)
    k = np.sqrt(kx**2 + ky**2)

    wz = domain.new_field(name='wz')
    norm_grad_wz = domain.new_field(name='wz')

    for s in np.arange(Nsnapshots):
        
        # Generate random normal field with fourier scaling that dampens higher k modes
        wz['g'] = np.random.normal(0,1,(x.shape[0],y.shape[1]))
        # The + 1 term is to avoid division by zero
        wz['c'] = wz['c']*(k + (k == 0))**k_scaling
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

training_costs = []
testing_costs = []

rand = np.random.RandomState(seed=perm_seed)

# Build network and optimizer
#tf.random.set_seed(tf_seed)
model = unet.Unet(stacks,stack_width,filters_base,output_channels, **unet_kw)
optimizer = tf.keras.optimizers.Adam(learning_rate)

#model.load_weights(checkpoint_path + "checkpoint_")

# Generate training and testing inputs and labels
train_inputs , train_labels = snapshot_generator(Nsnapshots=training_size)
#test_inputs , test_labels = data_label_generator(Nsnapshots=testing_size)

train_idx = np.arange(train_inputs.shape[0])
#test_idx = np.arange(test_inputs.shape[0])

for epoch in range(epochs):
    print(f"Beginning epoch {epoch}", flush=True)

    batched_train_idx = rand.permutation(train_idx).reshape((-1,batch_size))
    #Train
    for iteration, snapshot_num in enumerate(batched_train_idx):

        # Load adjascent outputs
        inputs_0 = train_inputs[snapshot_num]
        labels_0 = train_labels[snapshot_num]

        with tf.device(device):
            # Combine inputs to predict layer outputs
            tf_inputs = [tf.cast(inputs_0,datatype)]
            tf_labels = tf.cast(labels_0,datatype)

            with tf.GradientTape() as tape:
                tape.watch(model.variables)
                tf_outputs = model.call(tf_inputs)
                cost = cost_function(tf_outputs,tf_labels)
            weight_grads = tape.gradient(cost,model.variables)
            optimizer.apply_gradients(zip(weight_grads,model.variables))

            # Status and outputs
            print('epoch.iter.save: {}.{}.{}, training cost : {}'.format(epoch, iteration, snapshot_num, cost.numpy()), flush=True)
    
            training_costs.append(cost)
    if epoch % 10 == 0:
        model.save_weights(checkpoint_path + "checkpoint_{}".format(epoch))
        print("weights saved")
    """
    for iteration, snapshot_num in enumerate(rand.permutation(snapshots_test)):
        # Load adjascent outputs
        inputs_0 = test_inputs[snapshot_num][np.newaxis,...]
        labels_0 = test_labels[snapshot_num][np.newaxis,...]
        
        with tf.device(device):
            # Combine inputs to predict layer outputs
            tf_inputs = [tf.cast(inputs_0,datatype)]
            tf_labels = tf.cast(labels_0,datatype)
            tf_outputs = model.call(tf_inputs)
            cost = cost_function(tf_outputs,tf_labels)

            # Status and outputs
            print('epoch.iter.save: %i.%i.%i, testing cost : %.3e' %(epoch, iteration, snapshot_num, cost.numpy()), flush=True)

            testing_costs.append(cost)
    """
# Create plot of learning curves and correlation of truth and predictions on newly generated data

inputs , labels = snapshot_generator(Nsnapshots=1)

tf_inputs = [tf.cast(inputs,datatype)]
tf_labels = tf.cast(labels,datatype)

tf_outputs = model.call(tf_inputs)

fig, ax = plt.subplots(1,1,figsize=(8,6))

ax.plot(np.array(training_costs)/batch_size,color='b',label="Training")
#ax.plot(np.array(testing_costs),color='r',label="Testing")
ax.set_ylabel("Cost",fontsize=14)
ax.set_xlabel("Epochs",fontsize=14)
ax.legend()
plt.savefig("unet_cost.png",dpi=200)
plt.clf()