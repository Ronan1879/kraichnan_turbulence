"""Train u-net."""

import numpy as np
import tensorflow as tf
import unet
import time
from parameters import *
import csv
from misc_functions import deviatoric_part
from misc_functions import array_of_tf_components
from dedalus import public as de
import fluid_functions as ff
import xarray

# Divide training data
# Randomly permute snapshots
tf.random.set_seed(tf_seed)
rand = np.random.RandomState(seed = perm_seed)

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

def cost_function(outputs,labels):
    # Load components into object arrays, take deviatoric part of stresses
    #S_true = array_of_tf_components(inputs[0])
    tau_pred = deviatoric_part(array_of_tf_components(outputs))
    tau_true = deviatoric_part(array_of_tf_components(labels))

    # Compute cost of predicted subgrid stress tensor
    # Pointwise deviatoric stress error
    tau_d_diff = tau_true - tau_pred
    f2_tau_d_diff = np.trace(np.dot(tau_d_diff,tau_d_diff.T))
    L2_tau_d_error = tf.reduce_sum(f2_tau_d_diff)

    # Pointwise dissipation error
    #D_true = np.trace(np.dot(tau_true,S_true.T))
    #D_pred = np.trace(np.dot(tau_pred,S_true.T))
    #D_diff = D_true - D_pred
    # L2-squared dissipation error
    #L2_D_error = tf.reduce_sum(D_diff**2)

    #cost = (1-diss_cost) * L2_tau_d_error + diss_cost * L2_D_error
    cost = L2_tau_d_error

    return cost


train_path = 'training_costs.csv'
test_path = 'testing_costs.csv'

train_cost_file = open(train_path,"ab")
test_cost_file = open(test_path,"ab")

# Build network and optimizer
model = unet.Unet(stacks,stack_width,filters_base,output_channels, **unet_kw)
optimizer = tf.keras.optimizers.Adamax(learning_rate)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=20)

if resume_training == True:
    checkpoint.restore(manager.latest_checkpoint)#.assert_consumed()
    print('Restored from {}'.format(manager.latest_checkpoint))
else:
    print('Initializing from scratch.')
initial_epoch = checkpoint.save_counter.numpy() + 1

x_basis = de.Fourier('x', N_filter, interval=Bx, dealias=3/2)
y_basis = de.Fourier('y', N_filter, interval=By, dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64, comm=None)

inputs = np.zeros((snapshots,N_filter,N_filter,input_channels))
labels = np.zeros((snapshots,N_filter,N_filter,output_channels))

for i in range(snapshots):
    input_set, label_set = load_data(domain,i+1)
    inputs[i] = input_set
    labels[i] = label_set

train_inputs = inputs[:training_size]
train_labels = labels[:training_size]

test_inputs = inputs[training_size:training_size+testing_size]
test_labels = labels[training_size:training_size+testing_size]

best_cost = np.inf

start_time = time.time()
for epoch in range(initial_epoch,initial_epoch+epochs):
    print(f"Beginning epoch {epoch}", flush=True)

    # Train
    cost_epoch = 0
    rand.seed(perm_seed + epoch)

    batch_idx = rand.permutation(np.arange(train_inputs.shape[0])).reshape((-1,batch_size))
    for iteration, snapshot_num in enumerate(batch_idx):

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
                cost_epoch = tf.add(cost_epoch,cost)
            weight_grads = tape.gradient(cost,model.variables)
            optimizer.apply_gradients(zip(weight_grads,model.variables))

    # Status and outputs
    print('epoch: %i, training cost : %.3e' %(epoch, cost_epoch.numpy()), flush=True)

    np.savetxt(train_cost_file,[cost_epoch.numpy()/training_size],fmt='%1.4f')

    with tf.device(device):
        # Combine inputs to predict layer outputs
        tf_inputs = [tf.cast(test_inputs,datatype)]
        tf_labels = tf.cast(test_labels,datatype)

        tf_outputs = model.call(tf_inputs)
        cost = cost_function(tf_outputs,tf_labels)

        # Status and outputs
        print('epoch: %i, testing cost : %.3e' %(epoch, cost.numpy()), flush=True)

    # Save testing cost
    np.savetxt(test_cost_file,[cost.numpy()/testing_size],fmt='%1.4f')

    if best_cost > cost.numpy():
        model.save_weights(checkpoint_path + '/best_epoch')
        best_cost = cost.numpy()
    manager.save()


end_time = time.time()

print("Total training time: %.3e" % (end_time-start_time))

train_cost_file.close()
test_cost_file.close()