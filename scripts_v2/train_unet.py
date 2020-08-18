"""Train u-net."""

import numpy as np

import tensorflow as tf
import unet
from parameters import *
from misc_functions import deviatoric_part
from misc_functions import array_of_tf_components

import os
from filter_functions import *
from post import *
from create_uxuy import *
import xarray
import fluid_functions as ff

# Divide training data
# Randomly permute snapshots
rand = np.random.RandomState(seed = perm_seed)


def create_inputs_labels(Nsnapshots):
    domain = build_domain(params)
    filter = build_gaussian_filter(domain,N,epsilon)

    inputs = np.zeros((Nsnapshots,N,N,input_channels))
    labels = np.zeros((Nsnapshots,N,N,output_channels))

    for i in range(Nsnapshots):
        ux,uy = create_random_uxuy(domain)

        filt_ux = filter(ux).evaluate()
        filt_uy = filter(uy).evaluate()
        input_fields = ff.uxuy_derivatives(domain,filt_ux,filt_uy)
        label_fields = ff.implicit_subgrid_stress(domain,filter,ux,uy)
        
        for input_field in input_fields:
            input_field.set_scales(N/Nx)
        for label_field in label_fields:
            label_field.set_scales(N/Nx)
        
        input_list = [input_field['g'] for input_field in input_fields]
        label_list = [label_field['g'] for label_field in label_fields]

        # Reshape as (batch, *shape, channels)
        inputs[i,:,:,:] = np.moveaxis(np.array(input_list), 0, -1)[None]
        labels[i,:,:,:] = np.moveaxis(np.array(label_list), 0, -1)[None]

    return inputs, labels

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

    return L2_tau_d_error

# Build network and optimizer
model = unet.Unet(stacks,stack_width,filters_base,output_channels, **unet_kw)
optimizer = tf.keras.optimizers.Adam(learning_rate)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=model)

if restore_counter:
    restore_path = f"{checkpoint_path}-{restore_counter}"
    checkpoint.restore(restore_path)#.assert_consumed()
    print('Restored from {}'.format(restore_path))
else:
    print('Initializing from scratch.')
initial_epoch = save_interval*checkpoint.save_counter.numpy() + 1

num_batches = training_size//batch_size

train_inputs, train_labels = create_inputs_labels(training_size)
test_inputs, test_labels = create_inputs_labels(testing_size)

train_path = 'training_costs.npy'
test_path = 'testing_costs.npy'

if os.path.exists(train_path):
    training_costs = np.load(train_path)[:initial_epoch].tolist()
    print("Loaded training costs list")
else:
    training_costs = []

if os.path.exists(test_path):
    testing_costs = np.load(test_path)[:initial_epoch].tolist()
    print("Loaded testing costs list")
else:
    testing_costs = []


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
            print('epoch.iter.snapshots: {}.{}.{}, training cost : {}'.format(epoch, iteration, snapshot_num, cost.numpy()), flush=True)

    # Save training cost
    training_costs.append(cost_epoch/num_batches)

    with tf.device(device):
        # Combine inputs to predict layer outputs
        tf_inputs = [tf.cast(test_inputs,datatype)]
        tf_labels = tf.cast(test_labels,datatype)

        tf_outputs = model.call(tf_inputs)
        cost = cost_function(tf_outputs,tf_labels)
        
        # Status and outputs
        print('epoch : {}, testing cost : {}'.format(epoch, cost.numpy()), flush=True)
    
    # Save testing cost
    testing_costs.append(cost)

    if epoch % save_interval == 0:
        print("Saving weights.", flush=True)
        checkpoint.save(checkpoint_path)
        np.save(train_path,np.array(training_costs))
        np.save(test_path,np.array(testing_costs))


