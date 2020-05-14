"""Train u-net."""

import numpy as np
import xarray
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf
import unet
import time
tf.enable_eager_execution()
#tf.executing_eagerly()
from parameters import *
from closure_term import *
import os


# Randomly permute snapshots
rand = np.random.RandomState(seed = perm_seed)
snapshots_perm = 1 + rand.permutation(snapshots)

# Select testing data
snapshots_test = snapshots_perm[:testing_size]
# Select training data
snapshots_train = snapshots_perm[testing_size:training_size+testing_size]

if os.path.exists("training_info.txt") == False:
    text_file = open("training_info.txt","a")
    text_file.write("training snapshots number : "+str(snapshots_train)+'\n')
    text_file.write("testing snapshots number : "+str(snapshots_test)+'\n')
    text_file.close()

# Data loading
def load_data(savenum):
    filename = 'filtered/snapshots_s%i.nc' %savenum
    dataset = xarray.open_dataset(filename)
    comps = ['xx', 'yy', 'xy']

    # Strain rate
    S = [dataset['S'+c].data for c in comps]

    grad_w_norm = dataset['grad_w_norm'].data
    eddy_visc = (leith_coef * grid_size)**3 * grad_w_norm
    # Leith model
    L = [2*eddy_visc*dataset['S'+c].data for c in comps]
    
    # Reshape as (batch, *shape, channels)
    inputs = np.moveaxis(np.array(S), 0, -1)[None]
    labels = np.moveaxis(np.array(L), 0, -1)[None]
    return inputs, labels

# Build network and optimizer
tf.set_random_seed(tf_seed)
#tf.random.set_seed(tf_seed)
model = unet.Unet(stacks,stack_width,filters_base,output_channels, **unet_kw)
optimizer = tf.train.AdamOptimizer(learning_rate)
#optimizer = tf.optimizers.Adam(learning_rate)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=model)
if restore_epoch:
    restore_path = f"{checkpoint_path}-{restore_epoch}"
    checkpoint.restore(restore_path)#.assert_consumed()
    print('Restored from {}'.format(restore_path))
else:
    print('Initializing from scratch.')
initial_epoch = checkpoint.save_counter.numpy() + 1


def array_of_tf_components(tf_tens):
    """Create object array of tensorflow packed tensor components."""
    # Collect components
    # Tensorflow shaped as (batch, *shape, channels)
    comps = ['xx', 'yy', 'xy']
    c = {comp: tf_tens[..., n] for n, comp in enumerate(comps)}
    c['yx'] = c['xy']
    # Build object array
    tens_array = np.array([[None, None],
                           [None, None]], dtype=object)
    for i, si in enumerate(['x', 'y']):
        for j, sj in enumerate(['x', 'y']):
            tens_array[i, j] = c[si+sj]
    return tens_array

def deviatoric_part(tens):
    """Compute deviatoric part of tensor."""
    tr_tens = np.trace(tens)
    tens_d = tens.copy()
    N = tens.shape[0]
    for i in range(N):
        tens_d[i, i] = tens[i, i] - tr_tens / N
    return tens_d

def cost_function(inputs,outputs,labels):
    # Load components into object arrays, take deviatoric part of stresses
    S_true = array_of_tf_components(inputs[0])
    tau_pred = deviatoric_part(array_of_tf_components(outputs))
    tau_true = deviatoric_part(array_of_tf_components(labels))

    # Needs some work
    # Compute cost of predicted closure term for vorticity formalism of the Navier-Stokes equation
    pi_pred = get_pi(tau_pred)
    pi_true = get_pi(tau_true)
    pi_diff = pi_true - pi_pred
    pi_cost = tf.reduce_sum(pi_diff**2)**0.5

    # Compute cost of predicted subgrid stress tensor
    # Pointwise deviatoric stress error
    tau_d_diff = tau_true - tau_pred
    f2_tau_d_diff = np.trace(np.dot(tau_d_diff,tau_d_diff.T))
    L2_tau_d_error = tf.reduce_mean(f2_tau_d_diff)

    # Pointwise dissipation error
    D_true = np.trace(np.dot(tau_true,S_true.T))
    D_pred = np.trace(np.dot(tau_pred,S_true.T))
    D_diff = D_true - D_pred
    # L2-squared dissipation error
    L2_D_error = tf.reduce_mean(D_diff**2)

    cost = (1-diss_cost) * L2_tau_d_error + diss_cost * L2_D_error

    return cost, pi_cost

# Learning loop
training_costs = []
testing_costs = []
training_costs_pi = []
testing_costs_pi = []

for epoch in range(initial_epoch,initial_epoch+epochs):
    print(f"Beginning epoch {epoch}", flush=True)
    # Train
    cost_epoch = 0
    cost_epoch_pi = 0
    rand.seed(perm_seed + epoch)
    
    for iteration, savenum in enumerate(rand.permutation(snapshots_train)):
        # Load adjascent outputs
        inputs_0, labels_0 = load_data(savenum)
        with tf.device(device):
            # Combine inputs to predict layer outputs
            tf_inputs = [tf.cast(inputs_0,datatype)]
            tf_labels = tf.cast(labels_0,datatype)
            with tf.GradientTape() as tape:
                tape.watch(model.variables)
                tf_outputs = model.call(tf_inputs)
                cost, cost_pi = cost_function(tf_inputs,tf_outputs,tf_labels)
                cost_epoch = tf.add(cost,cost_epoch)       
                cost_epoch_pi = tf.add(cost_pi,cost_epoch_pi)
            
            # Status and outputs
            print('epoch.iter.save: %i.%i.%i, training cost (Pi): %.3e' %(epoch, iteration, savenum, cost_pi.numpy()), flush=True)

    #cost_epoch = tf.divide(cost_epoch,training_size)
    #cost_epoch_pi = tf.divide(cost_epoch_pi,training_size)
    weight_grads = tape.gradient(cost_epoch,model.variables)
    optimizer.apply_gradients(zip(weight_grads,model.variables), global_step = tf.compat.v1.train.get_or_create_global_step())

    print("Saving weights.", flush=True)
    checkpoint.save(checkpoint_path)

    training_costs_pi.append(cost_epoch_pi/training_size)
    training_costs.append(cost_epoch/training_size)

    # Test
    cost_epoch = 0
    cost_epoch_pi = 0
    for iteration, savenum in enumerate(snapshots_test):
        # Load adjascent outputs
        inputs_0, labels_0 = load_data(savenum)

        with tf.device(device):
            # Combine inputs to predict later output
            tf_inputs = [tf.cast(inputs_0,datatype)]
            tf_labels = tf.cast(labels_0,datatype)
            tf_outputs = model.call(tf_inputs)
            cost, cost_pi = cost_function(tf_inputs,tf_outputs,tf_labels)
            cost_epoch = tf.add(cost,cost_epoch)       
            cost_epoch_pi = tf.add(cost_pi,cost_epoch_pi)
        # Status and outputs
        print('epoch.iter.save: %i.%i.%i, testing cost (Pi): %.3e' %(epoch, iteration, savenum, cost_pi.numpy()), flush=True)

    #cost_epoch /= testing_size
    #cost_epoch /= testing_size 
    testing_costs_pi.append(cost_epoch_pi/testing_size)
    testing_costs.append(cost_epoch/testing_size)

training_costs_pi = np.array(training_costs_pi)
testing_costs_pi = np.array(testing_costs_pi)
training_costs = np.array(training_costs)
testing_costs = np.array(testing_costs)
np.save('training_costs_pi.npy',training_costs_pi)
np.save('testing_costs_pi.npy',testing_costs_pi)
np.save('training_costs.npy',training_costs)
np.save('testing_costs.npy',testing_costs)
