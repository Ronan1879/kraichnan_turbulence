"""Train u-net."""

import numpy as np
#import xarray
import tensorflow as tf
import unet
from parameters import *
import parameters as params
from misc_functions import deviatoric_part
from misc_functions import array_of_tf_components
#from post import *
#from create_uxuy import create_random_uxuy
import time

"""
def create_inputs_labels(Nsnapshots):
    domain = build_domain(params)
    filter = build_filter_func(domain,N)

    inputs = np.zeros((Nsnapshots,Nx,Ny,input_channels))
    labels = np.zeros((Nsnapshots,Nx,Ny,output_channels))

    for i in range(Nsnapshots):
        ux,uy = create_random_uxuy(domain)

        filt_ux = filter(ux).evaluate()
        filt_uy = filter(uy).evaluate()
        input_fields = input_func(domain,filt_ux,filt_uy)
        label_fields = label_func(domain,filter,ux,uy)

        for input_field in input_fields:
            input_field.set_scales(1)
        for label_field in label_fields:
            label_field.set_scales(1)

        input = [input_field['g'] for input_field in input_fields]
        label = [label_field['g'] for label_field in label_fields]

        # Reshape as (batch, *shape, channels)
        inputs[i,:,:,:] = np.moveaxis(np.array(input), 0, -1)[None]
        labels[i,:,:,:] = np.moveaxis(np.array(label), 0, -1)[None]

    return inputs, labels
"""

def cost_function(outputs,labels):
    # Load components into object arrays, take deviatoric part of stresses
    tau_pred = deviatoric_part(array_of_tf_components(outputs))
    tau_true = deviatoric_part(array_of_tf_components(labels))

    # Compute cost of predicted subgrid stress tensor
    # Pointwise deviatoric stress error
    tau_d_diff = tau_true - tau_pred
    f2_tau_d_diff = np.trace(np.dot(tau_d_diff,tau_d_diff.T))
    L2_tau_d_error = tf.reduce_sum(f2_tau_d_diff)

    return L2_tau_d_error

def train_step(data_set):

    inputs, labels = data_set

    tf_inputs = [tf.cast(inputs,datatype)]
    tf_labels = tf.cast(labels,datatype)

    with tf.GradientTape() as tape:
        tape.watch(model.variables)
        tf_outputs = model.call(tf_inputs)
        cost = cost_function(tf_outputs,tf_labels)

    weight_grads = tape.gradient(cost,model.variables)
    optimizer.apply_gradients(zip(weight_grads,model.variables))

    return cost

def distributed_train_step(data_set):
    per_replica_losses = strategy.run(train_step, args=(data_set,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = unet.Unet(stacks,stack_width,filters_base,output_channels, **unet_kw)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=model)

if restore_epoch:
    restore_path = f"{checkpoint_path}-{restore_epoch}"
    checkpoint.restore(restore_path)#.assert_consumed()
    print('Restored from {}'.format(restore_path))
else:
    print('Initializing from scratch.')
initial_epoch = checkpoint.save_counter.numpy() + 1

training_costs = []

BUFFER_SIZE = 4 
#BATCH_SIZE_PER_REPLICA = 4
GLOBAL_BATCH_SIZE = 4#BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

# Only if one batch size is to be used during training
training_size = GLOBAL_BATCH_SIZE

#train_inputs, train_labels = create_inputs_labels(training_size)
train_inputs = np.load("../../../validation_inputs.npy")[:training_size]
train_labels = np.load("../../../validation_labels.npy")[:training_size]

train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
#test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels)).batch(GLOBAL_BATCH_SIZE)

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
#test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

start_time = time.time()
# Train loop
for epoch in range(initial_epoch,initial_epoch+epochs):
    print(f"Beginning epoch {epoch}", flush=True)

    total_loss = 0.0
    num_batches = 0
    for x in train_dist_dataset:
        total_loss += distributed_train_step(x)
        num_batches += 1
    train_loss = total_loss / num_batches
    training_costs.append(train_loss)

    print('epoch.batch_num: %i.%i, training cost : %.3e' %(epoch, num_batches, train_loss.numpy()), flush=True)

training_costs = np.array(training_costs)

end_time = time.time()

print("Training runtime : ",end_time-start_time)

np.save("training_costs.npy",training_costs)

