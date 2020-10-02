"""Train u-net."""

import numpy as np

import tensorflow as tf
import unet
from parameters import *
from misc_functions import deviatoric_part
from misc_functions import array_of_tf_components

from filter_functions import *
from post import *
from create_uxuy import *
import xarray
import fluid_functions as ff

# Set random seeds
np.random.seed(rand_seed)
tf.random.set_seed(tf_seed)


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

def cost_function(labels,outputs):
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


train_inputs, train_labels = create_inputs_labels(training_size)
test_inputs, test_labels = create_inputs_labels(testing_size)

# buffer_size must be atleast equal or bigger than training_size
buffer_size = training_size

train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs,train_labels)).shuffle(buffer_size).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs,test_labels)).batch(batch_size)

# Build network and optimizer
model = unet.Unet(stacks,stack_width,filters_base,output_channels, **unet_kw)

model.compile(optimizer = tf.keras.optimizers.Adamax(learning_rate), loss = cost_function)

# " mode = 'min' " saves model with lowest test_dataset loss, also known as validation loss
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
	save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


filename = 'data.csv'
csv_logger_callback = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

model.fit(train_dataset, epochs = epochs, validation_data = test_dataset, callbacks=[model_checkpoint_callback,csv_logger_callback])


