"""Train u-net."""

import numpy as np
import tensorflow as tf
#from unet import *
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

def axslice(axis, start, stop, step=None):
    """Slice array along a specified axis."""
    if axis < 0:
        raise ValueError("`axis` must be positive")
    slicelist = [slice(None)] * axis
    slicelist.append(slice(start, stop, step))
    return tuple(slicelist)
def pad_axis_periodic(tensor, axis, pad_left, pad_right):
    """Periodically pad tensor along a single axis."""
    N = tensor.shape[axis]
    left = tensor[axslice(axis, N-pad_left, N)]
    right = tensor[axslice(axis, 0, pad_right)]
    return tf.concat([left, tensor, right], axis)


class PeriodicConv2D(tf.keras.layers.Layer):
    """2D convolution layer with periodic padding."""

    def __init__(self, filters, kernel_size, kernel_center=None, strides=(1,1), batch_norm=False,**kwargs):
        super(PeriodicConv2D,self).__init__()
        

        # Handle integer kernel specifications
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(kernel_center, int):
            kernel_center = (kernel_center,) * 2
        if kernel_center is None:
            kernel_center = tuple(ks//2 for ks in kernel_size)

        # Store inputs
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_center = kernel_center
        self.strides = strides
        self.batch_norm = batch_norm
        # Calculate pads
        self.pad_left = kernel_center
        self.pad_right = [ks - kc - 1 for ks, kc in zip(kernel_size, kernel_center)]
        # Build layers
        self.conv_valid = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='valid',**kwargs)
        if batch_norm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()

    def check_input_shape(self, input_shape):
        # Check strides evenly divide data shape
        batch, *data_shape, channels = input_shape
        for n, s in zip(data_shape, self.strides):
            if n%s != 0:
                raise ValueError("Strides must evenly divide data shape in periodic convolution.")
    
    def __call__(self, x):
        # Check shape
        self.check_input_shape(x.shape)

        # Iteratively apply periodic padding, skipping first axis (batch)
        for axis in range(2):
            x = pad_axis_periodic(x, axis+1, self.pad_left[axis], self.pad_right[axis])

        # Apply valid convolution
        x = self.conv_valid(x)
        # Batch normalization
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        return x
    def get_config(self):
        config = super(PeriodicConv2D, self).get_config()
        config.update({'filters': self.filters,'kernel_size': self.kernel_size,'kernel_center': self.kernel_center,
            'strides': self.strides,'batch_norm': self.batch_norm})
        return config


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


train_inputs = np.load("inputs.npy")[:training_size]
train_labels = np.load("labels.npy")[:training_size]
test_inputs = np.load("inputs.npy")[training_size:training_size+testing_size]
test_labels = np.load("labels.npy")[training_size:training_size+testing_size]

# buffer_size must be atleast equal or bigger than training_size
buffer_size = training_size

train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs,train_labels)).shuffle(buffer_size).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs,test_labels)).batch(batch_size)

if resume_training == True:
    model = tf.keras.models.load_model(checkpoint_path + '_best.h5', custom_objects={'PeriodicConv2D': PeriodicConv2D,'cost_function':cost_function}, compile=True)
    initial_epoch = int(np.loadtxt('data.csv',delimiter=',',skiprows=1)[-1,0]) + 1
    print(initial_epoch)
    print("Succesfully loaded model from {}".format(checkpoint_path + '_latest.h5'))
else:    
    model = tf.keras.Sequential()
    # Input layer
    model.add(PeriodicConv2D(filters, kernel_size,activation=activation_func, input_shape=(N,N,input_channels)))
    # Hidden layers
    for i in range(hidden_layers):
        model.add(PeriodicConv2D(filters, kernel_size,activation=activation_func))
    # Output layer
    model.add(PeriodicConv2D(output_channels, (1,1), activation=None))
    model.compile(optimizer = tf.keras.optimizers.Adamax(learning_rate), loss=cost_function)
    # There is an issue when resuming training for the first time, does not properly load the model,
    # and the cost is of the order of an untrained model. The next three lines prevent this,
    # by loading the model a first time.
    model.evaluate(np.zeros((1,N,N,input_channels)))
    model.save('initializer_file.h5')
    model = tf.keras.models.load_model('initializer_file.h5', custom_objects={'PeriodicConv2D': PeriodicConv2D,'cost_function':cost_function}, compile=True)

    initial_epoch = 1
    print("Initialized Sequential model with {} hidden_layers and {} filters per layers".format(hidden_layers,filters))



# Save model state with lowest validation loss
best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path + '_best.h5',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# Save latest model state which is useful in order to resume training
latest_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path + '_latest.h5',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=False)


filename = 'data.csv'
csv_logger_callback = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

callback_set = [best_checkpoint_callback,latest_checkpoint_callback,csv_logger_callback]
model.fit(train_dataset, epochs = initial_epoch + epochs, validation_data = test_dataset, callbacks=callback_set,initial_epoch=initial_epoch)