
import tensorflow as tf
import numpy as np
# Domain parameters
L = 1
N = 256
Bx = By = (-np.pi*L, np.pi*L)
Nx = Ny = N

# Net parameters
datatype = tf.float64
stacks = 1
stack_width = 3
filters_base = 6
output_channels = 6
unet_kw = {}
unet_kw['kernel_size'] = 1
unet_kw['activation'] = 'relu'
unet_kw['use_bias'] = False
unet_kw['batch_norm'] = False

# Training parameters
restore_epoch = 0
epochs = 1000
snapshots = 40
testing_size = 15
training_size = snapshots - testing_size
perm_seed = 978
tf_seed = 718
learning_rate = 1e-4
checkpoint_path = "checkpoints/unet"
diss_cost = 0
device = "/device:GPU:0"
device = "/cpu:0"
