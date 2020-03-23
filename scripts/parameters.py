"""Parameters file."""

import numpy as np
import tensorflow as tf

# Domain
L = 1
Npix = 256
N_filter = 128
Bx = By = (-np.pi*L, np.pi*L)
Nx = Ny = Npix
mesh = None


# Physical parameters
Re = 32000  # V * L / ν
V = 1  # tc = L / V = 1
k = 1 / L
ν = V * L / Re

# Initial field proprety
k_scaling = 3/5

# Perform machine learning correction
correct_simulation = True

# Temporal parameters
dt = 0.0001
stop_sim_time = np.inf
stop_wall_time = np.inf
stop_iteration = int(8 // dt) + 1
snapshots_iter = int(0.05 // dt)
slices_iter = int(0.1 // dt)
scalars_iter = int(0.01 // dt)

# Net parameters
datatype = tf.float64
stacks = 1
stack_width = 3
filters_base = 10
output_channels = 3
unet_kw = {}
unet_kw['kernel_size'] = 1
unet_kw['activation'] = 'relu'
unet_kw['use_bias'] = False
unet_kw['batch_norm'] = False

# Training parameters
restore_epoch = 0
epochs = 10
snapshots = 2
testing_size = 1
training_size = snapshots - testing_size
perm_seed = 978
tf_seed = 718
learning_rate = 1e-4
checkpoint_path = "checkpoints/unet"
diss_cost = 0
device = "/device:GPU:0"
device = "/cpu:0"
