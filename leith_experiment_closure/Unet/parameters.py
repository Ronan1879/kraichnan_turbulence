"""Parameters file."""

import numpy as np
import tensorflow as tf

def lrelu(x, alpha=0.1):
    return tf.maximum(x, tf.multiply(x, alpha))

# Domain
L = 1
Npix = 256
N_dns = 2048
N_filter = 256
Bx = By = (-np.pi*L, np.pi*L)
Nx = Ny = Npix
mesh = None

grid_size = (Bx[1]-Bx[0])/N_filter

# Smagorinsky parameters
smag_coef = 0.3
# Leith parameters
leith_coef = 0.3

# Physical parameters
Re = 32000  # V * L / ν
V = 1  # tc = L / V = 1
k = 1 / L
ν = V * L / Re

# Initial field proprety
k_scaling = -3

# closure correction model (None, 'leith', 'smagorinsky', 'ml')
closure_type = None

# Temporal parameters
dt = 0.0001
stop_sim_time = np.inf
stop_wall_time = np.inf
stop_iteration = int(4 // dt) + 1
snapshots_iter = int(0.05 // dt)
slices_iter = int(0.1 // dt)
scalars_iter = int(0.01 // dt)

# Net parameters
datatype = tf.float64
stacks = 1
stack_width = 2
filters_base = 40
input_channels = 1
output_channels = 1
unet_kw = {}
unet_kw['kernel_size'] = 3
unet_kw['activation'] = lrelu
unet_kw['use_bias'] = False
unet_kw['batch_norm'] = False

# Training parameters
restore_epoch = 0
epochs = 5000
snapshots = 5
testing_size = 1
training_size = snapshots - testing_size
batch_size = 2
perm_seed = 978
tf_seed = 718
learning_rate = 6e-4
checkpoint_path = "checkpoints/unet"
diss_cost = 0
device = "/device:GPU:0"
device = "/cpu:0"
