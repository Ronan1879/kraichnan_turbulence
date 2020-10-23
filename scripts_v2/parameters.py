"""Parameters file."""

import numpy as np
import tensorflow as tf

# Domain
L = 1
Bx = By = (-np.pi*L, np.pi*L)
N = 2048
Nx = Ny = N
N_filter = 256
mesh = None

grid_size = (Bx[1]-Bx[0])/N

# Smagorinsky parameters
Cs = 0.3
# Leith parameters
Cl = 0.3

# Physical parameters
Re = 32000  # V * L / ν
V = 1  # tc = L / V = 1
k = 1 / L
ν = V * L / Re

# vorticity scaling
k_scaling = -1/2
# gaussian sharpness parameter
epsilon = 1e-10

# Temporal parameters
dt = 0.0001
stop_sim_time = np.inf
stop_wall_time = np.inf
stop_iteration = int(4 // dt) + 1
snapshots_iter = int(0.05 // dt)
slices_iter = int(0.1 // dt)
scalars_iter = int(0.01 // dt)

# U-Net parameters
datatype = tf.float32
stacks = 1
stack_width = 1
filters_base = 12
unet_kw = {}
unet_kw['kernel_size'] = 3
unet_kw['activation'] = 'relu'
unet_kw['use_bias'] = True
unet_kw['batch_norm'] = False

# Sequential model parameters
hidden_layers = 3
filters = 12
kernel_size = 3
kernel_center = 0
activation_func = 'relu'

input_channels = 4
output_channels = 3


# Training parameters
resume_training = True
epochs = 10
snapshots = 2
testing_size = 1
training_size = snapshots - testing_size
batch_size = 1
perm_seed = 978
tf_seed = 718
learning_rate = 5e-4
checkpoint_path = "checkpoint"
diss_cost = 0
device = "/cpu:0"#"/device:GPU:0"
