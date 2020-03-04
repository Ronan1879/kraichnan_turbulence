"""Parameters file."""

import numpy as np

# Domain
L = 1
N = 256
Bx = By = (-np.pi*L, np.pi*L)
Nx = Ny = N
mesh = None


# Physical parameters
Re = 32000  # V * L / ν
V = L  # tc = L / V = 1
k = 1 / L
ν = V * L / Re

# Temporal parameters
dt = 0.0001
stop_sim_time = np.inf
stop_wall_time = np.inf
stop_iteration = int(8 // dt) + 1
snapshots_iter = int(0.05 // dt)
slices_iter = int(0.1 // dt)
scalars_iter = int(0.01 // dt)
