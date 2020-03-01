"""
Generate initial field from random vorticity field.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools
import parameters as param

# Create bases and domain
x_basis = de.Fourier('x', param.N, interval=(param.Bx[0], param.Bx[1]), dealias=1)
y_basis = de.Fourier('y', param.N, interval=(param.Bx[0], param.By[1]), dealias=1)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Generate random vorticity field using red noise
x = domain.grid(0)
y = domain.grid(1)
kx = domain.elements(0)
ky = domain.elements(1)
k = np.array(np.meshgrid(ky,kx))
k = np.sqrt(k[0]**2 + k[1]**2)
w = domain.new_field(name='w')
w['g'] = np.random.normal(0,1,(x.shape[0],y.shape[1]))
# Small term in division by k is included to avoid blow up at k = 0
w['c'] = w['c']/(k+0.0000001)**(3/5)
w['g'] = 500*(w['g']-np.mean(w['g']))

problem = de.LBVP(domain, variables=['psi'])

problem.parameters['w'] = w

# Setup PDE for Poisson equation with periodic boundaries
problem.substitutions['L(a)'] = "dx(dx(a)) + dy(dy(a))"
problem.add_equation("L(psi)= -w",condition="(nx != 0) or (ny != 0)")
problem.add_equation("psi = 0", condition="(nx == 0) and (ny == 0)")

solver = problem.build_solver()
solver.solve()
psi = solver.state['psi']
psi.require_grid_space()

# Create fields for the velocities
ux = domain.new_field(name='ux')
uy = domain.new_field(name='uy')
ux.require_grid_space()
uy.require_grid_space()

# Differentiate the streamfunction to get ux and uy velocity fields
psi.differentiate('x',out=uy)
psi.differentiate('y',out=ux)

uy['g'] = -uy['g']
ux_init = ux['g']
uy_init = uy['g']




