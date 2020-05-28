"""
Generate initial velocity field from random vorticity field.
"""

import numpy as np
import pathlib
import matplotlib.pyplot as plt

from dedalus import public as de
import parameters as param

# Create bases and domain
x_basis = de.Fourier('x', param.Nx, interval=(param.Bx[0], param.Bx[1]), dealias=1)
y_basis = de.Fourier('y', param.Ny, interval=(param.By[0], param.By[1]), dealias=1)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Generate random vorticity field using red noise
x = domain.grid(0)
y = domain.grid(1)
kx = domain.elements(0)
ky = domain.elements(1)
k = np.array(np.meshgrid(ky,kx))
k = np.sqrt(k[0]**2 + k[1]**2)
ωz = domain.new_field(name='ωz')
ωz['g'] = np.random.normal(0,1,(x.shape[0],y.shape[1]))

# Small addition in division by k is included to avoid blow up at k = 0
ωz['c'] = ωz['c']/(k+0.0000001)**(param.k_scaling)

# Converting vorticity field to a random normal field
volume = np.abs(param.Bx[1]-param.Bx[0])*np.abs(param.By[1]-param.By[0])
std = np.sqrt(np.mean((((ωz-ωz.integrate('x','y')/volume).evaluate())**2).evaluate().integrate('x','y')['g']/volume))
ωz['g'] = (ωz-ωz.integrate('x','y')/volume).evaluate()['g']/std

problem = de.LBVP(domain, variables=['psi'])

problem.parameters['ωz'] = ωz

# Setup PDE for Poisson equation with periodic boundaries
problem.substitutions['L(a)'] = "dx(dx(a)) + dy(dy(a))"
problem.add_equation("L(psi)= -ωz",condition="(nx != 0) or (ny != 0)")
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

# Converting velocity fields to random normal fields
std_ux = np.sqrt(np.mean((((ux-ux.integrate('x','y')/volume).evaluate())**2).evaluate().integrate('x','y')['g']/volume))
std_uy = np.sqrt(np.mean((((uy-uy.integrate('x','y')/volume).evaluate())**2).evaluate().integrate('x','y')['g']/volume))
uy['g'] = (param.V/np.sqrt(2))*(uy-uy.integrate('x','y')/volume).evaluate()['g']/std_uy
ux['g'] = (param.V/np.sqrt(2))*(ux-ux.integrate('x','y')/volume).evaluate()['g']/std_ux

ux_init = ux['g']
uy_init = uy['g']