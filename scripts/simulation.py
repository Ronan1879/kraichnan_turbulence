
import os
import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools
import parameters as param
import initial_field as init_f

import logging
logger = logging.getLogger(__name__)


# Bases and domain
x_basis = de.Fourier('x', param.Nx, interval=param.Bx, dealias=1)
y_basis = de.Fourier('y', param.Ny, interval=param.By, dealias=1)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64, mesh=param.mesh)

# Problem
problem = de.IVP(domain, variables=['p','ux','uy'])
problem.parameters['ν'] = param.ν
problem.parameters['Fx'] = 0
problem.parameters['Fy'] = 0
problem.substitutions['ωz'] = "dx(uy) - dy(ux)"
problem.substitutions['ke'] = "(ux*ux + uy*uy) / 2"
problem.substitutions['en'] = "(ωz*ωz) / 2"
problem.substitutions['L(a)'] = "dx(dx(a)) + dy(dy(a))"
problem.substitutions['A(a)'] = "ux*dx(a) + uy*dy(a)"
problem.add_equation("dt(ux) - ν*L(ux) + dx(p) = -A(ux) + Fx")
problem.add_equation("dt(uy) - ν*L(uy) + dy(p) = -A(uy) + Fy")
problem.add_equation("dx(ux) + dy(uy) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_equation("p = 0", condition="(nx == 0) and (ny == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions
ux = solver.state['ux']
uy = solver.state['uy']

if pathlib.Path('restart.h5').exists():
    solver.load_state('restart.h5', -1)
else:
    ux['g'] = init_f.ux_init
    uy['g'] = init_f.uy_init

# Integration parameters
solver.stop_sim_time = param.stop_sim_time
solver.stop_wall_time = param.stop_wall_time
solver.stop_iteration = param.stop_iteration

folder_name = 'dns_simulation'
if os.path.exists(folder_name) == False:
    os.mkdir(folder_name)

# Analysist
snapshots = solver.evaluator.add_file_handler(folder_name + '/snapshots', iter=param.snapshots_iter, max_writes=1, mode='overwrite')
snapshots.add_system(solver.state)
snapshots.add_task("dx(uy) - dy(ux)",name='wz')

scalars = solver.evaluator.add_file_handler(folder_name + '/scalars', iter=param.scalars_iter, max_writes=100, mode='overwrite')
scalars.add_task("integ(ke)", name='KE')
scalars.add_task("integ(en)", name='EN')

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("integ(ke)", name='KE')

# Main loop
dt = param.dt
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        
        solver.step(dt)
        
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Total KE = %f' %flow.max('KE'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
