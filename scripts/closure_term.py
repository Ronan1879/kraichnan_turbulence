"""
Get closure term represented as the curl of the divergence of the stress tensor
"""

from dedalus import public as de
import parameters as param
import numpy as np

x_basis = de.Fourier('x', param.N, interval=(param.Bx[0], param.Bx[1]), dealias=1)
y_basis = de.Fourier('y', param.N, interval=(param.Bx[0], param.By[1]), dealias=1)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)
txx = domain.new_field(name='txx')
tyy = domain.new_field(name='tyy')
txy = domain.new_field(name='txy')
tyx = domain.new_field(name='tyx')
dx = domain.bases[0].Differentiate
dy = domain.bases[1].Differentiate

def get_pi(tf_array):

    # Collect stress tensor components
    txx['g'] = tf_array[0,0]
    txy['g'] = tf_array[0,1]
    tyx['g'] = tf_array[1,0]
    tyy['g'] = tf_array[1,1]

    # Evaluate divergence of stress tensor
    fx = (dx(txx) + dy(tyx)).evaluate()
    fy = (dx(txy) + dy(tyy)).evaluate()
    # Compute curl of subgrid force
    pi = (dx(fy) - dy(fx)).evaluate()
    return pi['g']
    