"""
Compute subgrid stress tensor from the Leith model
"""

import numpy as np
from parameters import *
    
def update_forcing(ux,uy,domain):

    txx = domain.new_field(name='txx')
    tyy = domain.new_field(name='tyy')
    txy = tyx =  domain.new_field(name='txy')
    dx = domain.bases[0].Differentiate
    dy = domain.bases[1].Differentiate

    Sxx = dx(ux)
    Syy = dy(uy)
    Sxy = Syx = (0.5*(dx(uy) + dy(ux)))

    w = (dx(uy) - dy(ux))
    norm_grad_w = np.sqrt(dx(w)**2 + dy(w)**2)

    eddy_visc = (leith_coef * grid_size)**3 * norm_grad_w

    txx = 2 * eddy_visc * Sxx
    tyy = 2 * eddy_visc * Syy
    tyx = 2 * eddy_visc * Syx
    txy = tyx

    Fx_temp = (dx(txx) + dy(tyx)).evaluate()
    Fy_temp = (dx(txy) + dy(tyy)).evaluate()

    return Fx_temp['g'], Fy_temp['g']
