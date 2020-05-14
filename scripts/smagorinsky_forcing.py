"""
Compute subgrid stress tensor from the Smagorinsky model
"""

import numpy as np
from parameters import *

    
def update_forcing(ux,uy,domain):

    txx = domain.new_field(name='txx')
    tyy = domain.new_field(name='tyy')
    txy = tyx =  domain.new_field(name='txy')
    dx = domain.bases[0].Differentiate
    dy = domain.bases[1].Differentiate

    Sxx = dx(ux).evaluate()
    Syy = dy(uy).evaluate()
    Sxy = Syx = (0.5*(dx(uy) + dy(ux))).evaluate()

    S = np.sqrt(2 * (Sxx['g']**2 + Syy['g']**2 + 2 * Sxy['g']**2))

    eddy_visc = (smag_coef * grid_size)**2 * S

    txx['g'] = 2 * eddy_visc * Sxx['g']
    tyy['g'] = 2 * eddy_visc * Syy['g']
    tyx['g'] = 2 * eddy_visc * Syx['g']
    txy['g'] = tyx['g']

    Fx_temp = (dx(txx) + dy(tyx)).evaluate()
    Fy_temp = (dx(txy) + dy(tyy)).evaluate()

    return Fx_temp['g'], Fy_temp['g']
