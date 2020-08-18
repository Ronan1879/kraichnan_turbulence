import numpy as np
from dedalus import public as de
from parameters import *

def create_random_uxuy(domain):

    x = domain.grid(0)
    y = domain.grid(1)
    kx = domain.elements(0)
    ky = domain.elements(1)
    k = np.sqrt(kx**2 + ky**2)

    wz = domain.new_field(name='wz')
    psi = domain.new_field(name='psi')

    # Generate random normal field with fourier scaling that dampens higher k modes

    phase = 2*np.pi*np.random.rand(*wz['c'].shape)
    wz['c'] = np.exp(1j*phase)*(k + (k==0))**(k_scaling)
    psi['c'] = wz['c'] / (k**2 + (k**2==0))

    # Create fields for the velocities

    ux = domain.new_field(name='ux')
    uy = domain.new_field(name='uy')
    ux.require_grid_space()
    uy.require_grid_space()

    # Differentiate the streamfunction to get ux and uy velocity fields

    psi.differentiate('x',out=uy)
    psi.differentiate('y',out=ux)

    uy['g'] = -uy['g']

    # Normalize velocity fields
    
    uy['g'] = (uy['g'] - np.mean(uy['g']))/np.std(uy['g'])
    ux['g'] = (ux['g'] - np.mean(ux['g']))/np.std(ux['g'])
    
    ux.set_scales(1)
    uy.set_scales(1)
    
    return ux,uy