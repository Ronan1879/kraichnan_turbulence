import numpy as np
from dedalus import public as de
from parameters import *

def strain_rate(domain,ux,uy):

    dx = domain.bases[0].Differentiate
    dy = domain.bases[1].Differentiate

    # Compute resolved strain components
    Sxx = dx(ux).evaluate()
    Syy = dy(uy).evaluate()
    Sxy = Syx = (0.5*(dx(ux) + dy(uy))).evaluate()

    S = [Sxx,Syy,Sxy]
    for comp in S:
        comp.set_scales(1)
    return S


def vorticity(domain,ux,uy):

    dx = domain.bases[0].Differentiate
    dy = domain.bases[1].Differentiate

    # Compute vorticity
    wz = (dx(uy) - dy(ux)).evaluate()
    wz.set_scales(1)
    return wz

def uxuy_derivatives(domain,ux,uy):

    dx = domain.bases[0].Differentiate
    dy = domain.bases[1].Differentiate

    # Compute strain components
    dx_ux = dx(ux).evaluate()
    dy_uy = dy(uy).evaluate()
    dx_uy = dx(uy).evaluate()
    dy_ux = dy(ux).evaluate()

    uxuy_deriv = [dx_ux,dx_uy,dy_ux,dy_uy]

    for comp in uxuy_deriv:
        comp.set_scales(1)

    return uxuy_deriv

def magn_strain_rate(domain,ux,uy):
    S = strain_rate(domain,ux,uy)
    Sxx = S[0]
    Syy = S[0]
    Sxy = S[0]
    Syx = Sxy
    S_magn =  np.sqrt(Sxx*Sxx + Sxy*Sxy + Syx*Syx + Syy*Syy).evaluate()

    S_magn.set_scales(1)

    return S_magn

def magn_vorticity_grad(domain,ux,uy):
    dx = domain.bases[0].Differentiate
    dy = domain.bases[1].Differentiate

    wz = vorticity(domain,ux,uy)
    wz_grad_magn = np.sqrt(dx(wz)**2 + dy(wz)**2).evaluate()

    wz_grad_magn.set_scales(1)

    return wz_grad_magn

def implicit_subgrid_stress(domain,filter,ux,uy):
    
    filt_ux = filter(ux).evaluate()
    filt_uy = filter(uy).evaluate()

    txx = (filt_ux*filt_ux - filter(ux*ux)).evaluate()
    tyy = (filt_uy*filt_uy - filter(uy*uy)).evaluate()
    txy = (filt_ux*filt_uy - filter(ux*uy)).evaluate()

    tau = [txx,tyy,txy]

    # Deviatoric subgrid stress
    tr_tau = (tau[0] + tau[1]).evaluate()
    tau[0] = (tau[0] - tr_tau/2).evaluate()
    tau[1] = (tau[1] - tr_tau/2).evaluate()

    for comp in tau:
        comp.set_scales(1)
        
    return tau

def explicit_subgrid_stress(domain,filter,ux,uy):

    filt_ux = filter(ux).evaluate()
    filt_uy = filter(uy).evaluate()

    txx = filter(filt_ux*filt_ux - ux*ux).evaluate()
    tyy = filter(filt_uy*filt_uy - uy*uy).evaluate()
    txy = tyx = filter(filt_ux*filt_uy - ux*uy).evaluate()

    tau = [txx,tyy,txy]
    
    # Deviatoric subgrid stress
    tr_tau = (tau[0] + tau[1]).evaluate()
    tau[0] = (tau[0] - tr_tau/2).evaluate()
    tau[1] = (tau[1] - tr_tau/2).evaluate()
    
    for comp in tau:
        comp.set_scales(1)

    return tau