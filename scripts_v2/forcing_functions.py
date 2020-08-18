import numpy as np
import parameters as param
import tensorflow as tf
import fluid_functions as ff
from misc_functions import array_of_tf_components

def smago_forcing(domain,ux,uy):
    S = ff.strain_rate(domain,ux,uy)
    magn_S = ff.magn_strain_rate(domain,ux,uy)

    S = [S_comp['g'] for S_comp in S]

    magn_S = magn_S[0]['g']

    eddy_visc = (param.Cs * param.grid_size)**2 * magn_S

    outputs = np.moveaxis(np.expand_dims(eddy_visc,axis=0)*np.array(S),0,-1)[None]

    tf_outputs = tf.cast(outputs,param.datatype)
    tau_pred = array_of_tf_components(tf_outputs)

    return tau_pred

def leith_forcing(domain,ux,uy):
    S = ff.strain_rate(domain,ux,uy)
    magn_wz_grad = ff.magn_vorticity_grad(domain,ux,uy)

    S = [S_comp['g'] for S_comp in S]

    magn_wz_grad = magn_wz_grad[0]['g']

    eddy_visc = (param.Cl * param.grid_size)**3 * magn_wz_grad

    outputs = np.moveaxis(np.expand_dims(eddy_visc,axis=0)*np.array(S),0,-1)[None]

    tf_outputs = tf.cast(outputs,param.datatype)
    tau_pred = array_of_tf_components(tf_outputs)

    return tau_pred
