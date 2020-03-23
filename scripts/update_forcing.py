"""Load weights and model of neural net"""

import unet

# Trying to import only two functions from train_unet.py but somehow runs the whole script instead.
#from train_unet import array_of_tf_components,deviatoric_part
import numpy as np
from parameters import *
import post as post_tools
import tensorflow as tf
tf.enable_eager_execution()
import xarray

model = unet.Unet(stacks,stack_width,filters_base,output_channels,strides=(2,2), **unet_kw)

# Loads the weights
model.load_weights("checkpoints/unet-" + str(epochs-1) + ".index")

def array_of_tf_components(tf_tens):
    """Create object array of tensorflow packed tensor components."""
    # Collect components
    # Tensorflow shaped as (batch, *shape, channels)
    comps = ['xx', 'yy', 'xy']
    c = {comp: tf_tens[..., n] for n, comp in enumerate(comps)}
    c['yx'] = c['xy']
    # Build object array
    tens_array = np.array([[None, None],
                           [None, None]], dtype=object)
    for i, si in enumerate(['x', 'y']):
        for j, sj in enumerate(['x', 'y']):
            tens_array[i, j] = c[si+sj]
    return tens_array

def deviatoric_part(tens):
    """Compute deviatoric part of tensor."""
    tr_tens = np.trace(tens)
    tens_d = tens.copy()
    N = tens.shape[0]
    for i in range(N):
        tens_d[i, i] = tens[i, i] - tr_tens / N
    return tens_d
    
def update_forcing(ux,uy,domain):

    txx = domain.new_field(name='txx')
    tyy = domain.new_field(name='tyy')
    txy = tyx =  domain.new_field(name='txy')
    dx = domain.bases[0].Differentiate
    dy = domain.bases[1].Differentiate

    Sxx = dx(ux).evaluate()
    Syy = dy(uy).evaluate()
    Sxy = Syx = (0.5*(dx(uy) + dy(ux))).evaluate()
    S = [Sxx['g'],Syy['g'],Syx['g']]

    inputs = np.moveaxis(np.array(S), 0, -1)[None]

    tf_inputs = [tf.cast(inputs,datatype)]
    tf_outputs = model.call(tf_inputs)

    tau_pred = deviatoric_part(array_of_tf_components(tf_outputs))

    txx['g'] = tau_pred[0,0]
    txy['g'] = tau_pred[0,1]
    tyx['g'] = tau_pred[1,0]
    tyy['g'] = tau_pred[1,1]

    Fx = (dx(txx) + dy(tyx)).evaluate()
    Fy = (dx(txy) + dy(tyy)).evaluate()

    diss_ux_grid = (dx(dx(ux)) + dy(dy(ux))).evaluate()['g']
    diss_uy_grid = (dx(dx(uy)) + dy(dy(uy))).evaluate()['g']

    correct_Fx = Fx['g']*diss_ux_grid > 0 
    correct_Fy = Fy['g']*diss_uy_grid > 0

    Fx['g'] *= correct_Fx 
    Fy['g'] *= correct_Fy 

    return Fx, Fy


