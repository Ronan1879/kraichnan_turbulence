"""Parameters file."""

import numpy as np
import tensorflow as tf
import parameters as params 


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
