
import numpy as np
import tensorflow as tf
from misc_functions import deviatoric_part
from misc_functions import array_of_tf_components
from parameters import *
import fluid_functions as ff
from filter_functions import *
from post import *
from create_uxuy import *

def create_inputs_labels(Nsnapshots):
    domain = build_domain(params)
    filter = build_gaussian_filter(domain,N,epsilon)

    inputs = np.zeros((Nsnapshots,N,N,input_channels))
    labels = np.zeros((Nsnapshots,N,N,output_channels))

    for i in range(Nsnapshots):
        ux,uy = create_random_uxuy(domain)

        filt_ux = filter(ux).evaluate()
        filt_uy = filter(uy).evaluate()
        input_fields = ff.uxuy_derivatives(domain,filt_ux,filt_uy)
        label_fields = ff.implicit_subgrid_stress(domain,filter,ux,uy)

        for input_field in input_fields:
            input_field.set_scales(N/Nx)
        for label_field in label_fields:
            label_field.set_scales(N/Nx)

        input_list = [input_field['g'] for input_field in input_fields]
        label_list = [label_field['g'] for label_field in label_fields]

        # Reshape as (batch, *shape, channels)
        inputs[i,:,:,:] = np.moveaxis(np.array(input_list), 0, -1)[None]
        labels[i,:,:,:] = np.moveaxis(np.array(label_list), 0, -1)[None]

    return inputs, labels

inputs, labels = create_inputs_labels(5)

np.save("inputs.npy",inputs)
np.save("labels.npy",labels)