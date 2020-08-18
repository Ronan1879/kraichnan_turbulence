import numpy as np
import unet
import parameters as param
import fluid_functions as ff
import tensorflow as tf
from misc_functions import deviatoric_part
from misc_functions import array_of_tf_components

model = unet.Unet(stacks,stack_width,filters_base,output_channels, **unet_kw)
checkpoint = tf.train.Checkpoint(net=model)

training_costs = np.load("training_costs.npy")

best_epoch = np.where(training_costs == np.min(training_costs))[0][0]

best_epoch -= best_epoch % 200

restore_path = f"{checkpoint_path}-{best_epoch//200}"
checkpoint.restore(restore_path)#.assert_consumed()
print('Restored from {}'.format(restore_path))

model.call(np.zeros((1,1,Nx,Ny,input_channels)))


def ml_forcing(domain,ux,uy):

    input_fields = ff.uxuy_derivatives(domain,ux,uy)

    inputs = [input_field['g'] for input_field in input_fields]

    # Reshape as (batch, *shape, channels)
    inputs = np.moveaxis(np.array(inputs), 0, -1)[None]

    tf_inputs = [tf.cast(inputs,param.datatype)]
    tf_outputs = model.call(tf_inputs)

    tau_pred = deviatoric_part(array_of_tf_components(tf_outputs))

    return tau_pred