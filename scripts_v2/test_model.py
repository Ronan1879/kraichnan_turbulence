
import unet
import numpy as np
from parameters import *
import tensorflow as tf

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.pyplot as plt

from misc_functions import deviatoric_part
from misc_functions import array_of_tf_components

from math import log10, floor
import sys
 
from dedalus import public as de
from filter_functions import *
from create_uxuy import *
from post import *
import fluid_functions as ff


def Linf_cost_function(outputs,labels):
    tau_pred = deviatoric_part(array_of_tf_components(outputs))
    tau_true = deviatoric_part(array_of_tf_components(labels))
    
    tau_d_diff = tau_true - tau_pred
    f1_tau_d_diff = np.sqrt(np.trace(np.dot(tau_d_diff,tau_d_diff.T)))
    Linf_cost = np.max(f1_tau_d_diff)

    return Linf_cost

def L2_cost_function(outputs,labels):
    tau_pred = deviatoric_part(array_of_tf_components(outputs))
    tau_true = deviatoric_part(array_of_tf_components(labels))
    
    tau_d_diff = tau_true - tau_pred
    f2_tau_d_diff = np.trace(np.dot(tau_d_diff,tau_d_diff.T))
    L2_cost = tf.reduce_sum(f2_tau_d_diff)

    return L2_cost

def L1_cost_function(outputs,labels):
    tau_pred = deviatoric_part(array_of_tf_components(outputs))
    tau_true = deviatoric_part(array_of_tf_components(labels))
    
    tau_d_diff = tau_true - tau_pred
    f1_tau_d_diff = np.sqrt(np.trace(np.dot(tau_d_diff,tau_d_diff.T)))
    L1_cost = tf.reduce_sum(f1_tau_d_diff)

    return L1_cost

def round_to_1(x):
    if x < 1:
        x = np.round(x, -int(floor(log10(abs(x)))))
    else:
        x = int(x)
    return x

def convert_notation(mean,error):
    mean = np.array(mean)
    error = np.array(error)
    rounded_error = round_to_1(error)
    num_deci = str(rounded_error)[::-1].find('.')
    return np.round(mean,num_deci),rounded_error,num_deci


model = unet.Unet(stacks,stack_width,filters_base,output_channels,**unet_kw)
checkpoint = tf.train.Checkpoint(net=model)

train_path = 'training_costs.npy'
output_path = 'data.csv'

training_costs = np.load(train_path)

best_epoch = np.where(training_costs == np.min(training_costs))[0][0]

best_epoch -= best_epoch % save_interval

restore_path = f"{checkpoint_path}-{best_epoch//save_interval}"
checkpoint.restore(restore_path)#.assert_consumed()
print('Restored from {}'.format(restore_path))

inputs = np.load("inputs.npy")[:5]
labels = np.load("labels.npy")[:5]

L1_costs = []
L2_costs = []
Linf_costs = []

for input,label in zip(inputs,labels):

    input = input[np.newaxis,...]
    label = label[np.newaxis,...]

    tf_input = [tf.cast(input,datatype)]
    tf_label = tf.cast(label,datatype)

    tf_output = model.call(tf_input)

    L1_costs.append(L1_cost_function(tf_output,tf_label))
    L2_costs.append(L2_cost_function(tf_output,tf_label))
    Linf_costs.append(Linf_cost_function(tf_output,tf_label))

L1_costs = np.array(L1_costs)
L2_costs = np.array(L2_costs)
Linf_costs = np.array(Linf_costs)

L1_mean = np.mean(L1_costs)
L2_mean = np.mean(L2_costs)
Linf_mean = np.mean(Linf_costs)

L1_error = np.std(L1_costs)
L2_error = np.std(L2_costs)
Linf_error = np.std(Linf_costs)

L1_mean, L1_error, L1_num_deci = convert_notation(L1_mean,L1_error)
L2_mean, L2_error, L2_num_deci = convert_notation(L2_mean,L2_error)
Linf_mean, Linf_error, Linf_num_deci = convert_notation(Linf_mean,Linf_error)

data = np.array([[best_epoch,stacks,stack_width,filters_base,unet_kw['kernel_size'],L1_mean,L1_error,L2_mean,L2_error,Linf_mean,Linf_error]])

np.savetxt(output_path,data, fmt=('%d','%d','%d','%d','%d','%g','%g','%g','%g','%g','%g'))
