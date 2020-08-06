"""Train u-net."""

import numpy as np
import xarray
import tensorflow as tf
import unet as unet
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dedalus import public as de
from parameters import *
from filter import *
import time
import os
from mpi4py import MPI
import itertools
from math import log10, floor

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

def snapshot_generator(Nsnapshots):

    inputs = np.zeros((Nsnapshots,Nx,Ny,input_channels))
    labels = np.zeros((Nsnapshots,Nx,Ny,output_channels))

    # Create bases and domain
    x_basis = de.Fourier('x', Nx, interval = Bx, dealias=3/2)
    y_basis = de.Fourier('y', Ny, interval = By, dealias=3/2)
    domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64,comm=MPI.COMM_SELF)

    gauss_filt = build_gauss_filter(domain,N_filter,epsilon)

    x = domain.grid(0)
    y = domain.grid(1)
    kx = domain.elements(0)
    ky = domain.elements(1)
    k = np.sqrt(kx**2 + ky**2)

    wz = domain.new_field(name='wz')
    psi = domain.new_field(name='psi')

    for s in np.arange(Nsnapshots):
        
        # Generate random normal field with fourier scaling that dampens higher k modes
        wz['g'] = np.random.normal(0,1,(x.shape[0],y.shape[1]))
        # The + 1 term is to avoid division by zero
        wz['c'] = wz['c']*(k + (k == 0))**k_scaling
        wz['g'] = (wz['g'] - np.mean(wz['g']))/np.std(wz['g'])

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

        dx = domain.bases[0].Differentiate
        dy = domain.bases[1].Differentiate

        txx = (gauss_filt(ux)*gauss_filt(ux) - gauss_filt(ux*ux)).evaluate()
        tyy = (gauss_filt(uy)*gauss_filt(uy)- gauss_filt(uy*uy)).evaluate()
        txy = (gauss_filt(uy)*gauss_filt(ux) - gauss_filt(ux*uy)).evaluate()
        
        filt_ux = gauss_filt(ux).evaluate()
        filt_uy = gauss_filt(uy).evaluate()

        input_0 = dx(filt_ux).evaluate()
        input_1 = dy(filt_ux).evaluate()
        input_2 = dx(filt_uy).evaluate()
        input_3 = dy(filt_uy).evaluate()
        
        input_0.set_scales(1)
        input_1.set_scales(1)
        input_2.set_scales(1)
        input_3.set_scales(1)

        txx.set_scales(1)
        tyy.set_scales(1)
        txy.set_scales(1)

        inputs[s,:,:,0] = input_0['g']
        inputs[s,:,:,1] = input_1['g']
        inputs[s,:,:,2] = input_2['g']
        inputs[s,:,:,3] = input_3['g']
        labels[s,:,:,0] = txx['g']
        labels[s,:,:,1] = tyy['g']
        labels[s,:,:,2] = txy['g']        

    return inputs, labels

def Linf_cost_function(outputs,labels):
    tau_pred = deviatoric_part(array_of_tf_components(outputs))
    tau_true = deviatoric_part(array_of_tf_components(labels))
    
    tau_d_diff = tau_true - tau_pred
    f1_tau_d_diff = np.sqrt(np.trace(np.dot(tau_d_diff,tau_d_diff.T)))
    Linf_costs = np.max(f1_tau_d_diff,axis=(1,2))
    Linf_error = np.std(Linf_costs)
    Linf_mean_cost = tf.reduce_mean(Linf_costs)
    Linf_tot_cost = tf.reduce_sum(Linf_costs)

    return Linf_tot_cost,Linf_mean_cost,Linf_error

def L2_cost_function(outputs,labels):
    tau_pred = deviatoric_part(array_of_tf_components(outputs))
    tau_true = deviatoric_part(array_of_tf_components(labels))
    
    tau_d_diff = tau_true - tau_pred
    f2_tau_d_diff = np.trace(np.dot(tau_d_diff,tau_d_diff.T))
    L2_costs = tf.reduce_sum(f2_tau_d_diff,axis=(1,2))
    L2_error = np.std(L2_costs)
    L2_mean_cost = tf.reduce_mean(L2_costs)
    L2_tot_cost = tf.reduce_sum(f2_tau_d_diff)

    return L2_tot_cost,L2_mean_cost,L2_error

def L1_cost_function(outputs,labels):
    tau_pred = deviatoric_part(array_of_tf_components(outputs))
    tau_true = deviatoric_part(array_of_tf_components(labels))
    
    tau_d_diff = tau_true - tau_pred
    f1_tau_d_diff = np.sqrt(np.trace(np.dot(tau_d_diff,tau_d_diff.T)))
    L1_costs = tf.reduce_sum(f1_tau_d_diff,axis=(1,2))
    L1_error = np.std(L1_costs)
    L1_mean_cost = tf.reduce_mean(L1_costs)
    L1_tot_cost = tf.reduce_sum(f1_tau_d_diff)

    return L1_tot_cost,L1_mean_cost,L1_error

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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

training_costs = []
output_perrank = np.zeros((1,7))

if rank == 0:
    # Create folder for costs
    if not os.path.exists('costs'):
        os.makedirs('costs')
    # Generate training and testing inputs and labels
    train_inputs , train_labels = snapshot_generator(Nsnapshots=training_size)
else:
    train_inputs = None
    train_labels = None

train_inputs = comm.bcast(train_inputs, root=0)
train_labels = comm.bcast(train_labels, root=0)
    
train_idx = np.arange(train_inputs.shape[0])

rand = np.random.RandomState(seed=perm_seed)

stacks_opt = [1,2]#[1,2]
stack_width_opt = [1,2]#[1,2,3]
filter_base_opt = [12,24]#[12,24,48,72,96,120,168]
kernel_size_opt = [3,5]#[3,5,7]

opts = [stacks_opt,stack_width_opt,filter_base_opt,kernel_size_opt]
combinations = list(itertools.product(*opts))

perrank = len(combinations)//size

comm.Barrier()

start_time = time.time()

for combination in combinations[rank*perrank:(rank+1)*perrank]:

    stacks = combination[0]
    stack_width = combination[1]
    filters = combination[2]
    kernel_size = (combination[3],combination[3])
    checkpoint_path = "checkpoints/{}_{}_{}_{}/".format(stacks,stack_width,filters,combination[3])
    costs_path = "costs/{}_{}_{}_{}/".format(stacks,stack_width,filters,combination[3])

    model = unet.Unet(stacks,stack_width,filters,output_channels, kernel_size = kernel_size, **unet_kw)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(epochs):

        batched_train_idx = rand.permutation(train_idx).reshape((-1,batch_size))
        #Train
        for iteration, snapshot_num in enumerate(batched_train_idx):

            # Load adjascent outputs
            inputs_0 = train_inputs[snapshot_num]
            labels_0 = train_labels[snapshot_num]

            with tf.device(device):
                # Combine inputs to predict layer outputs
                tf_inputs = [tf.cast(inputs_0,datatype)]
                tf_labels = tf.cast(labels_0,datatype)

                with tf.GradientTape() as tape:
                    tape.watch(model.variables)
                    tf_outputs = model.call(tf_inputs)
                    cost, mean_cost, error = L2_cost_function(tf_outputs,tf_labels)

                weight_grads = tape.gradient(cost,model.variables)
                optimizer.apply_gradients(zip(weight_grads,model.variables))

                # Status and outputs
                #print('epoch.iter.save: {}.{}.{}, training cost : {}'.format(epoch, iteration, snapshot_num, cost.numpy()), flush=True)
        
                training_costs.append(cost)
        if epoch % 1 == 0:
            model.save_weights(checkpoint_path + "checkpoint_{}".format(epoch))
            #print("weights saved")

    # Create validation data set
    valid_inputs , valid_labels = snapshot_generator(Nsnapshots=validation_size)
    tf_valid_inputs = [tf.cast(valid_inputs,datatype)]
    tf_valid_labels = tf.cast(valid_labels,datatype)

    tf_valid_outputs = model.call(tf_valid_inputs)

    L1_sum,L1_mean,L1_error = L1_cost_function(tf_valid_outputs,tf_valid_labels)
    L2_sum,L2_mean,L2_error = L2_cost_function(tf_valid_outputs,tf_valid_labels)
    Linf_sum,Linf_mean,Linf_error = Linf_cost_function(tf_valid_outputs,tf_valid_labels)

    L1_mean, L1_error, L1_num_deci = convert_notation(L1_mean,L1_error)
    L2_mean, L2_error, L2_num_deci = convert_notation(L2_mean,L2_error)
    Linf_mean, Linf_error, Linf_num_deci = convert_notation(Linf_mean,Linf_error)

    if (output_perrank == 0).all() == True:
        output_perrank = np.array([[stacks,stack_width,filters,combination[3],L1_mean,L1_error,L2_mean,L2_error,Linf_mean,Linf_error]])
    else:
        output_perrank = np.concatenate((output_perrank,np.array([[stacks,stack_width,filters,combination[3],L1_mean,L1_error,L2_mean,L2_error,Linf_mean,Linf_error]])),axis=0)
    
    # Create folder for costs and save data
    if not os.path.exists(costs_path):
        os.makedirs(costs_path)
    np.save(costs_path + "training_costs.npy",np.array(training_costs))

comm.Barrier()

end_time = time.time()

print("Training time :",end_time-start_time)

output_data = comm.gather(output_perrank,root=0)

if rank == 0:
    output_data = np.concatenate(output_data,axis=0)
    np.savetxt("data.csv",output_data, fmt=('%d','%d','%d','%d','%g','%g','%g','%g','%g','%g'))
