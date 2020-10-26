import matplotlib.pyplot as plt
import unet
import numpy as np
from parameters import *
import tensorflow as tf
from dedalus import public as de
from misc_functions import deviatoric_part
from misc_functions import array_of_tf_components
import xarray
import fluid_functions as ff

def load_data(domain,savenum):

    ux_field = domain.new_field(name = 'ux')
    uy_field = domain.new_field(name = 'uy')
    
    filename = 'filtered/snapshots_s%i.nc' %savenum
    dataset = xarray.open_dataset(filename)
    comps = ['xx', 'yy','xy']
    ux = dataset['ux'].data
    uy = dataset['uy'].data
    
    ux_field['g'] = ux
    uy_field['g'] = uy

    # Compute inputs from velocity fields
    du_field = ff.uxuy_derivatives(domain,ux_field,uy_field)

    # Velocity field derivatives
    du = [du['g'] for du in du_field]
    # Implicit subgrid stress
    tau = [dataset['im_t'+c].data for c in comps]
    # Deviatoric subgrid stress
    tr_tau = tau[0] + tau[1]
    tau[0] = tau[0] - tr_tau/2
    tau[1] = tau[1] - tr_tau/2
    # Reshape as (batch, *shape, channels)
    input_set = np.moveaxis(np.array(du), 0, -1)[None]
    label_set = np.moveaxis(np.array(tau), 0, -1)[None]

    return input_set, label_set 

def pspec(k,field):
    pspec = (np.abs(field['c'])**2 + np.abs(field['c'])**2)/2

    n_count,bins = np.histogram(k.flatten(),bins=2*N_filter)

    n, bins = np.histogram(k.flatten(),bins=2*N_filter,weights=pspec.flatten())

    return bins, n/n_count


model = unet.Unet(stacks,stack_width,filters_base,output_channels, **unet_kw)
checkpoint = tf.train.Checkpoint(net=model)

checkpoint.restore(checkpoint_path + '/best_epoch')#.assert_consumed()

x_basis = de.Fourier('x', N_filter, interval=Bx, dealias=3/2)
y_basis = de.Fourier('y', N_filter, interval=By, dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64, comm=None)

dx = domain.bases[0].Differentiate
dy = domain.bases[1].Differentiate

kx = domain.elements(0)
ky = domain.elements(1)
k = np.sqrt(kx**2 + ky**2)

input_set, label_set = load_data(domain,66)

tf_input = [tf.cast(input_set,datatype)]
tf_label = tf.cast(label_set,datatype)
tf_output = model.call(tf_input)

tau_pred = deviatoric_part(array_of_tf_components(tf_output))
tau_true = deviatoric_part(array_of_tf_components(tf_label))

fig, ax = plt.subplots(1,3,figsize=(10*3,8))

titles = [r'$\tau_{xy}$',r'$\tau_{xx}$',r'$\tau_{yy}$',]

l = 0
for i in [0,1]:
    for j in [1,0]:
        if l == 3:
            break
        for axis in ['top','bottom','left','right']:
            ax[l].spines[axis].set_linewidth(3)
        ax[l].tick_params(direction='out',width=2,length=5,labelsize = 20)

        pred = domain.new_field(name='pred')
        true = domain.new_field(name='true')

        pred['g'] = tau_pred[i,j]
        true['g'] = tau_true[i,j]

        pred_bins, pred_ps = pspec(k,pred)
        true_bins, true_ps = pspec(k,true)

        ax[l].plot(pred_bins[:-1],pred_ps,label=pred.name)
        ax[l].plot(true_bins[:-1],true_ps,label=true.name)
        ax[l].set_title(titles[l],fontsize=26)

        ax[l].set_ylabel("$P(k)$",fontsize=20)
        ax[l].set_xlabel("$k$",fontsize=20)
        ax[l].set_yscale("log")
        ax[l].set_xscale("log")
        plt.legend()

        l += 1
plt.savefig('ps.png',bbox_inches='tight')


