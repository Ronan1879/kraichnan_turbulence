
import matplotlib.pyplot as plt
import unet
import numpy as np
import parameters as param
from parameters import *
import tensorflow as tf
from dedalus import public as de
import post as post_tools
from misc_functions import deviatoric_part
from misc_functions import array_of_tf_components


def pspec(k,field):
    pspec = (np.abs(field['c'])**2 + np.abs(field['c'])**2)/2

    n_count,bins = np.histogram(k.flatten(),bins=2*param.N)

    n, bins = np.histogram(k.flatten(),bins=2*param.N,weights=pspec.flatten())

    return bins, n/n_count


model = unet.Unet(stacks,stack_width,filters_base,param.output_channels, **unet_kw)
checkpoint = tf.train.Checkpoint(net=model)

train_path = 'training_costs.npy'
output_path = 'ps.png'

training_costs = np.load(train_path)

best_epoch = np.where(training_costs == np.min(training_costs))[0][0]

best_epoch -= best_epoch % save_interval

restore_path = f"{checkpoint_path}-{best_epoch//save_interval}"
checkpoint.restore(restore_path)#.assert_consumed()
print('Restored from {}'.format(restore_path))

inputs = np.load("inputs.npy")[2]
labels = np.load("labels.npy")[2]

inputs = inputs[np.newaxis,...]
labels = labels[np.newaxis,...]

tf_inputs = [tf.cast(inputs,datatype)]
tf_labels = tf.cast(labels,datatype)
tf_outputs = model.call(tf_inputs)

tau_pred = deviatoric_part(array_of_tf_components(tf_outputs))
tau_true = deviatoric_part(array_of_tf_components(tf_labels))

x_basis = de.Fourier('x', N, interval=Bx, dealias=3/2)
y_basis = de.Fourier('y', N, interval=By, dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64, comm=None)


dx = domain.bases[0].Differentiate
dy = domain.bases[1].Differentiate

kx = domain.elements(0)
ky = domain.elements(1)

k = np.sqrt(kx**2 + ky**2)

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
plt.savefig(output_path,bbox_inches='tight')
plt.clf()


