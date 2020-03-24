"""
Create a png image of multiple cost functions. 

The cost functions must only be different in number of neurons used.

Specify the number of neurons of each model in order.

Separate folders with the name "(number of neurons)_neurons" must exist.

Example : python plot_multiple_cost.py (20,30,40,50,60)

Usage:
    python plot_multiple_cost.py <list>

"""

import numpy as np
import matplotlib.pyplot as plt
from parameters import *

import sys
neurons = []

for i in range(len(sys.argv)):
	if i != 0:
		neurons.append(sys.argv[i])


train_loss = np.zeros((len(neurons),epochs))
test_loss = np.zeros((len(neurons),epochs))
labels = []
i = 0
for n in neurons:
	train_loss[i,:] = np.load("{}_neurons/training_costs_pi.npy".format(str(n)))
	test_loss[i,:] = np.load("{}_neurons/testing_costs_pi.npy".format(str(n)))
	labels.append("{}_neurons".format(str(n)))
	i += 1
colors = ["r","b","g","k","m","lime","brown","yellow","cyan","darkorange"]

for i in range(0,len(neurons)):
	plt.plot(train_loss[i,:],label=labels[i],color=colors[i])
	plt.plot(test_loss[i,:],color=colors[i],ls='--',alpha=0.5)

#plt.title("Training DNS : 2048pix, Filtered : 256pix",fontsize=14)
plt.ylabel("Cost",fontsize=14)
plt.xlabel("Epochs",fontsize=14)
plt.legend()
plt.yscale('log')
plt.savefig("multiple_cost.png",dpi=300)
