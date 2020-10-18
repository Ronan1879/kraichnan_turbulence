import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from parameters import *

#filename = 'data.csv'
train_file_name = "training_costs.npy"
test_file_name = "testing_costs.npy"

#epochs = np.loadtxt(filename,delimiter=',',skiprows=1)[:,0]
#train_cost = np.loadtxt(filename,delimiter=',',skiprows=1)[:,1]
#test_cost = np.loadtxt(filename,delimiter=',',skiprows=1)[:,2]

train_cost = np.load(train_file_name)/batch_size
test_cost = np.load(test_file_name)/testing_size

fig, ax = plt.subplots(1,1,figsize=(10,8))

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
ax.tick_params(direction='out',width=2,length=5,labelsize = 20)

plt.plot(train_cost,label="Training",color='b')
plt.plot(test_cost,label="Testing",ls='--',color='r')

plt.ylabel("Average cost per snapshot",fontsize=20)
plt.xlabel("Epochs",fontsize=20)
plt.yscale('log')
plt.legend(prop={'size': 14})
plt.savefig("cost.png")
