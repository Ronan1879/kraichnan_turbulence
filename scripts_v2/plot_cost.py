
import numpy as np
import matplotlib.pyplot as plt
from parameters import *

filename = 'data.csv'

#epochs = np.loadtxt(filename,delimiter=',',skiprows=1)[:,0]
train_cost = np.loadtxt(filename,delimiter=',',skiprows=1)[:,1]
test_cost = np.loadtxt(filename,delimiter=',',skiprows=1)[:,2]

fig, ax = plt.subplots(1,1,figsize=(10,8))

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
ax.tick_params(direction='out',width=2,length=5,labelsize = 20)

plt.plot(train_cost,label="Training",color='b',alpha=0.5)
plt.plot(test_cost,label="Testing",color='r')

plt.ylabel("Cost",fontsize=20)
plt.xlabel("Epochs",fontsize=20)
plt.yscale('log')
plt.legend(prop={'size': 14})
plt.savefig("cost.png",dpi=300)