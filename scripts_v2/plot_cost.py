
import numpy as np
import matplotlib.pyplot as plt
from parameters import *


train_path = "training_costs.npy"
test_path = "testing_costs.npy"

train_cost = np.load(train_path)
test_cost = np.load(test_path)

fig, ax = plt.subplots(1,1,figsize=(10,8))

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
ax.tick_params(direction='out',width=2,length=5,labelsize = 20)

plt.plot(train_cost,label="Training",color='b')
plt.plot(test_cost,label="Testing",ls='--',color='r')

plt.ylabel("Cost",fontsize=20)
plt.xlabel("Epochs",fontsize=20)
plt.yscale('log')
plt.legend(prop={'size': 14})
plt.savefig("cost.png",dpi=300)