"""
Create a png image of a single cost function.

file1 is associated with training data, fil2 with testing data.

Example : python plot_single_cost.py training_costs_pi.npy testing_costs_pi.npy

Usage:
    python plot_single_cost.py <file1> <file2>

"""

import numpy as np
import matplotlib.pyplot as plt
from parameters import *

import sys

training_data = str(sys.argv[1])
testing_data = str(sys.argv[2])

train_cost = np.load(training_data)
test_cost = np.load(testing_data)

plt.plot(train_cost,label="Training")
plt.plot(test_cost,label="Testing",ls='--')

plt.ylabel("Cost",fontsize=14)
plt.xlabel("Epochs",fontsize=14)
plt.yscale('log')
plt.legend()
plt.savefig("single_cost.png",dpi=300)
plt.show()