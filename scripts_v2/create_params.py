import numpy as np
import itertools
import os


stacks_opt = [1]
stack_width_opt = [1,2,3,4]
filter_base_opt = [12,24,48,96]
kernel_size_opt = [1,3,5,7]

comb_per_file = 1

opts = [stacks_opt,stack_width_opt,filter_base_opt,kernel_size_opt]
combinations = list(itertools.product(*opts))

i = 1
for combination in combinations:
    np.savetxt("unet_params/unet_param_{}.txt".format(i),np.array(combination))
    i += 1
