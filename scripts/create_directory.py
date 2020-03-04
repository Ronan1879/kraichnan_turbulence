"""
Print parameters and create folder file.

Type in terminal which type of simulation to create directory.
Either type "dns" or "ml"

@author: Ronan Legin
Feb. 29th 2020
"""

import os
import sys
import sim_parameters as param
import ml_parameters as param_ml

sim_type = sys.argv[1]

i = 0
while os.path.exists("./simulation_"+str(sim_type)+"_%s/" % i):
    i += 1
os.makedirs("./simulation_"+str(sim_type)+"_%s/" % i)

# Save values of parameters
text_file = open("./simulation_"+str(sim_type)+"_%s/parameters.txt" % i, "a")
text_file.write("L = "+str(param.L)+'\n')
text_file.write("N = "+str(param.N)+'\n')
text_file.write("Bx = "+str(param.Bx)+'\n')
text_file.write("By = "+str(param.By)+'\n')
text_file.write("Re = "+str(param.Re)+'\n')
text_file.write("V = "+str(param.V)+'\n')
text_file.write("k = "+str(param.k)+'\n')
text_file.write("ν = "+str(param.ν)+'\n')
text_file.write("dt = "+str(param.dt)+'\n')
text_file.write("stop_iteration = "+str(param.stop_iteration)+'\n')
text_file.write("snapshots_iter = "+str(param.snapshots_iter)+'\n')
text_file.write("slices_iter = "+str(param.slices_iter)+'\n')
text_file.write("scalars_iter = "+str(param.scalars_iter)+'\n\n')

if str(sim_type) == "ml":
	# Net and training parameters
	text_file.write("stacks = "+str(param_ml.stacks)+'\n')
	text_file.write("stack width = "+str(param_ml.stack_width)+'\n')
	text_file.write("filters base = "+str(param_ml.filters_base)+'\n')
	text_file.write("output channels = "+str(param_ml.output_channels)+'\n')
	text_file.write("kernel size = "+str(param_ml.unet_kw['kernel_size'])+'\n')
	text_file.write("activation = "+str(param_ml.unet_kw['activation'])+'\n')
	text_file.write("use bias = "+str(param_ml.unet_kw['use_bias'])+'\n')
	text_file.write("batch norm = "+str(param_ml.unet_kw['batch_norm'])+'\n')
	text_file.write("restore epoch = "+str(param_ml.restore_epoch)+'\n')
	text_file.write("epochs = "+str(param_ml.epochs)+'\n')
	text_file.write("snapshots = "+str(param_ml.snapshots)+'\n')
	text_file.write("testing size = "+str(param_ml.testing_size)+'\n')
	text_file.write("learning rate = "+str(param_ml.learning_rate)+'\n')
	text_file.write("diss cost = "+str(param_ml.diss_cost)+'\n')
text_file.close()

