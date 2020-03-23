"""
Print parameters in a text file.
"""

from parameters import *

# Save values of parameters
text_file = open("parameters.txt" % i, "a")
text_file.write("L = "+str(L)+'\n')
text_file.write("N = "+str(N)+'\n')
text_file.write("Bx = "+str(Bx)+'\n')
text_file.write("By = "+str(By)+'\n')
text_file.write("Re = "+str(Re)+'\n')
text_file.write("V = "+str(V)+'\n')
text_file.write("k = "+str(k)+'\n')
text_file.write("k scaling = "+str(k_scaling)+'\n')
text_file.write("ν = "+str(ν)+'\n')
text_file.write("dt = "+str(dt)+'\n')
text_file.write("stop_iteration = "+str(stop_iteration)+'\n')
text_file.write("snapshots_iter = "+str(snapshots_iter)+'\n')
text_file.write("slices_iter = "+str(slices_iter)+'\n')
text_file.write("scalars_iter = "+str(scalars_iter)+'\n\n')

# Net and training parameters
text_file.write("stacks = "+str(stacks)+'\n')
text_file.write("stack width = "+str(stack_width)+'\n')
text_file.write("filters base = "+str(filters_base)+'\n')
text_file.write("output channels = "+str(output_channels)+'\n')
text_file.write("kernel size = "+str(unet_kw['kernel_size'])+'\n')
text_file.write("activation = "+str(unet_kw['activation'])+'\n')
text_file.write("use bias = "+str(unet_kw['use_bias'])+'\n')
text_file.write("batch norm = "+str(unet_kw['batch_norm'])+'\n')
text_file.write("restore epoch = "+str(restore_epoch)+'\n')
text_file.write("epochs = "+str(epochs)+'\n')
text_file.write("snapshots = "+str(snapshots)+'\n')
text_file.write("testing size = "+str(testing_size)+'\n')
text_file.write("learning rate = "+str(learning_rate)+'\n')
text_file.write("diss cost = "+str(diss_cost)+'\n')
text_file.close()

