# Machine learning 2D incompressible turbulence

This repository contains the scripts used in simulating 2D incompressible turbulence as well as using machine learning to correct coarsed grained simulations


Here is a description of the functionality of each python script :

# closure_term.py

Computes the closure term for the vorticity formalism of the Navier-Stokes equation from the stress tensor.

# initial_field.py

Generates initial vorticity field based on random noise with fourier scaling. Computes the ux and uy velocity fields based on it.

# parameters.py

Contains all parameters used by simulations, machine learning model training, etc.

# plot_single_cost.py

Plots cost function of a single machine learning model.

# plot_multiple_cost.py

Plots cost function of multiple machine learning models.

# plot_snapshot.py

Plots a snapshot from the simulation

# plot_spectrum.py

Plots kinetic energy power spectrum of a simulation snapshot

# print_params.py

Prints in a .txt file all the current parameters

# simulation.py

Runs simulation. In parameters.py, there is a boolean parameter called "correct_simulation" that will determine if the user wants to apply machine learning correction to the simulation.

# ml_forcing.py

Computes the correction predicted by a trained machine learning model and returns the updated forcing to low_res_simulation.py

# leith_forcing.py

Computes the correction predicted by the Leith model and returns the updated forcing to low_res_simulation.py

# smagorinsky_forcing.py

Computes the correction predicted by the Smagorinsky model and returns the updated forcing to low_res_simulation.py

# train_unet_leith.py

Trains the neural network in predicting the Leith model instead of the subgrid stress tensor
