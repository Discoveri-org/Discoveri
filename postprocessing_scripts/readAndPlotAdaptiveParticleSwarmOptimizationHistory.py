##### Author      : Francesco Massimo (Laboratoire de Physique des Gaz et des Plasmas, CNRS)
##### Purpose     : read and visualize the history of the hyperparameters of the Adaptive Particle Swarm Optimization
import numpy as np
import matplotlib.pyplot as plt
import os,sys

########## Read data of positions and function values

starting_directory = os.getcwd()

# read history files
filename   = starting_directory+"/history_evolutionary_factor_f.npy" # complete history
history_f  = np.load(filename)

filename   = starting_directory+"/history_c1.npy" # complete history
history_c1 = np.load(filename)

filename   = starting_directory+"/history_c2.npy" # complete history
history_c2 = np.load(filename)

filename   = starting_directory+"/history_w.npy" # complete history
history_w  = np.load(filename)

iterations = [i for i in range(0,np.size(history_w))]

plt.ion();plt.figure()
plt.plot(iterations,history_c1,label="c1",marker=".")
plt.plot(iterations,history_c2,label="c2",marker=".")
plt.plot(iterations,history_f ,label="f",marker=".")
plt.plot(iterations,history_w ,label="w",marker=".")
plt.xlabel("Iteration Number")
plt.legend()