##### Author      : Francesco Massimo (Laboratoire de Physique des Gaz et des Plasmas, CNRS)
##### Purpose     : read and visualize the history of the hyperparameters of the Adaptive Particle Swarm Optimization
import numpy as np
import matplotlib.pyplot as plt
import os,sys

########## Read data of positions and function values

starting_directory = os.getcwd()


# find the name of the optimization history file of f to read
filename_f = starting_directory+"/history_evolutionary_factor_f.npy" # complete history
if (os.path.isfile(filename_f)!=True): # the complete history file does not exist 
    # find all files with partial history of optimization 
    files_with_history = []
    for file in os.listdir(starting_directory):
        if (("history_evolutionary_factor_f" in file) and ("iteration" in file)):
            files_with_history.append(file)
    files_with_history = sorted(files_with_history)
    # pick the most recent one
    filename_f = files_with_history[len(files_with_history)-1]
    
history_f  = np.load(filename_f)


# find the name of the optimization history file of c1 to read
filename_c1 = starting_directory+"/history_c1.npy" # complete history
if (os.path.isfile(filename_c1)!=True): # the complete history file does not exist 
    # find all files with partial history of optimization 
    files_with_history = []
    for file in os.listdir(starting_directory):
        if (("history_c1" in file) and ("iteration" in file)):
            files_with_history.append(file)
    files_with_history = sorted(files_with_history)
    # pick the most recent one
    filename_c1 = files_with_history[len(files_with_history)-1]
    
history_c1  = np.load(filename_c1)

# find the name of the optimization history file of c2 to read
filename_c2 = starting_directory+"/history_c2.npy" # complete history
if (os.path.isfile(filename_c2)!=True): # the complete history file does not exist 
    # find all files with partial history of optimization 
    files_with_history = []
    for file in os.listdir(starting_directory):
        if (("history_c2" in file) and ("iteration" in file)):
            files_with_history.append(file)
    files_with_history = sorted(files_with_history)
    # pick the most recent one
    filename_c2 = files_with_history[len(files_with_history)-1]
    
history_c2  = np.load(filename_c2)


# find the name of the optimization history file of w to read
filename_w = starting_directory+"/history_w.npy" # complete history
if (os.path.isfile(filename_w)!=True): # the complete history file does not exist 
    # find all files with partial history of optimization 
    files_with_history = []
    for file in os.listdir(starting_directory):
        if (("history_w" in file) and ("iteration" in file)):
            files_with_history.append(file)
    files_with_history = sorted(files_with_history)
    # pick the most recent one
    filename_w = files_with_history[len(files_with_history)-1]
    
history_w  = np.load(filename_w)



iterations = [i for i in range(0,np.size(history_w))]

plt.ion();plt.figure()
plt.plot(iterations,history_c1,label="c1",marker=".")
plt.plot(iterations,history_c2,label="c2",marker=".")
plt.plot(iterations,history_f ,label="f",marker=".")
plt.plot(iterations,history_w ,label="w",marker=".")
plt.xlabel("Iteration Number")
plt.legend()