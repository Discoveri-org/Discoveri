##### Author      : Francesco Massimo (Laboratoire de Physique des Gaz et des Plasmas, CNRS)
##### Purpose     : read and visualize the history of the hyperparameters of the FST-PSO
import numpy as np
import matplotlib.pyplot as plt
import os,sys

########## Read data of positions and function values

starting_directory = os.getcwd()

# find the name of the optimization history file of delta to read
filename_delta = starting_directory+"/history_FSTPSO_delta.npy" # complete history
if (os.path.isfile(filename_delta)!=True): # the complete history file does not exist 
    # find all files with partial history of optimization 
    files_with_history = []
    for file in os.listdir(starting_directory):
        if (("history_FSTPSO_delta" in file) and ("iteration" in file)):
            files_with_history.append(file)
    files_with_history = sorted(files_with_history)
    # pick the most recent one
    filename_delta = files_with_history[len(files_with_history)-1]
    
history_delta  = np.load(filename_delta)

# find the name of the optimization history file of delta to read
filename_Phi = starting_directory+"/history_FSTPSO_Phi.npy" # complete history
if (os.path.isfile(filename_Phi)!=True): # the complete history file does not exist 
    # find all files with partial history of optimization 
    files_with_history = []
    for file in os.listdir(starting_directory):
        if (("history_FSTPSO_Phi" in file) and ("iteration" in file)):
            files_with_history.append(file)
    files_with_history = sorted(files_with_history)
    # pick the most recent one
    filename_Phi = files_with_history[len(files_with_history)-1]
    
history_Phi  = np.load(filename_Phi)

# find the name of the optimization history file of w to read
filename_w = starting_directory+"/history_FSTPSO_w.npy" # complete history
if (os.path.isfile(filename_w)!=True): # the complete history file does not exist 
    # find all files with partial history of optimization 
    files_with_history = []
    for file in os.listdir(starting_directory):
        if (("history_FSTPSO_w" in file) and ("iteration" in file)):
            files_with_history.append(file)
    files_with_history = sorted(files_with_history)
    # pick the most recent one
    filename_w = files_with_history[len(files_with_history)-1]
    
history_w  = np.load(filename_w)

# find the name of the optimization history file of c1 to read
filename_c1 = starting_directory+"/history_FSTPSO_c1.npy" # complete history
if (os.path.isfile(filename_c1)!=True): # the complete history file does not exist 
    # find all files with partial history of optimization 
    files_with_history = []
    for file in os.listdir(starting_directory):
        if (("history_FSTPSO_c1" in file) and ("iteration" in file)):
            files_with_history.append(file)
    files_with_history = sorted(files_with_history)
    # pick the most recent one
    filename_c1 = files_with_history[len(files_with_history)-1]
    
history_c1  = np.load(filename_c1)

# find the name of the optimization history file of c2 to read
filename_c2 = starting_directory+"/history_FSTPSO_c2.npy" # complete history
if (os.path.isfile(filename_c2)!=True): # the complete history file does not exist 
    # find all files with partial history of optimization 
    files_with_history = []
    for file in os.listdir(starting_directory):
        if (("history_FSTPSO_c2" in file) and ("iteration" in file)):
            files_with_history.append(file)
    files_with_history = sorted(files_with_history)
    # pick the most recent one
    filename_c2 = files_with_history[len(files_with_history)-1]
    
history_c2  = np.load(filename_c2)

# find the name of the optimization history file of U to read
filename_U = starting_directory+"/history_FSTPSO_U.npy" # complete history
if (os.path.isfile(filename_U)!=True): # the complete history file does not exist 
    # find all files with partial history of optimization 
    files_with_history = []
    for file in os.listdir(starting_directory):
        if (("history_FSTPSO_U" in file) and ("iteration" in file)):
            files_with_history.append(file)
    files_with_history = sorted(files_with_history)
    # pick the most recent one
    filename_U = files_with_history[len(files_with_history)-1]
    
history_U  = np.load(filename_U)

# find the name of the optimization history file of L to read
filename_L = starting_directory+"/history_FSTPSO_L.npy" # complete history
if (os.path.isfile(filename_L)!=True): # the complete history file does not exist 
    # find all files with partial history of optimization 
    files_with_history = []
    for file in os.listdir(starting_directory):
        if (("history_FSTPSO_L" in file) and ("iteration" in file)):
            files_with_history.append(file)
    files_with_history = sorted(files_with_history)
    # pick the most recent one
    filename_L = files_with_history[len(files_with_history)-1]
    
history_L  = np.load(filename_L)



### Plot
plt.ion()
iterations = [i for i in range(0,np.size(history_w[0,:]))]
number_of_particles = np.size(history_w[:,0])


plt.figure()
for iparticle in range(0,number_of_particles):
    plt.plot(iterations,history_delta[iparticle,:],marker=".",label="Sample "+str(iparticle))
plt.xlabel("Number of iterations")
plt.ylabel("delta")
plt.legend()

plt.figure()
for iparticle in range(0,number_of_particles):
    plt.plot(iterations,history_Phi[iparticle,:],marker=".",label="Sample "+str(iparticle))
plt.xlabel("Number of iterations")
plt.ylabel("Phi")
plt.legend()

plt.figure()
for iparticle in range(0,number_of_particles):
    plt.plot(iterations,history_w[iparticle,:],marker=".",label="Sample "+str(iparticle))
plt.xlabel("Number of iterations")
plt.ylabel("Inertia weight w")
plt.legend()

plt.figure()
for iparticle in range(0,number_of_particles):
    plt.plot(iterations,history_c1[iparticle,:],marker=".",label="Sample "+str(iparticle))
plt.xlabel("Number of iterations")
plt.ylabel("Acceleration coefficient c1")
plt.legend()

plt.figure()
for iparticle in range(0,number_of_particles):
    plt.plot(iterations,history_c2[iparticle,:],marker=".",label="Sample "+str(iparticle))
plt.xlabel("Number of iterations")
plt.ylabel("Acceleration coefficient c2")
plt.legend()

plt.figure()
for iparticle in range(0,number_of_particles):
    plt.plot(iterations,history_L[iparticle,:],marker=".",label="Sample "+str(iparticle))
plt.xlabel("Number of iterations")
plt.ylabel("Minimum absolute value of velocity L")
plt.legend()

plt.figure()
for iparticle in range(0,number_of_particles):
    plt.plot(iterations,history_U[iparticle,:],marker=".",label="Sample "+str(iparticle))
plt.xlabel("Number of iterations")
plt.ylabel("Maximum absolute value of velocity U")
plt.legend()






