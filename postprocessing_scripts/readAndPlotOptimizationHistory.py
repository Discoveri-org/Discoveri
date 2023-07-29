##### Author      : Francesco Massimo (Laboratoire de Physique des Gaz et des Plasmas, CNRS)
##### Purpose     : read and visualize the optimization history of only one optimization run
##### Last update : 29/07/2023

import numpy as np
import matplotlib.pyplot as plt
import os,sys

########## Read data of positions and function values

starting_directory = os.getcwd()

# find the name of the optimization history file to read
filename = starting_directory+"/history_particles_positions_and_function_values.npy" # complete history
if (os.path.isfile(filename)!=True): # the complete history file does not exist 
    # find all files with partial history of optimization 
    files_with_history = []
    for file in os.listdir(starting_directory):
        if (("history" in file) and ("iteration" in file)):
            files_with_history.append(file)
    files_with_history = sorted(files_with_history)
    # pick the most recent one
    filename = files_with_history[len(files_with_history)-1]
    
# Read data of positions and function values, extract number of dimensions, iterations and samples

history_particles_positions_and_function_values = np.load(filename)

number_of_iterations,num_particles,num_dimensions = np.shape(history_particles_positions_and_function_values)

num_dimensions = np.size(history_particles_positions_and_function_values[0,0,:])-1

num_samples    = np.size(history_particles_positions_and_function_values[0,:,0])

iterations = np.linspace(0,number_of_iterations-1,num=number_of_iterations)


########## Print optimum over all optimization history
optimum_function_value       = np.amax(history_particles_positions_and_function_values[:,:,num_dimensions])
index_optimum_function_value = np.argwhere(history_particles_positions_and_function_values[:,:,num_dimensions] == optimum_function_value)[0]
optimum_position             = history_particles_positions_and_function_values[index_optimum_function_value[0],index_optimum_function_value[1],0:num_dimensions]
best_configuration_number    = index_optimum_function_value[0]*num_samples+index_optimum_function_value[1]

print("\n Optimum function value = ",optimum_function_value,", Found at position ",optimum_position," in Configuration ",best_configuration_number,"\n" )


########## Plot data from optimization history of the samples
plt.ion()

###  plot the value of the function to optimize of all samples over the iterations
plt.figure()
for isample in range(0,num_samples):
    plt.plot(iterations,history_particles_positions_and_function_values[:,isample,-1],marker='.',label="Sample "+str(isample))
plt.ylabel("Function value")
plt.xlabel("Iteration number")
plt.legend()

### plot the convergence plot, i.e. the maximum value of the function to optimize (for the current iteration and for all the optimization history)

optimizer_maximum_of_function_to_optimize                       = np.zeros(number_of_iterations)
optimizer_maximum_of_function_to_optimize_for_this_iteration    = np.zeros(number_of_iterations)

optimizer_maximum_of_function_to_optimize[0]                    = np.amax(history_particles_positions_and_function_values[0,:,num_dimensions])
optimizer_maximum_of_function_to_optimize_for_this_iteration[0] = optimizer_maximum_of_function_to_optimize[0]


for iteration in range(1,number_of_iterations):
    optimizer_maximum_of_function_to_optimize[iteration]                    = np.amax(history_particles_positions_and_function_values[iteration,:,num_dimensions])
    optimizer_maximum_of_function_to_optimize_for_this_iteration[iteration] = optimizer_maximum_of_function_to_optimize[iteration]
    optimizer_maximum_of_function_to_optimize[iteration]                    = max(optimizer_maximum_of_function_to_optimize[iteration-1],optimizer_maximum_of_function_to_optimize[iteration])

plt.figure()
plt.scatter(iterations,optimizer_maximum_of_function_to_optimize_for_this_iteration,label="Optimizer maximum at this iteration")
plt.plot(iterations,optimizer_maximum_of_function_to_optimize,label="Optimizer maximum until now",marker='.')
plt.xlabel("Iteration number")
plt.ylabel("Function value")


### plots used only in some special cases for the number of dimensions

# optimization in a 1-dimensional parameter space
if (num_dimensions==1):
    
    # visualize how the samples move in the 1-dimensional space in the whole optimization history
    plt.figure()
    for iparticle in range(0,num_particles):
        plt.plot(iterations,history_particles_positions_and_function_values[:,iparticle,0],label="Particle "+str(iparticle),marker='.')
    plt.xlabel("Iteration number")
    plt.ylabel("Dimension 0")
    plt.legend()
    
    # visualize the sampling of the function 
    plt.figure()
    plt.scatter(history_particles_positions_and_function_values[:,:,0],history_particles_positions_and_function_values[:,:,1])
    plt.xlabel("Dimension 0")
    plt.ylabel("Function value")
    plt.xlim()

elif (num_dimensions==2):
    
    # visualize how the samples move in the 1-dimensional space in the whole optimization history
    plt.figure()
    for iparticle in range(0,num_particles):
        plt.plot(history_particles_positions_and_function_values[:,iparticle,0],history_particles_positions_and_function_values[:,iparticle,1],label="Particle "+str(iparticle),marker='.')
    plt.xlabel("Dimension 0")
    plt.ylabel("Dimension 1")
    plt.legend()
    
    # visualize the sampling of the function, the colorbar tells the function value    
    plt.figure()
    im=plt.scatter(history_particles_positions_and_function_values[:,:,0],history_particles_positions_and_function_values[:,:,1],c=history_particles_positions_and_function_values[:,:,2])
    plt.xlabel("Dimension 0")
    plt.ylabel("Dimension 1")
    clb = plt.colorbar(im)
    clb.ax.set_title('Function value')
    plt.legend()
    
########## Read data of time needed to complete the iterations
# find the name of the optimization history file to read
filename_time = starting_directory+"/time_to_complete_iterations.npy" # complete history
if (os.path.isfile(filename_time)!=True): # the complete history file does not exist 
    # find all files with partial history of optimization 
    files_with_time = []
    for file in os.listdir(starting_directory):
        if (("time" in file) and ("iteration" in file)):
            files_with_time.append(file)
    files_with_time = sorted(files_with_time)
    # pick the most recent one
    filename_time = files_with_time[len(files_with_time)-1]

data_time_to_complete_iterations = np.load(filename_time)
plt.figure()
plt.plot(iterations,data_time_to_complete_iterations/3600.,marker=".")
plt.xlabel("Number of iterations")
plt.ylabel("Time lapsed [h]")



