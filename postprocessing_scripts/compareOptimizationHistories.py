##### Author      : Francesco Massimo (Laboratoire de Physique des Gaz et des Plasmas, CNRS)
##### Purpose     : compare optimization histories from multiple optimization runs in the same directory


import numpy as np
import matplotlib.pyplot as plt
import os, sys


##### Find the optimization runs to read 
##### assuming that all the directories in the present path have one optimization run for each directory

starting_directory = os.getcwd()

runs_to_read       = []
for file in os.listdir(starting_directory):
    directory_name = file
    if os.path.isdir(directory_name):
        runs_to_read.append(directory_name)

filename                                        = "/history_particles_positions_and_function_values.npy"
filename_time                                   = "/time_to_complete_iterations.npy"
#filename = "/history_particles_positions_and_function_values_iteration_00028.npy"

##### assuming that all runs have the same shape, find number of iterations and dimensions of first one
history_particles_positions_and_function_values = np.load(runs_to_read[0]+filename)
number_of_iterations,number_of_samples_per_iteration,number_of_dimensions = np.shape(history_particles_positions_and_function_values)
number_of_dimensions                                  = number_of_dimensions-1 # the last index in the file was used for the function to optimize
iterations                                      = np.linspace(0,number_of_iterations-1,num=number_of_iterations)

#### Read all data from each run
history_optimization_runs                       = np.zeros(shape=(number_of_iterations,len(runs_to_read)))
history_time_runs                               = np.zeros(shape=(number_of_iterations,len(runs_to_read)))

for run in runs_to_read: 
    index_run = runs_to_read.index(run)           
    # Read data of positions and function values
    history_particles_positions_and_function_values                 = np.load(run+filename)
    # Read data of positions
    history_time_to_complete_iterations                             = np.load(run+filename_time)

    ###### Find optimization history
    optimizer_maximum_of_function_to_optimize                       = np.zeros(number_of_iterations)
    optimizer_maximum_of_function_to_optimize_for_this_iteration    = np.zeros(number_of_iterations)

    optimizer_maximum_of_function_to_optimize[0]                    = np.amax(history_particles_positions_and_function_values[0,:,number_of_dimensions])
    optimizer_maximum_of_function_to_optimize_for_this_iteration[0] = optimizer_maximum_of_function_to_optimize[0]
    
    for iteration in range(0,number_of_iterations):
        optimizer_maximum_of_function_to_optimize[iteration]                    = np.amax(history_particles_positions_and_function_values[iteration,:,number_of_dimensions])
        optimizer_maximum_of_function_to_optimize_for_this_iteration[iteration] = optimizer_maximum_of_function_to_optimize[iteration]
        optimizer_maximum_of_function_to_optimize[iteration]                    = max(optimizer_maximum_of_function_to_optimize[iteration-1],optimizer_maximum_of_function_to_optimize[iteration])
        
        history_optimization_runs[iteration,index_run]                          = optimizer_maximum_of_function_to_optimize[iteration]
        history_time_runs[iteration,index_run]                                  = history_time_to_complete_iterations[iteration]
        
##### Plot averages of optimization history, individual optimization histories and error bars

color               = "b"
optimization_method = "PSO" #"Bayesian Optimization"#"Random Search with Halton Sequence"#"Bayesian Optimization"
plt.ion()



# as function of iteration number
plt.figure(1)
label = ""
for irun in range(0,len(runs_to_read)):
    if irun == 0:
        label="individual, "+optimization_method 
    else:
        label=""
    plt.plot(iterations,history_optimization_runs[:,irun],color=color,label=label,alpha=0.3,linewidth=0.8)
plt.plot(iterations,np.average(history_optimization_runs,axis=1 ),label="average, "+optimization_method ,color=color)
plt.fill_between(iterations,np.average(history_optimization_runs,axis=1)-np.std(history_optimization_runs,axis=1 ),np.average(history_optimization_runs,axis=1 )+np.std(history_optimization_runs,axis=1 ),color=color,alpha=0.1)
    
plt.xlabel("Iteration number")
plt.ylabel("Maximum objective function value found")
plt.legend()

# as function of time
plt.figure(2)
label = ""
for irun in range(0,len(runs_to_read)):
    if irun == 0:
        label="individual, "+optimization_method 
    else:
        label=""
    plt.plot(history_time_runs[:,irun]/3600.,history_optimization_runs[:,irun],color=color,label=label,alpha=0.3,linewidth=0.8)
    
average_time_for_iterations = np.average(history_time_runs,axis=1)
plt.plot(average_time_for_iterations/3600.,np.average(history_optimization_runs,axis=1 ),label="average, "+optimization_method ,color=color)
plt.fill_between(average_time_for_iterations/3600.,np.average(history_optimization_runs,axis=1 )-np.std(history_optimization_runs,axis=1 ),np.average(history_optimization_runs,axis=1 )+np.std(history_optimization_runs,axis=1 ),color=color,alpha=0.1)
    
plt.xlabel("Time lapsed [h]")
plt.ylabel("Maximum objective function value found")
plt.legend()

# as function of number of function evaluations
plt.figure(3)
label = ""
for irun in range(0,len(runs_to_read)):
    if irun == 0:
        label="individual, "+optimization_method 
    else:
        label=""
    plt.plot((iterations+1)*number_of_samples_per_iteration,history_optimization_runs[:,irun],color=color,label=label,alpha=0.3,linewidth=0.8)
plt.plot((iterations+1)*number_of_samples_per_iteration,np.average(history_optimization_runs,axis=1 ),label="average, "+optimization_method ,color=color)
plt.fill_between((iterations+1)*number_of_samples_per_iteration,np.average(history_optimization_runs,axis=1)-np.std(history_optimization_runs,axis=1 ),np.average(history_optimization_runs,axis=1 )+np.std(history_optimization_runs,axis=1 ),color=color,alpha=0.1)
    
plt.xlabel("Number of function evaluations")
plt.ylabel("Maximum objective function value found")
plt.legend()

    




#print("Optimum function value = ",optimum_function_value,", Found at position ",optimum_position)


