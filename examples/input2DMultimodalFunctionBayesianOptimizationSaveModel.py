# This example shows the use of a Bayesian Optimization
# with a function which has  multiple peaks


import pickle
import numpy as np
import os,sys
import matplotlib.pyplot as plt 




path_discoveri = "/Users/francescomassimo/Codes/Optimization_on_cluster/Discoveri"
sys.path.insert(0, path_discoveri)

# import Discoveri class
from discoveriMain import createOptimizationRun
# import library to analyse Smilei simulation
from toolsSmileiAnalysis import *


###########################################################################
############# Parameters for a generic optimization method ################
###########################################################################

optimization_method                = "Bayesian Optimization" 

#### Parameter space to explore
number_of_dimensions               = 2 
search_interval                    = [[0.,10.],[0.,10.]]
input_parameters_names             = ["dim0","dim1"]

number_of_samples_per_iteration    = 1

#### Optimization parameters
number_of_iterations               = 50

#### Bayesian Optimization parameters
# This parameter is particularly important especially in more than one dimension
# See its description in the doc and in the links it provides.
# Check how the figures and the optimization results change if you choose length_scale = 1 or length_scale = 0.001
length_scale                       = 0.1


#### Diagnostic and output dump periodicity
iterations_between_outputs         = 1

#### Flag used to set if a numpy function or simulation results are optimized: 
#### if True it optimizes (i.e. maximizes) a numpy function defined in test_function
use_test_function                  = True

test_function                      = None
simulation_postprocessing_function = None

def my_test_function(x): # maximum near (4.5,4.5)
    return np.sum(np.sinc(x-5.2))
    
test_function                      = my_test_function


starting_directory = ""

if __name__ == '__main__':
    
    starting_directory = os.getcwd()
    # initialize an optimization run 
    # the optimizer hyperparameters will be the default ones
    optimization_run   = createOptimizationRun(starting_directory              = starting_directory,               \
                                               optimization_method             = optimization_method,              \
                                               number_of_samples_per_iteration = number_of_samples_per_iteration,  \
                                               number_of_dimensions            = number_of_dimensions,             \
                                               search_interval                 = search_interval,                  \
                                               number_of_iterations            = number_of_iterations,             \
                                               use_test_function               = use_test_function,                \
                                               test_function                   = test_function,                    \
                                               iterations_between_outputs      = iterations_between_outputs,       \
                                               input_parameters_names          = input_parameters_names,           \
                                               length_scale                    = length_scale )
    # execute optimization run
    optimization_run.execute()
    
    
    # Save the optimization_run object to a file
    with open('optimization_run.pkl', 'wb') as file:
        pickle.dump(optimization_run, file)
        
        
    # You can load the file later
    with open('optimization_run.pkl', 'rb') as file:
        loaded_optimization_run = pickle.load(file)
        
    n_grid_points = 200
    x_mesh = np.linspace(loaded_optimization_run.optimizer.search_interval[0][0],loaded_optimization_run.optimizer.search_interval[0][1],num=n_grid_points)
    y_mesh = np.linspace(loaded_optimization_run.optimizer.search_interval[1][0],loaded_optimization_run.optimizer.search_interval[1][1],num=n_grid_points)

    # array for the predicted function values
    predicted_function_value_mesh = np.zeros(shape=(n_grid_points,n_grid_points))
    # array for the real function values
    true_function_value_mesh      = np.zeros(shape=(n_grid_points,n_grid_points))
    for i in range(0,n_grid_points):
        for j in range(0,n_grid_points):
            # remember that the surrogate model inside the optimizer takes for each dimension idim the coordinates
            # of the sample normalized by search_interval_size[idim], i.e. the size of the search interval in that dimension
            x_sample              = x_mesh[i]/loaded_optimization_run.optimizer.search_interval_size[0]
            y_sample              = y_mesh[j]/loaded_optimization_run.optimizer.search_interval_size[1]
            sample_normalized     = (np.array([x_sample,y_sample])).reshape(1,2)
            # predict the value of the function with a surrogate model
            predicted_function_value_mesh[i,j]      = loaded_optimization_run.optimizer.model.predict(sample_normalized)
            sample                = np.array([x_mesh[i],y_mesh[j]])
            true_function_value_mesh[i,j] = my_test_function(sample)
           
            
    # plot the predicted function values
    # especially in more than one dimension, you will need a lot of points to see something in this plot
    plt.ion()
    plt.figure(1)
    plt.imshow(np.flipud(predicted_function_value_mesh),extent=[search_interval[0][0],search_interval[0][1],search_interval[1][0],search_interval[1][1]],aspect="auto",vmin=0,vmax=2)
    plt.xlabel("dim0");plt.ylabel("dim1")
    plt.title("Predicted function values")
    #plt.scatter(loaded_optimization_run.optimizer.X[:,0]*loaded_optimization_run.optimizer.search_interval_size[0],loaded_optimization_run.optimizer.X[:,1]*loaded_optimization_run.optimizer.search_interval_size[1],c=loaded_optimization_run.optimizer.y)
    plt.colorbar()
    
    # plot the real function values
    plt.figure(2)
    plt.imshow(np.flipud(true_function_value_mesh),extent=[search_interval[0][0],search_interval[0][1],search_interval[1][0],search_interval[1][1]],aspect="auto",vmin=0,vmax=2)
    plt.xlabel("dim0");plt.ylabel("dim1")
    plt.title("True function values")
    #plt.scatter(loaded_optimization_run.optimizer.X[:,0]*loaded_optimization_run.optimizer.search_interval_size[0],loaded_optimization_run.optimizer.X[:,1]*loaded_optimization_run.optimizer.search_interval_size[1],c=loaded_optimization_run.optimizer.y)
    plt.colorbar()
    
    plt.show()

    
       
