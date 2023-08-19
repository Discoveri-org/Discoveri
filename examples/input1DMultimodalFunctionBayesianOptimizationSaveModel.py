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
number_of_dimensions               = 1 
search_interval                    = [[0.,10.]]
input_parameters_names             = ["dim0"]

number_of_samples_per_iteration    = 1

#### Optimization parameters
number_of_iterations               = 25

#### Bayesian Optimization parameters
# this parameter is particularly important especially in more than one dimension
# see its description in the doc and in the links it provides
length_scale                       = 1.0 

#### Diagnostic and output dump periodicity
iterations_between_outputs         = 1

#### Flag used to set if a numpy function or simulation results are optimized: 
#### if True it optimizes (i.e. maximizes) a numpy function defined in test_function
use_test_function                  = True

test_function                      = None
simulation_postprocessing_function = None

def my_test_function(x): # global maximum near (4.5,4.5)
    return np.sum( -np.cos(x)-np.sin(x)-5/2.*np.cos(2.*x)+1/2.*np.sin(2.*x)  )
    
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
    
    
    # Save the optimizer predictive model to a file
    with open('model.pkl', 'wb') as file:
        pickle.dump(optimization_run.optimizer.model, file)
        
        
    # You can load the predictive model after the optimization
    with open('model.pkl', 'rb') as file:
        loaded_prediction_model = pickle.load(file)
    
    # load the optimization history
    filename             = "history_particles_positions_and_function_values.npy" # complete history
    history_particles_positions_and_function_values = np.load(filename)
        
    # and use the surrogate model of bayesian optimization to have a reconstruction of the function to optimize
    n_grid_points        = 300
    x_mesh               = np.linspace(search_interval[0][0],search_interval[0][1],num=n_grid_points)
    search_interval_size = [search_interval[idim][1]-search_interval[idim][0] for idim in range(0,number_of_dimensions)]

    # array for the predicted function values
    function_value_mesh  = np.zeros(n_grid_points)
    # array for the uncertainity of the prediction as standard deviation
    std = np.zeros(n_grid_points)
    for i in range(0,n_grid_points):
        # remember that the surrogate model inside the optimizer takes for each dimension idim the coordinates
        # of the sample normalized by search_interval_size[idim], i.e. the size of the search interval in that dimension
        sample = (np.array([x_mesh[i]/search_interval_size[0]])).reshape(1,1)
        # predict the value of the function with a surrogate model
        function_value_mesh[i],std[i] = loaded_prediction_model.predict(sample,return_std=True)
           
            
    # Plot    
    plt.figure();plt.ion();plt.show()
    # plot the predicted values
    plt.plot(x_mesh,function_value_mesh,"g",label="surrogate model")
    # plot also the uncertainity of the prediction at each point
    plt.fill_between(x_mesh,function_value_mesh-3*std,function_value_mesh+3*std,color="g",alpha=0.1)
    # print the real value of the function to optimize (which is unknown to the optimizer)
    plt.plot(x_mesh,[my_test_function(x) for x in x_mesh],"-b",label="function to optimize")
    
    # print the points sampled by the optimizer, remembering to the normalize the X points
    
    # positions sampled during the optimization run
    X_sampled = np.reshape(history_particles_positions_and_function_values[:,:,0:number_of_dimensions],(number_of_iterations*number_of_samples_per_iteration,number_of_dimensions))
    # function values sampled during the optimization run
    y_sampled = history_particles_positions_and_function_values[:,:,number_of_dimensions]
    
    plt.plot(X_sampled,y_sampled,"b.",label="sampled points")
    plt.xlabel("x");plt.ylabel("y")
    plt.legend()

    
       
