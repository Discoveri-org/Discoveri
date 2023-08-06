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

optimization_method                     = "Bayesian Optimization" 

#### Parameter space to explore
number_of_dimensions                    = 1 
search_interval                         = [[-15.,15.]]
input_parameters_names                  = ["dim0"]

number_of_samples_per_iteration         = 1

#### Optimization parameters
number_of_iterations                    = 25

#### Diagnostic and output dump periodicity
iterations_between_outputs              = 1

#### Flag used to set if a numpy function or simulation results are optimized: 
#### if True it optimizes (i.e. maximizes) a numpy function defined in test_function
#### Otherwise, it will postprocess simulation results using the function defined in simulation_postprocessing_function
#### In both cases it is absolutely necessary that the function to optimize gives a number, different from inf,-inf and from nan
#### It is also suggested, especially for Bayesian Optimization, to reduce the orders of magnitude spanned by the function, e.g. with a logarithm
use_test_function                       = True

test_function                           = None
simulation_postprocessing_function      = None

def my_test_function(position):
    return np.sum(  np.sinc(x-3.) )
    
test_function                           = my_test_function

###########################################################################
##################### Parameters for Random Search ########################
###########################################################################

#### parameters used only if Random Search is used, otherwise they are ignored
# if True, a scrambled Halton sequence will generate the random samples

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
                                               input_parameters_names          = input_parameters_names )
    # execute optimization run
    optimization_run.execute()
    
    
    # Save the optimization_run object to a file
    with open('optimization_run.pkl', 'wb') as file:
        pickle.dump(optimization_run, file)
        
        
    # You can load the file later
    with open('optimization_run.pkl', 'rb') as file:
        loaded_optimization_run = pickle.load(file)
        
    # and use the surrogate model of bayesian optimization to have a reconstruction of the function to optimize
    n_grid_points = 300
    x_mesh = np.linspace(loaded_optimization_run.optimizer.search_interval[0][0],loaded_optimization_run.optimizer.search_interval[0][1],num=n_grid_points)
  
    function_value_mesh = np.zeros(n_grid_points)
    std = np.zeros(n_grid_points)
    for i in range(0,n_grid_points):
        sample = (np.array([x_mesh[i]])).reshape(1,1)
        function_value_mesh[i],std[i] = loaded_optimization_run.optimizer.model.predict(sample,return_std=True)
           
            
    # plot    
    plt.figure();plt.ion();plt.show()
    plt.plot(x_mesh,function_value_mesh,"r",label="surrogate model")
    plt.fill_between(x_mesh,function_value_mesh-3*std,function_value_mesh+3*std,color="r",alpha=0.1)
    plt.plot(x_mesh,[my_test_function(x) for x in x_mesh],"-b",label="function to optimize")
    plt.plot(loaded_optimization_run.optimizer.X,loaded_optimization_run.optimizer.y,"b.",label="sampled points")
    plt.xlabel("x");plt.ylabel("y")
    plt.legend()

    
       
