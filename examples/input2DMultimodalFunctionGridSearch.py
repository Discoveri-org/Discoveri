# This example shows the use of a classic Grid Search
# with a function which has multiple peaks


import numpy as np
import os,sys

path_discoveri = "/Users/francescomassimo/Codes/Optimization_on_cluster/Discoveri"
sys.path.insert(0, path_discoveri)

# import Discoveri class
from discoveriMain import createOptimizationRun
# import library to analyse Smilei simulation
from toolsSmileiAnalysis import *


###########################################################################
############# Parameters for a generic optimization method ################
###########################################################################

#### Parameter space to explore
number_of_dimensions               = 2  
search_interval                    = [[0.,10.],[0.,10.]]
input_parameters_names             = ["dim0","dim1"]

number_of_samples_per_iteration    = 1800

samples_per_dimension              = [30,60]

#### Optimization parameters
number_of_iterations               = 1

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

###########################################################################
##################  Set the optimization method ###########################
###########################################################################

optimization_method                = "Grid Search" 

###########################################################################
######################## Run the optimization #############################
###########################################################################

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
                                               samples_per_dimension           = samples_per_dimension,            \
                                               input_parameters_names          = input_parameters_names )
    # execute optimization run
    optimization_run.execute()
