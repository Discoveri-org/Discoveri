# This example shows the use of Particle Swarm Optimization 
# in an easy problem with a unique maximum, the same as in input2DParaboloidPSO.py,
# but with better hyperparameters. Note how this improves convergence compared 
# to using default hyperparameters in input2DParaboloidPSO.py. 

import numpy as np
import os,sys

path_discoveri = "/Users/francescomassimo/Codes/Optimization_on_cluster/Discoveri"
sys.path.insert(0, path_discoveri)

# import Discoveri class
from discoveriMain import createOptimizationRun


###########################################################################
############# Parameters for a generic optimization method ################
###########################################################################

#### Parameter space to explore
number_of_dimensions               = 2 
search_interval                    = [[-100.,100.],[-100.,100.]]
input_parameters_names             = ["dim0","dim1"]

number_of_samples_per_iteration    = 6 

#### Optimization parameters
number_of_iterations               = 50 

#### Diagnostic and output dump periodicity
iterations_between_outputs         = 10

#### Flag used to set if a numpy function or simulation results are optimized: 
#### if True it optimizes (i.e. maximizes) a numpy function defined in test_function
use_test_function                  = True

test_function                      = None
simulation_postprocessing_function = None

def my_test_function(x): # global maximum in (2.,2.)
    return -np.sum(np.square(x-2.))+9.
    
test_function                      = my_test_function


starting_directory = ""

###########################################################################
##################  Set the optimization method ###########################
###########################################################################

optimization_method                = "Particle Swarm Optimization" 
c1                                 = 0.4
c2                                 = 0.4
w1                                 = 0.8
w2                                 = 0.4

###########################################################################
####################### Run the optimization ##############################
###########################################################################


if __name__ == '__main__':
    
    starting_directory = os.getcwd()
    # initialize an optimization run 
    # the optimizer hyperparameters will be the default ones
    optimization_run   = createOptimizationRun(starting_directory              = starting_directory,              \
                                               optimization_method             = optimization_method,             \
                                               number_of_samples_per_iteration = number_of_samples_per_iteration, \
                                               number_of_dimensions            = number_of_dimensions,            \
                                               search_interval                 = search_interval,                 \
                                               number_of_iterations            = number_of_iterations,            \
                                               use_test_function               = use_test_function,               \
                                               test_function                   = test_function,                   \
                                               iterations_between_outputs      = iterations_between_outputs,      \
                                               input_parameters_names          = input_parameters_names,          \
                                               c1=c1.4,c2=c2,w1=w1,w2=w2)
    # execute optimization run
    optimization_run.execute()