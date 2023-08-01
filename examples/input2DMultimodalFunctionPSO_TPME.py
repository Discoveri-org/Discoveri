##### Author      : Francesco Massimo (Laboratoire de Physique des Gaz et des Plasmas, CNRS)
##### Purpose     : inputs for the optimization run


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


#### Optimization method options: 
#### - "Particle Swarm Optmization"
#### - "Random Search"
#### - "Bayesian optimization"
#### - "IAPSO"
#### - "PSO-TPME"
optimization_method                     = "PSO-TPME" 

#### Parameter space to explore
number_of_dimensions                    = 2 
search_interval                         = [[0.,1.],[0.,1.]]

number_of_samples_per_iteration         = 30 

#### Optimization parameters

number_of_iterations                    = 20 
iterations_between_outputs              = 1

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
    return np.sum(   np.multiply(  np.square(position), np.power(np.sin(5 * np.pi * position),6.0)  )  )
    
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
                                               amplitude_mutated_range_2       = 0.1 )
    # execute optimization run
    optimization_run.execute()