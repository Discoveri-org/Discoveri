##### Author  : Francesco Massimo (Laboratoire de Physique des Gaz et des Plasmas, CNRS, Université Paris Saclay)
##### Purpose : show the optimization on a multi-cpu laptop, workstation or cluster

import numpy as np
import os,sys
import pickle

############# Parameters for a generic optimization method 

optimization_method                        = "Bayesian Optimization"

#### Parameter space to explore
number_of_dimensions                       = 2 
input_parameters_names                     = ["bunch_charge_pC","electron_bunch_delay_behind_laser_um"]
search_interval                            = [[20,100],[15,25]]

#### Optimization parameters

number_of_iterations                       = 20 
number_of_samples_per_iteration            = 1  
iterations_between_outputs                 = 1

#### Diagnostic and output dump periodicity
iterations_between_outputs                 = 1

#### Simulation job managing

### Commands and parameters to launch and check simulation jobs
# command to submit job to the system (e.g. SLURM, Condor)
command_to_launch_jobs                     = "source launch_Smilei_simulation.sh"
# name of the file where the word marking the end of the simulation can be found
name_log_file_simulations                  = "smilei.log"
# if this word appears in name_log_file_simulations, the simulation can be postprocessed
word_marking_end_of_simulation_in_log_file = "END"

# time in seconds to wait between the creation of the sample directories (and in case launching the simulations)
# and the analysis yielding the value of the function for the samples.
# If the simulation has not finished, this is the time that is waited periodically until all the simulation ended and gave a result
time_to_wait_for_iteration_results         = 20.   
        
############# Paths for the relevant files
home_directory                             = os.getcwd()
# path to executable
path_executable                            = os.path.dirname(os.getcwd())+"/smilei"
# path to basic namelist
path_input_namelist                        = os.path.dirname(os.getcwd())+"/namelist_optimization_LWFA.py"
# name of the input namelist
name_input_namelist                        = "namelist_optimization_LWFA.py"
# path to simulation submission script
path_submission_script                     = os.path.dirname(os.getcwd())+"/launch_Smilei_simulation.sh" 
# path to second submission script. If not needed, just use the same file
path_second_submission_script              = path_submission_script

############# Function to optimize

#### Flag used to set if a numpy function or simulation results are optimized: 
#### if True it optimizes (i.e. maximizes) a numpy function defined in test_function
#### Otherwise, it will postprocess simulation results using the function defined in simulation_postprocessing_function
#### In both cases it is absolutely necessary that the function to optimize gives a float, different from inf,-inf and from nan
use_test_function                         = False

#### import the simulation postprocessing function from a library
sys.path.insert(0, os.path.dirname(os.getcwd()))
from postprocessing_functions import *
def my_postprocessing_analysis():
    return get_sqrtCharge_times_median_Energy_over_MAD_Energy_Percent()

simulation_postprocessing_function        = my_postprocessing_analysis
    
############# Define and launch the optimization run
path_discoveri = "/Users/francescomassimo/Codes/Optimization_on_cluster/Discoveri"
sys.path.insert(0, path_discoveri)

# import Discoveri class
from discoveriMain import createOptimizationRun
if __name__ == '__main__':
    
    starting_directory = os.getcwd()
    # initialize an optimization run 
    optimization_run   = createOptimizationRun(starting_directory                         = starting_directory,                         \
                                               home_directory                             = home_directory,                             \
                                               command_to_launch_jobs                     = command_to_launch_jobs,                     \
                                               name_log_file_simulations                  = name_log_file_simulations,                  \
                                               word_marking_end_of_simulation_in_log_file = word_marking_end_of_simulation_in_log_file, \
                                               path_executable                            = path_executable,                            \
                                               path_input_namelist                        = path_input_namelist,                        \
                                               name_input_namelist                        = name_input_namelist,                        \
                                               path_submission_script                     = path_submission_script,                     \
                                               path_second_submission_script              = path_second_submission_script,              \
                                               optimization_method                        = optimization_method,                        \
                                               number_of_samples_per_iteration            = number_of_samples_per_iteration,            \
                                               number_of_dimensions                       = number_of_dimensions,                       \
                                               search_interval                            = search_interval,                            \
                                               number_of_iterations                       = number_of_iterations,                       \
                                               time_to_wait_for_iteration_results         = time_to_wait_for_iteration_results,         \
                                               input_parameters_names                     = input_parameters_names,                     \
                                               use_test_function                          = use_test_function,                          \
                                               test_function                              = None,                                       \
                                               simulation_postprocessing_function         = simulation_postprocessing_function,         \
                                               iterations_between_outputs                 = iterations_between_outputs,                 \
                                               nu = 1.5, xi=0.,length_scale = np.array([1.e-1,1.e-1]) )
    # execute optimization run
    optimization_run.execute()

    # Save the optimizer predictive model to a file
    with open('model.pkl', 'wb') as file:
        pickle.dump(optimization_run.optimizer.model, file)