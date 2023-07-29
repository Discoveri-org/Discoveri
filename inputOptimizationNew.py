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
optimization_method                     = "PSO-TPME" #"Bayesian Optimization"

#### Parameter space to explore
num_dimensions                          = 2 #4 #1 #2
search_interval                         = [[0.,1.],[0.,1.]]#[[13.,25.],[1.,100.],[1.7,2.],[1.3e18,1.6e18]]#[[0.,1.],[0.,1.]]#[[-20.,20.],[-20.,20.]] ##[[16.,24.]] #[[0.,5.],[0.,5.]]
input_parameters_names                  = ["dim1","dim2"] #["delay_behind_laser_micron","bunch_charge_pC","a0","plasma_plateau_density_1_ov_cm3"]

num_samples                             = 6 #6 #1 # 10 #3 #3 # for a Particle Swarm Optimization, this corresponds to the number of particles in the swarm

#### Optimization parameters

max_iterations                          = 30 #100 
iterations_between_outputs              = 1


#### Flag used to set if a numpy function or simulation results are optimized: 
#### if True it optimizes (i.e. maximizes) a numpy function defined in test_function
#### Otherwise, it will postprocess simulation results using the function defined in simulation_postprocessing_function
#### In both cases it is absolutely necessary that the function to optimize gives a number, different from inf,-inf and from nan
#### It is also suggested, especially for Bayesian Optimization, to reduce the orders of magnitude spanned by the function, e.g. with a logarithm
use_test_function                       = True

test_function                           = None
simulation_postprocessing_function      = None

# def test_function(position):
#     return -np.sum(np.square(position-3.))+9. #-rosen(position)
def test_function(position):
    noise = 0.
    return np.sum(   np.multiply(  np.square(position), np.power(np.sin(5 * np.pi * position),6.0)  )  ) + noise
    #return -np.log(rosen(position))
    ##return -np.sum(np.square(position-0.3))+9.
    #return np.sum(   np.multiply(  np.square(position), np.power(np.sin(5 * np.pi * position),6.0)  )  ) + noise

def my_postprocessing_analysis():
    return np.maximum(0,np.log(get_sqrtCharge_times_median_Energy_over_MAD_Energy()))


if (use_test_function==True):
    test_function                           = test_function
else:
    simulation_postprocessing_function      = my_postprocessing_analysis


###########################################################################
####################   Job Managing on Cluster     ########################
###########################################################################

iterations_between_outputs                 = 1

####### Commands and parameters to launch and check jobs
# command to submit job to the system (e.g. SLURM, Condor)
command_to_launch_jobs                     = "condor_submit submit_smilei_job.sub"
# name of the file where the word marking the end of the simulation can be found
name_log_file_simulations                  = "smilei.log"
# if this word appears in name_log_file_simulations, the simulation can be postprocessed
word_marking_end_of_simulation_in_log_file = "END"

# time in seconds to wait between the creation of the sample directories (and in case launching the simulations)
# and the analysis yielding the value of the function for the samples
# if the simulation has not finished, this is the time that is waited periodically until all the simulation ended and gave a result
time_to_wait_for_iteration_results         = 1.   
        
####### Paths
home_directory                             = "/home/INTERNE.LPGP.U-PSUD.FR/f.massimo/test_Particle_Swarm_Optimization/test_2D_optimization" #"/home/INTERNE.LPGP.U-PSUD.FR/f.massimo/test_Particle_Swarm_Optimization/test_2D_optimization" #"/home/INTERNE.LPGP.U-PSUD.FR/f.massimo/test_Particle_Swarm_Optimization" #"/home/INTERNE.LPGP.U-PSUD.FR/f.massimo"
# path to executable
path_executable                            = home_directory+"/smilei/smilei"
# path to basic namelist
path_input_namelist                        = '/home/INTERNE.LPGP.U-PSUD.FR/f.massimo/test_Particle_Swarm_Optimization/test_2D_optimization/test_4_parameters/Input_Namelist.py' #home_directory+"/test_4_parameters/Input_Namelist.py"I
# name of the input namelist
name_input_namelist                        = "Input_Namelist.py"
# path to submission script
path_submission_script                     = home_directory+"/submit_smilei_job.sub" #home_directory+"/test_Particle_Swarm_Optimization/submit_smilei_job.sub"
# path to second submission script
path_second_submission_script              = home_directory+"/smilei_job.sh" #home_directory+"/test_Particle_Swarm_Optimization/smilei_job.sh"



#### only the parameters for the chosen optimizer will be used
#### the others will be ignored
#### the order of the elements in the list optimizer_hyperparameters is important

if (optimization_method=="Random Search"):
    use_Halton_sequence       = True
    
    optimizer_hyperparameters = [use_Halton_sequence]
    
elif (optimization_method=="Bayesian Optimization"):
    
    optimizer_hyperparameters = []
    
elif (optimization_method=="Particle Swarm Optimization"):
    # Parameters specific to Particle Swarm Optimization
    # "classic Particle Swarm Optimization", but using an inertia term w to avoid velocity divergence
    c1                                         = 0.1 # cognitive parameter, must be <1
    c2                                         = 0.1 # social parameter, must be <1
    w                                          = 0.5 # 0.8 #0.8 # inertia
    initial_velocity_over_search_space_size    = 0.1#0.3 #0.03 # parameter limiting the initial velocity of particles, must be <1
    
    optimizer_hyperparameters                  = [c1,c2,w,initial_velocity_over_search_space_size]
    
elif (optimization_method=="IAPSO"): 
    # Parameters specific to Particle Swarm Optimization version called IAPSO
    # (Wanli Yang et al 2021 J. Phys.: Conf. Ser. 1754 012195, doi:10.1088/1742-6596/1754/1/012195)
    # (with some modifications listed in the class definition)
    w1                                         = 0.9 # initial inertia weight parameter, must be < 1
    w2                                         = 0.4 # final inertia weight parameter, must be < w1; this value would be reached after an infinite amount of iterations
    m                                          = 10  # inertia weight decay parameter, the higher it is, the faster the inertia weight will decrease
    initial_velocity_over_search_space_size    = 0.1 #0.3 #0.03 # parameter limiting the initial velocity of particles, must be <1
    
    optimizer_hyperparameters                  = [w1,w2,m,initial_velocity_over_search_space_size]
    
elif (optimization_method=="PSO-TPME"): 
    # Parameters specific to Particle Swarm Optimization version called PSO-TPE
    # (T. Shaquarin, B. R. Noack, International Journal of Computational Intelligence Systems (2023) 16:6, https://doi.org/10.1007/s44196-023-00183-z)
    # (some improvements detailed in the class definition have been made)
    # maximum speed for a particle, it must be a vector with num_dimensions elements
    initial_velocity_over_search_space_size    = 0.03
    w1                                         = 0.9 # initial inertia weight parameter, must be < 1
    w2                                         = 0.1 # final inertia weight parameter, must be < w1; this value would be reached after an infinite amount of iterations
    c1                                         = 1.5 # cognitive parameter, must be <1
    c2                                         = 1.5 # social parameter, must be <1
    Nmax_iterations_bad_particles                          = 2 #maximum number of iterations in which "bad" particles are allowed to explore (Ne in the original paper)
    portion_of_mean_classification_levels      = 0.02 # portion of the mean to define the classification levels mean*(1-p), mean*(1+p)
    # "bad" particles that remain "bad" for more than Nmax_iterations_bad_particlesiterations
    # are relocated around the best swarm particle, within an interval (1-a) and (1+a) in all dimensions
    # in this version the coefficient will decrease linearly
    amplitude_mutated_range_1                  = 0.4
    amplitude_mutated_range_2                  = 0.01
    
    optimizer_hyperparameters                  = [c1,c2,w1,w2,                                      \
                                                 initial_velocity_over_search_space_size,           \
                                                 Nmax_iterations_bad_particles,                                  \
                                                 portion_of_mean_classification_levels,             \
                                                 amplitude_mutated_range_1,amplitude_mutated_range_2]



###########################################################################
##################### Parameters for Random Search ########################
###########################################################################

#### parameters used only if Random Search is used, otherwise they are ignored
# if True, a scrambled Halton sequence will generate the random samples

starting_directory = ""

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
                                               num_samples                                = num_samples,                                \
                                               num_dimensions                             = num_dimensions,                             \
                                               search_interval                            = search_interval,                            \
                                               max_iterations                             = max_iterations,                             \
                                               optimizer_hyperparameters                  = optimizer_hyperparameters,                  \
                                               time_to_wait_for_iteration_results         = time_to_wait_for_iteration_results,         \
                                               input_parameters_names                     = input_parameters_names,                     \
                                               use_test_function                          = use_test_function,                          \
                                               test_function                              = test_function,                              \
                                               simulation_postprocessing_function         = simulation_postprocessing_function,         \
                                               iterations_between_outputs                 = iterations_between_outputs                  )
    # execute optimization run
    optimization_run.execute()