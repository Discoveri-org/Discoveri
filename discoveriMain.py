########################################################################################################################
########################################################################################################################
####                                                                                                                ####
####                                                                                                                ####
####                                             DISCOVERI                                                          ####
####                      Data-driven Investigation through Simulations on Clusters                                 ####
####             for the Optimization of the physical Variables' Effects in Regimes of Interest                     ####
####                                                                                                                ####
####                                   Cooperative OpenSource Project                                               ####
####                                          started May 2023                                                      ####
####                                                                                                                ####
####                                                                                                                ####
########################################################################################################################
########################################################################################################################

##### Author      : Francesco Massimo (Laboratoire de Physique des Gaz et des Plasmas, CNRS)
##### Purpose     : perform an optimizing parameter space exploration on a cluster using PIC simulations (or test functions I guess)
#####               available optimization algorithms: Random Search, Particle Swarm Optimization ("classic", IAPSO, PSO-TPME), Bayesian Optimization 


# import "standard" libraries
import random
import numpy as np
import os,sys
import time
from scipy.optimize import rosen
# import libraries from this software
from optimizerClasses import *
from jobManagerClass import *
from generalUtilities import *



class optimizationRun:
    def __init__(self, starting_directory=None, home_directory="", command_to_launch_jobs="",
                 name_log_file_simulations="", word_marking_end_of_simulation_in_log_file="",
                 path_executable="", path_input_namelist="", name_input_namelist="",
                 path_submission_script="", path_second_submission_script="",
                 optimization_method="", number_of_samples_per_iteration=1, number_of_dimensions=1, search_interval=[[0., 1.]],
                 number_of_iterations=1,time_to_wait_for_iteration_results = 1.,
                 input_parameters_names=[],use_test_function=True,test_function=None,simulation_postprocessing_function=None,iterations_between_outputs=1,**kwargs):
                 
        printLogo()
        
        if starting_directory is None:
            self.starting_directory = os.getcwd()
        else:
            self.starting_directory = starting_directory

        # initialize configuration number
        self.config_id                   = 0
        self.starting_time               = 0.
        self.iterations_between_outputs  = iterations_between_outputs
        
        self.running_jobs                = []
        self.list_configurations         = []   
        self.time_to_complete_iterations = []
        
        
        self.optimization_method         = optimization_method
        
        # Initialize classes
        self.optimizer     = None
        self.job_manager   = None


        # Initialize job manager object
        print("\n Initializing the job manager")
        self.job_manager = jobManager( starting_directory = self.starting_directory, home_directory = home_directory, \
                                  command_to_launch_jobs = command_to_launch_jobs, \
                                  name_log_file_simulations = name_log_file_simulations, \
                                  word_marking_end_of_simulation_in_log_file = word_marking_end_of_simulation_in_log_file, \
                                  path_executable = path_executable, \
                                  path_input_namelist = path_input_namelist, \
                                  name_input_namelist = name_input_namelist, \
                                  path_submission_script = path_submission_script, \
                                  path_second_submission_script = path_second_submission_script,\
                                  time_to_wait_for_iteration_results = time_to_wait_for_iteration_results,\
                                  input_parameters_names=input_parameters_names,\
                                  use_test_function=use_test_function,\
                                  test_function=test_function,\
                                  simulation_postprocessing_function=simulation_postprocessing_function,\
                                  iterations_between_outputs=iterations_between_outputs )
        
        print("\n Initializing the optimizer\n\n")

        if (self.optimization_method   == "Random Search"):
            # initialize a Random Search
            self.optimizer             = RandomSearch         (name=optimization_method,                                      \
                                                              number_of_samples_per_iteration=number_of_samples_per_iteration, number_of_dimensions=number_of_dimensions,         \
                                                              search_interval=search_interval, number_of_iterations=number_of_iterations, \
                                                              **kwargs )
        elif (self.optimization_method == "Grid Search"):
            # initialize a Grid Search
            self.optimizer             = GridSearch           (name=optimization_method,                                      \
                                                              number_of_samples_per_iteration=number_of_samples_per_iteration, number_of_dimensions=number_of_dimensions,         \
                                                              search_interval=search_interval, number_of_iterations=number_of_iterations, \
                                                              **kwargs )
                                                              
        elif (self.optimization_method == "Bayesian Optimization"):
            # initialize a Bayesian Optimizer
            self.optimizer             = BayesianOptimization (name=optimization_method,                                      \
                                                              number_of_samples_per_iteration=number_of_samples_per_iteration, number_of_dimensions=number_of_dimensions,         \
                                                              search_interval=search_interval, number_of_iterations=number_of_iterations, \
                                                              **kwargs )
        elif (self.optimization_method == "Particle Swarm Optimization"):
            # initialize a swarm of particles, "classic" version of the Particle Swarm Optimization
            self.optimizer             = ParticleSwarmOptimization(name=optimization_method,                                  \
                                                              number_of_samples_per_iteration=number_of_samples_per_iteration, number_of_dimensions=number_of_dimensions,         \
                                                              search_interval=search_interval, number_of_iterations=number_of_iterations, \
                                                              **kwargs )
                                                              
                                                              
        elif (self.optimization_method == "Adaptive Particle Swarm Optimization"):
            # initialize a swarm of particles, version of the Particle Swarm Optimization
            # where the hyperparameters of the swarm are automatically tuned during the optimization
            self.optimizer             = ParticleSwarmOptimization(name=optimization_method,                                  \
                                                              number_of_samples_per_iteration=number_of_samples_per_iteration, number_of_dimensions=number_of_dimensions,         \
                                                              search_interval=search_interval, number_of_iterations=number_of_iterations, \
                                                              **kwargs )
                                                              
        elif (self.optimization_method == "FST-PSO"):
            # initialize a swarm of particles, version of the Particle Swarm Optimization
            # where the hyperparameters of each particle are automatically tuned during the optimization
            self.optimizer             = ParticleSwarmOptimization(name=optimization_method,                                  \
                                                              number_of_samples_per_iteration=number_of_samples_per_iteration, number_of_dimensions=number_of_dimensions,         \
                                                              search_interval=search_interval, number_of_iterations=number_of_iterations, \
                                                              **kwargs )
                                                              
        # elif (self.optimization_method == "Genetic Algorithm"):
        #     # initialize an optimizer which uses a Genetic Algorithm
        #     self.optimizer             = GeneticAlgorithm     (name=optimization_method, \
        #                                                       number_of_samples_per_iteration=number_of_samples_per_iteration, number_of_dimensions=number_of_dimensions, \
        #                                                       search_interval=search_interval, number_of_iterations=number_of_iterations, \
        #                                                       **kwargs )
            
        else:
            print("Error, the optimization_method must be either 'Grid Search', 'Random Search', 'Bayesian Optimization', 'Particle Swarm Optimization', 'PSO-TPME' or 'Adaptive Particle Swarm Optimization' \n")
            sys.exit()


        
        
    def execute(self):
        
        iteration              = 0

        # create file to store all the simulation launched
        os.system("touch list_configurations.csv")
        
        print("\nSTART OF THE OPTIMIZATION RUN \n\n")

        # execute first iteration with the initial particle positions
        print("\n\n Iteration:", iteration+1,"/",self.optimizer.number_of_iterations,"\n\n\n")

        self.starting_time          = time.time()
        new_simulation_directory    = None
        configuration_parameters    = None
        
        ######### Perform a iteration = 0 to evaluate the function to optimise at the first random points

        for isample in range(0,self.optimizer.number_of_samples_per_iteration):
            input_parameters   = self.optimizer.samples[isample].position
            if (self.job_manager.use_test_function==False): # samples are made of simulations
                # launch the simulation corresponding to isample
                new_simulation_directory, configuration_parameters = self.job_manager.launchSimulation(self.config_id,input_parameters)
                # add simulation to list of jobs , store information for later use          
                self.running_jobs.append(new_simulation_directory); 
                self.list_configurations.append(new_simulation_directory+", "+configuration_parameters+" \n") 
            else: # samples are evaluated with a test function
                configuration_parameters = self.job_manager.generateConfigurationToWriteOnNamelist(input_parameters,self.optimizer.number_of_dimensions)
                self.list_configurations.append("Configuration_id_"+str(self.config_id).zfill(5)+": "+configuration_parameters+" \n")    
            # increase the configuration number
            self.config_id = self.config_id+1      
            
  
    
        # append on file the configurations that were launched in this iteration
        with open('list_configurations.csv', 'w') as f:
            for configuration in self.list_configurations:
                f.write(configuration)


        # Check if the simulations launched in this iteration have finished
        # and analyse all of them until all are finished 
        # then, update the global optimum
        self.job_manager.performFunctionEvaluations(self.optimizer,self.list_configurations,iteration)
        #self.job_manager.checkAndAnalyseSimulations(self.optimizer,self.list_configurations,iteration)

        #Diagnostic
        self.optimizer.printOptimumFunctionValueAndOptimumPosition() 
        if (iteration%self.iterations_between_outputs==0):
            self.optimizer.printSamplesPositions()                 

        ######### Optimization loop, iterations > 1

        # Repeat until the last iteration
        for iteration in range(1,self.optimizer.number_of_iterations):
            print("\n\n\n\n Iteration:", iteration+1,"/",self.optimizer.number_of_iterations)
            time_start_new_iteration = time.time()-self.starting_time
            self.time_to_complete_iterations.append(time_start_new_iteration)
            print("\n Total time lapsed from the start of the optimization = ",time_start_new_iteration," s\n\n\n" )
    
            self.running_jobs = []
            self.list_configurations = []
    
            # generate new samples position to explore, store position history
            self.optimizer.updateSamplesForExploration()
    
            # Launching the simulations - one simulation corresponding to each particle of the swarm
            for isample in range(0,self.optimizer.number_of_samples_per_iteration):
                input_parameters   = self.optimizer.samples[isample].position
                if (self.job_manager.use_test_function==False): # samples are made of simulations
                    # launch the simulation corresponding to isample
                    new_simulation_directory, configuration_parameters = self.job_manager.launchSimulation(self.config_id,input_parameters)
                    # add simulation to list of jobs, store information for later use            
                    self.running_jobs.append(new_simulation_directory)
                    self.list_configurations.append(new_simulation_directory+", "+configuration_parameters+" \n") 
                else: # samples are evaluated with a test function
                    configuration_parameters = self.job_manager.generateConfigurationToWriteOnNamelist(input_parameters,self.optimizer.number_of_dimensions)
                    self.list_configurations.append("Configuration_id_"+str(self.config_id).zfill(5)+": "+configuration_parameters+" \n")       
        
                # increase the configuration number
                self.config_id = self.config_id+1
    
            # append on file the configurations that were launched in this iteration
            with open('list_configurations.csv', 'a') as f:
                for configuration in self.list_configurations:
                    f.write(configuration)
    
            # Check if the simulations launched in this iteration have finished
            # and analyse all of them until all are finished 
            # then, update swarm optimum
            self.job_manager.performFunctionEvaluations(self.optimizer,self.list_configurations,iteration)    
            #self.job_manager.checkAndAnalyseSimulations(self.optimizer,self.list_configurations,iteration)
        
            # Diagnostic
            self.optimizer.printOptimumFunctionValueAndOptimumPosition() 
            if (iteration%self.iterations_between_outputs==0):
                self.optimizer.printSamplesPositions()
                with open('history_particles_positions_and_function_values_iteration_'+str(iteration).zfill(5)+'.npy', 'wb') as f:
                    np.save( f, self.optimizer.history_samples_positions_and_function_values[0:iteration+1,:,:] )
                with open('time_to_complete_iterations_up_to_iteration_'+str(iteration).zfill(5)+'.npy', 'wb') as f:
                    np.save( f,np.asarray(self.time_to_complete_iterations[0:iteration+1]))
                if (self.optimizer.name=="Adaptive Particle Swarm Optimization"):
                    self.optimizer.APSOSavePartialHyperparametersHistory()
                if (self.optimizer.name=="FST-PSO"):
                    self.optimizer.FSTPSOSavePartialHyperparametersHistory()
              
    

        ######### Print all the optimization history

        print("\n\n\n\n End of the optimization\n\n\n")
        print("\n",self.optimizer.number_of_iterations," iterations were completed")
        print("\n Total time for the optimization = ",time.time()-self.starting_time," s\n\n" )
        self.time_to_complete_iterations.append(time.time()-self.starting_time)
        self.time_to_complete_iterations = np.asarray(self.time_to_complete_iterations)
        #print("\n Explored positions and function values during the optimization:")
        #print("iteration | isample | particle position (number_of_dimensions columns) | function value")
        #print(optimizer.history_samples_positions_and_function_values)

        optimum_function_value,optimum_position,best_configuration_number = getOptimumPositionAndFunctionValueAfterOptimization(self.optimizer)
        print("\n Optimum function value = ",optimum_function_value,", Found at position ",optimum_position," in Configuration ",best_configuration_number,"\n\n" )

        ######### Save all the optimization history

        print("\n Saving optimization history and time lapsed")
        with open('history_particles_positions_and_function_values.npy', 'wb') as f:
            np.save( f, self.optimizer.history_samples_positions_and_function_values )
        with open('time_to_complete_iterations.npy', 'wb') as f:
            np.save( f,self.time_to_complete_iterations)
        if (self.optimizer.name=="Adaptive Particle Swarm Optimization"):
            # execute one last evaluation of the evolutionary state and then save history of the hyperparameters
            self.optimizer.APSOLastEvaluationAndDumpHyperparameters()
        if (self.optimizer.name=="FST-PSO"):
            # execute one last adaptation then save history of the hyperparameters
            self.optimizer.FSTPSOLastEvaluationAndDumpHyperparameters()
        print("\n Optimization history and time lapsed saved")
        print("\n\nEND OF THE OPTIMIZATION RUN")
        print("\n\n\n")
        
def createOptimizationRun(starting_directory=None, home_directory="", command_to_launch_jobs="",
                         name_log_file_simulations="", word_marking_end_of_simulation_in_log_file="",
                         path_executable="", path_input_namelist="", name_input_namelist="",
                         path_submission_script="", path_second_submission_script="",
                         optimization_method="", number_of_samples_per_iteration=1, number_of_dimensions=1, search_interval=[[0., 1.]],
                         number_of_iterations=1,time_to_wait_for_iteration_results = 1.,
                         input_parameters_names=[],use_test_function=True,test_function=None,simulation_postprocessing_function=None,iterations_between_outputs=1,**kwargs ):
    return optimizationRun(starting_directory=starting_directory, home_directory=home_directory,
                           command_to_launch_jobs=command_to_launch_jobs,
                           name_log_file_simulations=name_log_file_simulations,
                           word_marking_end_of_simulation_in_log_file=word_marking_end_of_simulation_in_log_file,
                           path_executable=path_executable, path_input_namelist=path_input_namelist,
                           name_input_namelist=name_input_namelist, path_submission_script=path_submission_script,
                           path_second_submission_script=path_second_submission_script,
                           optimization_method=optimization_method, number_of_samples_per_iteration=number_of_samples_per_iteration,
                           number_of_dimensions=number_of_dimensions, search_interval=search_interval,
                           number_of_iterations=number_of_iterations,
                           time_to_wait_for_iteration_results = time_to_wait_for_iteration_results,
                           input_parameters_names=input_parameters_names,use_test_function=use_test_function,
                           test_function=test_function,simulation_postprocessing_function=simulation_postprocessing_function,iterations_between_outputs=iterations_between_outputs,**kwargs)

            



