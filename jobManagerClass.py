##### Author      : Francesco Massimo (Laboratoire de Physique des Gaz et des Plasmas, CNRS)
##### Purpose     : definition of the class that manages the jobs
##### Last update : 29/07/2023

import os,sys,time

from optimizerClasses import *
from toolsSmileiAnalysis import *
from generalUtilities import *

class jobManager:
    def __init__(self, \
                 starting_directory                         = "",           \
                 home_directory                             = "",           \
                 command_to_launch_jobs                     = "",           \
                 name_log_file_simulations                  = "smilei.log", \
                 word_marking_end_of_simulation_in_log_file = "END",        \
                 path_executable                            = "smilei",     \
                 path_input_namelist                        = "",           \
                 name_input_namelist                        = "",           \
                 path_submission_script                     = "",           \
                 path_second_submission_script              = "",           \
                 time_to_wait_for_iteration_results         = 1.,           \
                 input_parameters_names                     = [],           \
                 use_test_function                          = True,         \
                 test_function                              = None,         \
                 simulation_postprocessing_function         = None,         \
                 iterations_between_outputs                 = 1             ):
                 
                 
                 self.home_directory                             = home_directory
                 self.starting_directory                         = starting_directory
                 self.command_to_launch_jobs                     = command_to_launch_jobs
                 self.name_log_file_simulations                  = name_log_file_simulations
                 self.word_marking_end_of_simulation_in_log_file = word_marking_end_of_simulation_in_log_file
                 self.path_executable                            = path_executable
                 self.path_input_namelist                        = path_input_namelist
                 self.name_input_namelist                        = name_input_namelist
                 self.path_submission_script                     = path_submission_script
                 self.path_second_submission_script              = path_second_submission_script
                 self.time_to_wait_for_iteration_results         = time_to_wait_for_iteration_results
                 self.input_parameter_names                      = input_parameters_names
                 self.use_test_function                          = use_test_function
                 self.test_function                              = test_function
                 self.simulation_postprocessing_function         = simulation_postprocessing_function
                 self.iterations_between_outputs                 = iterations_between_outputs
                 
    def prepareOneSimulationDirectory( self, config_id, input_parameters ):
        # create and fill directory with simulation
        directory = self.starting_directory+"/Config_id_" +str(config_id).zfill(7)
        os.makedirs(directory)
        if (self.use_test_function==False):
            os.chdir(directory)
            #os.system("cp "+path_executable+" .")
            os.system("cp "+self.path_submission_script+" .")
            os.system("cp "+self.path_second_submission_script+" .")
            os.system("cp "+self.path_input_namelist+" .")
        
            # prepare namelist writing the parameter configuration 
            line_to_write_in_namelist = self.generateConfigurationToWriteOnNamelist(input_parameters)
            writeConfigurationDictionaryInNamelist(line_to_write_in_namelist,self.name_input_namelist,self.path_input_namelist)
        else:
            line_to_write_in_namelist = ""
        
        return directory,line_to_write_in_namelist
        
    def launchSimulation( self, config_id, input_parameters):
        
        print("\n - Preparing directory for Config_id="+str(config_id).zfill(7))
        # prepare the simulation directory
        new_simulation_directory, configuration_parameters = self.prepareOneSimulationDirectory( config_id,input_parameters )
        
        if (self.use_test_function==False):
            print("\n - Launching simulation for Config_id="+str(config_id).zfill(7))
            # launch the simulation
            os.system(self.command_to_launch_jobs)
            # go back to the starting directory
            os.chdir(self.starting_directory)
        
        return new_simulation_directory, configuration_parameters
        
    def generateConfigurationToWriteOnNamelist(self,input_parameters):
        # generate the line to insert in the input namelist to select the configuration to explore
        write_to_namelist = "external_config = { "
        for i_parameter in range(0,len(self.input_parameters_names)):
            write_to_namelist = write_to_namelist + "'" + self.input_parameters_names[i_parameter]+  "':"
            write_to_namelist = write_to_namelist + str(input_parameters[i_parameter]) 
            if (i_parameter == len(self.input_parameters_names)-1):
                write_to_namelist = write_to_namelist + " }"
            else:
                write_to_namelist = write_to_namelist + ", "
        return write_to_namelist
        
    def writeConfigurationDictionaryInNamelist(self,line_to_write_in_namelist):
        # create an input namelist file identical to the one in the path_input_namelist
        # the only difference will be a line with the parameters of the configuration to explore
        # this line will be written after the line containng "External_config"
        # after this line, a dictionary with the sample parameters will be written
        # you need to ensure that the input file of your namelist uses this dictionary
        with open(self.path_input_namelist, "r") as file:
    	       namelist_file_content = file.readlines()
        # insert in the original namelist file dictionary with parameters after the line "External_config" 
        with open(self.name_input_namelist, 'w') as namelist:
            for line in namelist_file_content:
                newline = line
                if "External_config" in line:
                    #print(line)
                    newline = newline + "\n" + line_to_write_in_namelist + "\n"
                namelist.write(newline)

    def checkAndAnalyseSimulations(self, optimizer, list_configurations, iteration):
        
        num_samples = optimizer.num_samples
        # create list of sample to keep track which ones are finished
        # this because we do not know in which order the samples will finish thei simulation
        samples_running_a_simulation = [i for i in range(0,optimizer.num_samples)] 
        
        ## while while running_simulations_ > 0: 
        while len(list_configurations)>0:
            # periodically sleep and then check if the simulations finished
            time.sleep(self.time_to_wait_for_iteration_results);printCheckingSamplesAtDatetime()
            for configuration in list_configurations:
                os.chdir(configuration.split(',', 1)[0]) # go inside the considered directory
                index_configuration = list_configurations.index(configuration)
                isample             = samples_running_a_simulation[index_configuration]
                
                if (self.use_test_function==False): # check the results of the simulation
                    with open(self.name_log_file_simulations, 'r') as simulation_log_file:
                        for line in simulation_log_file:
                            if self.word_marking_end_of_simulation_in_log_file in line: # if a simulation has ended
                                # remove the directory and the sample from the list of those still running a simulation
                                list_configurations.remove(list_configurations[index_configuration])
                                samples_running_a_simulation.remove(isample)
                                # analyse the result, i.e. compute function value at new sample position
                                function_value = self.simulation_postprocessing_function() #np.maximum(np.log(get_sqrtCharge_times_median_Energy_over_MAD_Energy()),0.)#get_average_bunch_energy()
                                # update history array, and if function_value is better, update sample's best function_value and positon
                                optimizer.updateHistoryAndCheckIfFunctionValueIsBetter(iteration,isample,function_value)
                                # you can stop reading the log file of the simulation
                                break
                else: # just use a test function, instead of running a simulation
                    # remove the directory and the sample from the list of those still running a simulation
                    list_configurations.remove(list_configurations[index_configuration])
                    samples_running_a_simulation.remove(isample)
                    # analyse the result, i.e. compute function value at new sample position
                    function_value = self.test_function(optimizer.samples[isample].position)
                    # update history array, and if function_value is better, update sample's best function_value and positon
                    optimizer.updateHistoryAndCheckIfFunctionValueIsBetter(iteration,isample,function_value)
                    
                # return to starting folder
                os.chdir(self.starting_directory)
        
        #### here running_simulations = 0
                 
        # update the optimizer optimum, which is the optimum among the optima of the individual samples
        optimizer.updateOptimumFunctionValueAndPosition()
        
        # perform some special operations for the data structure needed by the Bayesian Optimization kernel
        if ( optimizer.name == "Bayesian Optimization" ):
            if iteration == 0:
                for isample in range(0,optimizer.num_samples):
                    optimizer.y[isample] = optimizer.history_samples_positions_and_function_values[iteration,isample,optimizer.num_dimensions]
                optimizer.model.fit(optimizer.X, optimizer.y)
            else:
                for isample in range(0,optimizer.num_samples):
                    optimizer.X = np.vstack((optimizer.X, optimizer.Xsamples[isample]))
                    optimizer.y = np.vstack((optimizer.y, optimizer.history_samples_positions_and_function_values[iteration,isample,optimizer.num_dimensions]))
                optimizer.model.fit(optimizer.X, optimizer.y)
        if (optimizer.name == "PSO-TPME"):
            
            average_function_value_of_swarm = np.average(optimizer.history_samples_positions_and_function_values[iteration,:,optimizer.num_dimensions])
            average_function_value_of_swarm = np.maximum(average_function_value_of_swarm,optimizer.best_average_function_value)
            if (average_function_value_of_swarm > optimizer.best_average_function_value):
                optimizer.best_average_function_value = average_function_value_of_swarm
            bad_function_value_level        = average_function_value_of_swarm*(1.-optimizer.portion_of_mean_classification_levels)
            good_function_value_level       = average_function_value_of_swarm*(1.+optimizer.portion_of_mean_classification_levels)
            
            print("\n PSO-TPME: Bad, average and good function value levels :",bad_function_value_level,average_function_value_of_swarm,good_function_value_level)
            old_particle_category       = optimizer.particle_category
        
            for isample in range(0,optimizer.num_samples):
                
                if optimizer.history_samples_positions_and_function_values[iteration,isample,optimizer.num_dimensions] > good_function_value_level:
                    optimizer.particle_category[isample] = "good"
                    optimizer.particles_number_of_iterations_remained_in_current_category[isample] = 1
                elif optimizer.history_samples_positions_and_function_values[iteration,isample,optimizer.num_dimensions]< bad_function_value_level:
                    if (optimizer.particles_number_of_iterations_remained_in_current_category[isample]>optimizer.Nmax_exploration-1):
                        optimizer.particle_category[isample] = "hopeless"
                        optimizer.particles_number_of_iterations_remained_in_current_category[isample] = 1
                    else:
                        if (iteration==0):
                            optimizer.particle_category[isample] = "bad" 
                            optimizer.particles_number_of_iterations_remained_in_current_category[isample] = 1
                        else:
                            if (old_particle_category[isample]=="bad"):
                                optimizer.particle_category[isample] = "bad" 
                                optimizer.particles_number_of_iterations_remained_in_current_category[isample] = optimizer.particles_number_of_iterations_remained_in_current_category[isample]+1
                            else:
                                optimizer.particle_category[isample] = "bad" 
                                optimizer.particles_number_of_iterations_remained_in_current_category[isample] = 1
                else:
                    optimizer.particle_category[isample] = "fair"
                    optimizer.particles_number_of_iterations_remained_in_current_category[isample] = 1

                        
                print("Particle ",isample,"with function value ","{:.3f}".format(optimizer.history_samples_positions_and_function_values[iteration,isample,optimizer.num_dimensions])," is classified as ",optimizer.particle_category[isample], \
                        ", remained in this category for",optimizer.particles_number_of_iterations_remained_in_current_category[isample]," iterations")
                    
                 
                 
                 
                 