##### Author      : Francesco Massimo (Laboratoire de Physique des Gaz et des Plasmas, CNRS)
##### Purpose     : utilities (minor or essential) for the optimization process


import os,sys
import time
from datetime import datetime
import numpy as np


def printLogo():
    print("                                                                                  ")       
    print("           _____    _                                              _              ")  
    print("          |  __ \  (_)                                            (_)             ")  
    print("          | |  | |  _   ___    ___    ___   __   __   ___   _ __   _              ")  
    print("          | |  | | | | / __|  / __|  / _ \  \ \ / /  / _ \ | '__| | |             ")  
    print("          | |__| | | | \__ \ | (__  | (_) |  \ V /  |  __/ | |    | |             ")  
    print("          |_____/  |_| |___/  \___|  \___/    \_/    \___| |_|    |_|             ")
    print("                                                                                  ")  
    print("           Data-driven Investigation through Simulations on Clusters              ")
    print("  for the Optimization of the physical Variables' Effects in Regimes of Interest  ")
    print("                                                                                  ")
    print("                                                                                  ")                                                                 
    
    
def printCheckingSamplesAtDatetime():
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("\n ... Checking samples, date and time =", dt_string)
        

def getOptimumPositionAndFunctionValueAfterOptimization(optimizer):    
    optimum_function_value       = np.amax(optimizer.history_samples_positions_and_function_values[:,:,optimizer.number_of_dimensions])
    index_optimum_function_value = np.argwhere(optimizer.history_samples_positions_and_function_values[:,:,optimizer.number_of_dimensions] == optimum_function_value)[0]
    optimum_position             = optimizer.history_samples_positions_and_function_values[index_optimum_function_value[0],index_optimum_function_value[1],0:optimizer.number_of_dimensions]
    best_configuration_number    = index_optimum_function_value[0]*optimizer.number_of_samples_per_iteration+index_optimum_function_value[1]
    return optimum_function_value,optimum_position,best_configuration_number


#### function to compute the evolutionary state in Adaptive Particle Swarm Optimization
def evolutionary_state(f):
    if 0. <= f < 0.25:
        return 3
    elif 0.25 <= f < 0.5:
        return 2
    elif 0.5 <= f < 0.75:
        return 1
    elif 0.75 <= f <= 1.0:
        return 4
    else:
        raise ValueError("f must be in the interval [0, 1]") 
        
#### end membership functions for Adaptive Particle Swarm Optimization

def normalized_euclidean_distance(position1,position2,search_interval_size):
    num_dimensions             = np.size(position1)
    array_search_interval_size = np.asarray(search_interval_size)
    return np.sqrt( np.sum( np.square( np.divide( (position1 - position2), array_search_interval_size) ) ) )
                  
                  
                  

                  
                  



                                        