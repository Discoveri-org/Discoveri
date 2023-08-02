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


#### membership functions for Adaptive Particle Swarm Optimization
def mu_S1(f):
    if f <= 0.4:
        return 0
    elif 0.4 < f <= 0.6:
        return 5 * f - 2
    elif 0.6 < f <= 0.7:
        return 1
    elif 0.7 < f <= 0.8:
        return -10 * f + 8
    elif 0.8 < f:
        return 0

def mu_S2(f):
    if f <= 0.2:
        return 0
    elif 0.2 < f <= 0.3:
        return 10 * f - 2
    elif 0.3 < f <= 0.4:
        return 1
    elif 0.4 < f <= 0.6:
        return -5 * f + 3
    elif 0.6 < f:
        return 0

def mu_S3(f):
    if f <= 0.1:
        return 1
    elif 0.1 < f <= 0.3:
        return -5 * f + 1.5
    elif 0.3 < f:
        return 0

def mu_S4(f):
    if f <= 0.7:
        return 0
    elif 0.7 < f <= 0.9:
        return 5 * f - 3.5
    elif 0.9 < f:
        return 1  
        
#### end membership functions for Adaptive Particle Swarm Optimization

def normalized_euclidean_distance(position1,position2,search_interval_size):
    num_dimensions             = np.size(position1)
    array_search_interval_size = np.asarray(search_interval_size)
    return np.sqrt( np.sum( np.square( np.divide( (position1 - position2), array_search_interval_size) ) ) )
                  
                  
                  

                  
                  



                                        