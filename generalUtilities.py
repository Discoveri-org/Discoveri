##### Author      : Francesco Massimo (Laboratoire de Physique des Gaz et des Plasmas, CNRS)
##### Purpose     : utilities (minor or essential) for the optimization process


import os,sys
import time
from datetime import datetime
import numpy as np

# these imports are necessary for the Bayesian Optimization
import warnings
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel


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


# function to compute the evolutionary state in Adaptive Particle Swarm Optimization
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

# euclidean distance, where the coordinates are normalized by the size of the search interval
def normalized_euclidean_distance(position1,position2,search_interval_size):
    num_dimensions             = np.size(position1)
    array_search_interval_size = np.asarray(search_interval_size)
    return np.sqrt( np.sum( np.square( np.divide( (position1 - position2), array_search_interval_size) ) ) )


# Compute the result of a prediction model (surrogate model)
def predictFunctionValueWithSurrogateModel(model, input_position_array):
    # Catch any warning generated when making a prediction
    with warnings.catch_warnings():
        # Ignore generated warnings
        warnings.filterwarnings("ignore")
        return model.predict(input_position_array, return_std=True)

# Expected improvement acquisition function
def getExpectedImprovementAcquisitionFunctionResult(model, Xtest, X, xi):

    # Vectorized version
    # Calculate the best surrogate score found so far
    yhat, _ = predictFunctionValueWithSurrogateModel(model, X)
    best = np.max(yhat)

    # Calculate mean and standard deviation via surrogate model
    mu, std = predictFunctionValueWithSurrogateModel(model, Xtest)

    # Avoid division by zero in the computation of z
    epsilon = 1e-9

    z = (mu - best + xi) / (std + epsilon)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)

    ei = (mu - best + xi) * cdf_z + std * pdf_z
        
    return ei
    
def initializePredictiveModelForOptimization(number_of_samples_to_choose_per_iteration=1,number_of_dimensions=1,**kwargs):
    
    ### Set some parameters for the kernel
    
    # number of points to test through the surrogate function when choosing new samples to draw
    default_number_of_tests           = number_of_dimensions*2000
    number_of_tests                   = kwargs.get('num_tests', default_number_of_tests)
    
    # nu parameter of the Matern Kernel if different from [0.5, 1.5, 2.5, np.inf] 
    # the optimizer may incur a considerably higher computational cost (appr. 10 times higher)
    default_value_nu                  = 1.5
    nu                               = kwargs.get('nu', default_value_nu)
    
    # The kernel scales are normalized, so the input data to fit and predict must be normalized
    # This length_scale parameter is really important for the regressor.
    
    # See Example 1.7.2.1 of https://scikit-learn.org/stable/modules/gaussian_process.html#gp-kernels
    
    # If it is a float, it is applied to all dimensions.
    # If it is a list of length number_of_dimensions, it will contain the length scale for each dimension
    
    # smaller length_scales mean smaller scales of variations  --> risk of overfitted model
    # larger length_scales mean larger scales of variation     --> risk of biased model
    
    # Thus a good equilibrium should be found ideally
    
    # If the variations of the function to optimize with respect to a certain dimension within the search interval are large,
    # the length_scale associated to that dimension will be smaller.
    # if the function to optimize does not vary much in one dimension, the associated length scale of that dimension can be larger.
    default_length_scale             = 1.0  
    length_scale                     = kwargs.get('length_scale', default_length_scale)
    
    
    # the length_scale will be optimizer with optimizer="fmin_l_bfgs_b" (see below)
    # the following parameter fixes the bounds for this optimization
    default_length_scale_bounds      = (1e-5, 1e5)
    length_scale_bounds              = kwargs.get('length_scale_bounds', default_length_scale_bounds)
    
    # parameter used in the acquisition function (expected improvement)
    # to tune the balance between exploitation of good points already found and exploration of the parameter space
    # When equal to 0, exploitation is privileged. 
    # High values privilege exploration.
    default_xi                       = 0. # default privileges exploitation
    xi                               = kwargs.get('xi', default_xi )
    
    
    # now the kernel and the model 
    kernel                           = ConstantKernel(1.0, constant_value_bounds="fixed")*Matern(nu=nu,length_scale=length_scale,length_scale_bounds=length_scale_bounds) 
    model                            = GaussianProcessRegressor(optimizer="fmin_l_bfgs_b",kernel=kernel)
    XsamplesChosenWithSurrogateModel = np.zeros(shape=(number_of_samples_to_choose_per_iteration, number_of_dimensions))
    
    
    return number_of_tests,nu,length_scale,length_scale_bounds,xi,kernel,model,XsamplesChosenWithSurrogateModel

                  
                  
                  

                  
                  



                                        