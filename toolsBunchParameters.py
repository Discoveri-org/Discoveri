import math
import numpy as np
import scipy.constants

c                       = scipy.constants.c              # lightspeed in vacuum,  m/s
epsilon0                = scipy.constants.epsilon_0      # vacuum permittivity, Farad/m
me                      = scipy.constants.m_e            # electron mass, kg
q                       = scipy.constants.e              # electron charge, C

########## Laser-plasma Params
lambda0                 = 0.8e-6                         # laser central wavelength, m
length_conversion_factor= lambda0/2./math.pi*1.e6        # from c/omega0 to um, corresponds to laser wavelength 0.8 um
nc                      = epsilon0*me/q/q*(2.*math.pi/lambda0*c)**2 #critical density in m^-3 for lambda0
charge_conversion_factor= q*nc * (length_conversion_factor*1e-6)**3 * 10**(12) # from normalized units to pC

########## Auxiliary functions
def weighted_std(values, weights):
    """
    Return the weighted standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return math.sqrt(variance)

def weighted_covariance(values1, values2, weights):
    """
    Return the weighted covariance

    values1, values2 weights -- Numpy ndarrays with the same shape.
    """
    average1 = np.average(values1, weights=weights)
    average2 = np.average(values2, weights=weights)
    # Fast and numerically precise:
    covariance = np.average((values1-average1)*(values2-average2), weights=weights)
    return covariance

def normalized_emittance(transv_coordinate,transv_momentum,weights):
    sigma_transv          = weighted_std(transv_coordinate, weights)
    sigma_p_transv        = weighted_std(transv_momentum, weights)
    sigma_transv_p_transv = weighted_covariance(transv_coordinate,transv_momentum,weights)
    norm_emittance        = (sigma_transv**2)*(sigma_p_transv**2)-sigma_transv_p_transv**2
    return math.sqrt(norm_emittance)
    
def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median
