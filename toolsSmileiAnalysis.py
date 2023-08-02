import happi
import math
import numpy as np
import scipy.constants
from toolsBunchParameters import *


########## Constants
electron_mass_MeV        = scipy.constants.physical_constants["electron mass energy equivalent in MeV"][0]

########## Species name
species_name             = "electronfromion" #"electronbunch"

def get_average_bunch_energy():
    ########## Open a simulation S
    S                        = happi.Open(".", show=False,verbose=False)
    
    ########## Open the DiagTrackParticles
    chunk_size = 2000000
    timestep   = 3000 # timestep where the energy is extracted
    track_part = S.TrackParticles(species = species_name, chunksize=chunk_size, sort=False)

    if (timestep not in track_part.getAvailableTimesteps()):
        print(" ")
        print("Selected timestep not available in the DiagTrackParticles output")
        print("Available timesteps = "+str(track_part.getAvailableTimesteps()))
        print(" ")
        exit()
    
    average_energy = 0.
    # Read the DiagTrackParticles data
    for particle_chunk in track_part.iterParticles(timestep, chunksize=chunk_size):
    
        ### Read particles momentum arrays 
        # momenta
        px             = particle_chunk["px"]
        py             = particle_chunk["py"]
        pz             = particle_chunk["pz"]
        p              = np.sqrt((px**2+py**2+pz**2))                         # Particles Momentum
        # energy
        E              = np.sqrt((1.+p**2))                                   # Particles energy
        # weights
        w              = particle_chunk["w"]
        # compute average energy
        average_energy = electron_mass_MeV*np.average( E, weights = w)
    
    return average_energy

def get_sqrtCharge_times_median_Energy_over_MAD_Energy():
    ########## Open a simulation S
    S                        = happi.Open(".", show=False,verbose=False)
    
    ########## Open the DiagTrackParticles
    chunk_size = 2000000
    timestep   = 13000 # timestep where the energy is extracted
    track_part = S.TrackParticles(species = species_name, chunksize=chunk_size, sort=False)

    if (timestep not in track_part.getAvailableTimesteps()):
        print(" ")
        print("Selected timestep not available in the DiagTrackParticles output")
        print("Available timesteps = "+str(track_part.getAvailableTimesteps()))
        print(" ")
        exit()
    
    average_energy = 0.
    # Read the DiagTrackParticles data
    for particle_chunk in track_part.iterParticles(timestep, chunksize=chunk_size):
    
        ### Read particles momentum arrays 
        # momenta
        px             = particle_chunk["px"]
        py             = particle_chunk["py"]
        pz             = particle_chunk["pz"]
        p              = np.sqrt((px**2+py**2+pz**2))                         # Particles Momentum
        # energy
        E              = np.sqrt((1.+p**2))                                   # Particles energy
        # weights
        w              = particle_chunk["w"]
    median_Energy = electron_mass_MeV*weighted_median( E, weights = w); #print("Median Energy = ",median_Energy," MeV")
    Energy_deviation = electron_mass_MeV*E-median_Energy    #np.average(E,weights=w))
    abs_Energy_deviation = np.abs(Energy_deviation);MAD_Energy = weighted_median(abs_Energy_deviation,weights=w)
    bool_array = ( (E*electron_mass_MeV>(median_Energy-MAD_Energy)) & (E*electron_mass_MeV>(median_Energy-MAD_Energy)) )
    Q = abs(np.sum(w[bool_array])*charge_conversion_factor); #print("Q = ",Q," pC")
    result = math.sqrt(Q)*median_Energy/MAD_Energy
    
    try:
        # compute median energy
        median_Energy = electron_mass_MeV*weighted_median( E, weights = w); #print("Median Energy = ",median_Energy," MeV")
        # compute energy spread as Median absolute deviation of energy
        Energy_deviation = electron_mass_MeV*E-median_Energy    #np.average(E,weights=w))
        abs_Energy_deviation = np.abs(Energy_deviation)
    
        MAD_Energy = weighted_median(abs_Energy_deviation,weights=w); #print("MAD Energy = ",MAD_Energy," MeV")
        bool_array = ( (E*electron_mass_MeV>(median_Energy-MAD_Energy)) & (E*electron_mass_MeV>(median_Energy-MAD_Energy)) )
        # compute charge 
        Q = abs(np.sum(w[bool_array])*charge_conversion_factor); #print("Q = ",Q," pC")
        result = math.sqrt(Q)*median_Energy/MAD_Energy
        #print (Q,MAD_Energy,result,math.isinf(result),math.isinf(np.log(result)) )
        if (math.isinf(result) or math.isnan(result) or (MAD_Energy<1.e-2) or (result<1.e-1) or (Q<1.) or math.isinf(np.log(result))):
            return 0.
        else:
            return result
    except:
        return 0.
        
def get_sqrtCharge_times_median_Energy_over_MAD_Energy_Percent():
    # based on the objective function used in S. Jalas et al, Phys. Rev. Lett. 126, 104801 (2021), https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.104801
    ########## Open a simulation S
    S                        = happi.Open(".", show=False,verbose=False)
    
    ########## Open the DiagTrackParticles
    chunk_size = 2000000
    timestep   = 13000 # timestep where the energy is extracted
    track_part = S.TrackParticles(species = species_name, chunksize=chunk_size, sort=False)

    if (timestep not in track_part.getAvailableTimesteps()):
        print(" ")
        print("Selected timestep not available in the DiagTrackParticles output")
        print("Available timesteps = "+str(track_part.getAvailableTimesteps()))
        print(" ")
        exit()
    
    average_energy = 0.
    # Read the DiagTrackParticles data
    for particle_chunk in track_part.iterParticles(timestep, chunksize=chunk_size):
    
        ### Read particles momentum arrays 
        # momenta
        px             = particle_chunk["px"]
        py             = particle_chunk["py"]
        pz             = particle_chunk["pz"]
        p              = np.sqrt((px**2+py**2+pz**2))                         # Particles Momentum
        # energy
        E              = np.sqrt((1.+p**2))                                   # Particles energy
        # weights
        w              = particle_chunk["w"]
    median_Energy = electron_mass_MeV*weighted_median( E, weights = w); #print("Median Energy = ",median_Energy," MeV")
    Energy_deviation = electron_mass_MeV*E-median_Energy    #np.average(E,weights=w))
    abs_Energy_deviation = np.abs(Energy_deviation);MAD_Energy = weighted_median(abs_Energy_deviation,weights=w)
    bool_array = ( (E*electron_mass_MeV>(median_Energy-MAD_Energy)) & (E*electron_mass_MeV>(median_Energy-MAD_Energy)) )
    Q = abs(np.sum(w[bool_array])*charge_conversion_factor); #print("Q = ",Q," pC")
    result = math.sqrt(Q)*median_Energy/MAD_Energy
    
    try:
        # compute median energy
        median_Energy = electron_mass_MeV*weighted_median( E, weights = w); #print("Median Energy = ",median_Energy," MeV")
        # compute energy spread as Median absolute deviation of energy
        Energy_deviation = electron_mass_MeV*E-median_Energy    #np.average(E,weights=w))
        abs_Energy_deviation = np.abs(Energy_deviation)
    
        MAD_Energy = weighted_median(abs_Energy_deviation,weights=w); #print("MAD Energy = ",MAD_Energy," MeV")
        bool_array = ( (E*electron_mass_MeV>(median_Energy-MAD_Energy)) & (E*electron_mass_MeV>(median_Energy-MAD_Energy)) )
        # compute charge, using all the measured charge
        Q = abs(np.sum(w)*charge_conversion_factor); #print("Q = ",Q," pC")
        
        Energy_spread_percent = median_Energy/(MAD_Energy+1.e-15)*100.
        result = math.sqrt(Q)/Energy_spread_percent
        #print (Q,MAD_Energy,result,math.isinf(result),math.isinf(np.log(result)) )
        if (math.isinf(result) or math.isnan(result) or (MAD_Energy<1.e-2) or (result<1.e-1) or (Q<10.) :
            return 0.
        else:
            return np.maximum(result,0.)
    except:
        return 0.
