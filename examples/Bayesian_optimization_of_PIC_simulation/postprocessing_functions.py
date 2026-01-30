import numpy as np
import happi
import scipy.constants

electron_mass_MeV       = scipy.constants.physical_constants["electron mass energy equivalent in MeV"][0]


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


def get_sqrtCharge_times_median_Energy_over_MAD_Energy_Percent():
    # based on the objective function used in
    # S. Jalas et al, Phys. Rev. Lett. 126, 104801 (2021)
    
    # Open simulation
    S = happi.Open(".", show=False, verbose=False)

    # Open TrackParticles at last timestep
    timestep   = S.TrackParticles(species="electronbunch",sort=False).getAvailableTimesteps()[-1]
    track_part = S.TrackParticles(species="electronbunch",sort=False,
                                  axes=["w", "px","py","pz"],
                                  timestep_indices=-1)
    
    # Extract charge in pC
    charge_bunch_pC = (track_part.getData()[timestep]["w"]
                     * scipy.constants.e
                     * S.namelist.ncrit
                     * (S.namelist.lambda0 / (2 * np.pi)) ** 3
                     * 1e12)  # pC

    # Extract momenta (px only)
    px_bunch        = track_part.getData()[timestep]["px"]
    py_bunch        = track_part.getData()[timestep]["py"]
    pz_bunch        = track_part.getData()[timestep]["pz"]
    p_bunch         = np.sqrt(px_bunch**2+py_bunch**2+pz_bunch**2)

    # Electron energy in MeV
    E_MeV           = np.sqrt(1. + p_bunch**2) * electron_mass_MeV

    # Total charge
    Q               = abs(np.sum(charge_bunch_pC));print(np.size(Q)); print("Q =", Q, "pC")

    # Median energy
    median_Energy   = weighted_median(E_MeV, weights=charge_bunch_pC); print("Median Energy =", median_Energy, "MeV")
    
    # MAD energy
    Energy_deviation = E_MeV - median_Energy
    MAD_Energy = weighted_median(np.abs(Energy_deviation), weights=charge_bunch_pC); print("MAD Energy =", MAD_Energy, "MeV")
    
    # Energy spread
    Energy_spread_percent = (MAD_Energy) / (median_Energy + 1e-15) * 100.0
    result                = np.sqrt(Q) / Energy_spread_percent
    
    del px_bunch,py_bunch,pz_bunch,E_MeV,charge_bunch_pC,p_bunch
    
    if (np.isinf(result) or np.isnan(result)
        or (MAD_Energy < 1e-2)
        or (result < 1e-1)
        or (Q < 10.0)):
        return 0.0
    else:
        return max(result, 0.0)
        

def Plot_energy_spectrum(species="electronbunch",timestep_index=-1,nbins_Energy=200,Emin_MeV=0,Emax_MeV=40):       
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.constants as sc
    import happi
    
    ##### Open the simulation
    S                    = happi.Open()
    #### Define variables used for unit conversions
    try:
        lambda_0         = S.namelist.lambda_0       # laser wavelength, m
    except:
        try:
            lambda_0     = S.namelist.lambda0       # laser wavelength, m
        except:
            lambda_0     = 0.8e-6 
    c                    = sc.c                              # lightspeed, m/s
    omega_0              = 2*np.pi*c/lambda_0              # laser angular frequency, rad/s
    eps0                 = sc.epsilon_0                      # Vacuum permittivity, F/m
    e                    = sc.e                              # Elementary charge, C
    me                   = sc.m_e                            # Electron mass, kg
    ncrit                = eps0*omega_0**2*me/e**2           # Plasma critical number density, m-3   
    electron_mass_MeV    = sc.physical_constants['electron mass energy equivalent in MeV'][0]
    from_weight_to_pC    = e * ncrit * (lambda_0/np.pi/2.)**3 / 1e-12
    
    ### Extract the data from the TrackParticles
    track_part           = S.TrackParticles(species=species, sort=False)
    timesteps            = track_part.getAvailableTimesteps()
    #print timesteps
    timesteps            = list( dict.fromkeys(timesteps) ) # this is to eliminate doubles in case a checkpoint was used
    timestep             = timesteps[timestep_index]
    ## Extract the spectrum data at each timestep
    print("extracting timestep ",timestep,", i.e. c*T = ",timestep*S.namelist.dt*(lambda_0/np.pi/2.) /1e-6," um")

    track_part       = S.TrackParticles(species=species, sort=False,timesteps=timestep).getData()[timestep]
    px               = track_part["px"]               # px momentum array , m_e*c units, each element is one macro-particle
    py               = track_part["py"]               # py momentum array , m_e*c units, each element is one macro-particle
    pz               = track_part["pz"]               # pz momentum array , m_e*c units, each element is one macro-particle
    w                = track_part["w" ]*from_weight_to_pC;print(str(np.sum(w))+" pC") # charge of the macroparticles array, pC

    npart            = px.size # number of macro-particles
    p                = np.sqrt((px**2+py**2+pz**2))               # momentum magnitude array, m_e*c units, each element is one macro-particle
    E_MeV            = np.sqrt(1+np.square(p))*electron_mass_MeV  # energy array in MeV, each element is one macro-particle
    
    # compute the energy spectrum
    hist_energy_axis_MeV = np.linspace(Emin_MeV,Emax_MeV,num=nbins_Energy)   
    hist,bin_edges       = np.histogram(E_MeV,bins=hist_energy_axis_MeV,weights=w)
    Energy_bin_centres   = [(hist_energy_axis_MeV[i]+hist_energy_axis_MeV[i+1])/2. for i in range(len(hist_energy_axis_MeV)-1)]
    bin_width_E          = Energy_bin_centres[1]-Energy_bin_centres[0]
    hist                 = hist/bin_width_E

    # plot
    objective_function_value = get_sqrtCharge_times_median_Energy_over_MAD_Energy_Percent()
    label = f"f = sqrt(Q)*E_med/E_mad = {objective_function_value:.2f}"
    plt.ion()
    plt.figure(1)
    plt.plot(Energy_bin_centres,hist,label=label)
    plt.xlabel("Energy [MeV]")
    plt.ylabel("dQ/dE [pC/MeV]")
