##### Smilei namelist for 1D Laser Wakefield Acceleration simulation
import numpy as np
import scipy.constants as sc

# Configuration to simulate



##### Physical constants

lambda0  = 0.8e-6                                # laser wavelength, m
c        = sc.c                                  # lightspeed, m/s
omega0   = 2*np.pi*c/lambda0                     # laser angular frequency, rad/s

##### Variables used for unit conversions

c_norm   = c/c                                   # normalized lightspeed
ncrit    = sc.epsilon_0*omega0**2*sc.m_e/sc.e**2 # Plasma critical number density, m-3
um       = 1e-6/(lambda0/(2*np.pi))              # 1 micron in normalized units
fs       = 1e-15*omega0                          # 1 femtosecond in normalized units
pC       = 1.e-12/sc.e                           # 1 picoCoulomb in normalized units


############################# Input namelist for Laser Wakefield Acceleration 
############################# with external injection of a relativistic electron bunch

#########################  Simulation parameters

##### mesh resolution
dx       = 0.09*um                               # longitudinal mesh resolution
dr       = 0.6*um                                # transverse mesh resolution

##### Simulation window size
nx       = 576                                   # number of mesh points in the longitudinal direction
nr       = 52                                    # number of mesh points in the transverse direction
Lx       = nx * dx                               # longitudinal size of the simulation window
Lr       = nr * dr                               # transverse size of the simulation window

#### Integration timestep and Total simulated time
dt       = 0.95*dx/c_norm                        # integration timestep
T_sim    = 400*um/c_norm

##### Patches parameters (parallelization)
npatch_x = 32
npatch_r = 4


######################### Main simulation definition block

Main(
    geometry                       = "AMcylindrical",
    number_of_AM                   = 1,

    interpolation_order            = 2,

    timestep                       = dt,
    simulation_time                = T_sim,

    cell_length                    = [ dx, dr],
    grid_length                    = [ Lx,  Lr],

    number_of_patches              = [npatch_x,npatch_r],
 
    EM_boundary_conditions         = [["silver-muller"],["PML"],],
    number_of_pml_cells            = [[0,0],[20,20]],
 
    solve_poisson                  = False,
    solve_relativistic_poisson     = False,

    print_every                    = 100,
    
    use_BTIS3_interpolation        = True,
    
    maxwell_solver                 = "Terzani",

    reference_angular_frequency_SI = omega0,

    random_seed                    = smilei_mpi_rank
)

######################### Define the laser pulse

## laser parameters
laser_fwhm_field    = 25.5*fs*c_norm                                            # laser FWHM duration in field, i.e. FWHM duration in intensity*sqrt(2)
laser_waist         = 12*um                                                     # laser waist, conversion from um
x_center_laser      = Lx-1.7*c_norm*laser_fwhm_field                            # laser position at the start of the simulation
x_focus_laser       = (x_center_laser+0.1*c_norm*laser_fwhm_field)              # laser focal plane position
a0                  = 2.3                                                       # laser peak field, normalized by E0 defined above

## Define a Gaussian bunch with Gaussian temporal envelope
LaserEnvelopeGaussianAM(
   a0               = a0, 
   omega            = omega0/omega0,                                            # laser frequency, normalized
   focus            = [x_focus_laser],                                          # laser focus, [x] position
   waist            = laser_waist,                                              # laser waist
   time_envelope    = tgaussian(center=x_center_laser, fwhm=laser_fwhm_field),  # we choose a Gaussian time envelope for the laser pulse field
   envelope_solver  = 'explicit_reduced_dispersion',
   Envelope_boundary_conditions = [ ["reflective"],["PML"] ],
   Env_pml_sigma_parameters     = [[0.9 ,2     ],[80.0,2]     ,[80.0,2     ]],
   Env_pml_kappa_parameters     = [[1.00,1.00,2],[1.00,1.00,2],[1.00,1.00,2]],
   Env_pml_alpha_parameters     = [[0.90,0.90,1],[0.65,0.65,1],[0.65,0.65,1]]
)


######################### Define a moving window

MovingWindow(
    time_start             = 0.,                                # the simulation window starts  moving at the start of the simulation
    velocity_x             = c_norm,                            # speed of the moving window, normalized by c
)

########################### Define the plasma

###### Plasma plateau density
plasma_density_1_ov_cm3    = 1.e18                              # plasma plateau density in electrons/cm^3
n0                         = plasma_density_1_ov_cm3*1e6/ncrit  # plasma plateau density in units of critical density defined above

##### Initial plasma density distribution
Radius_plasma              = 30.*um                             # Radius of plasma
Lramp                      = 10.*um                             # Plasma density upramp length
Lplateau                   = 280. *um                           # Length of density plateau
Ldownramp                  = 10.*um                             # Length of density downramp
x_begin_upramp             = Lx                                 # x coordinate of the start of the density upramp
x_begin_plateau            = x_begin_upramp + Lramp             # x coordinate of the end of the density upramp / start of density plateau
x_end_plateau              = x_begin_plateau+ Lplateau          # x coordinate of the end of the density plateau start of the density downramp
x_end_downramp             = x_end_plateau  + Ldownramp         # x coordinate of the end of the density downramp

#### Define the density longitudinal profile: a polygonal where at the x coordinates in xpoints.
#### The corresponding unperturbed plasma density is given by the values in xvalues,
#### radially uniform until r=Radius_plasma, out of which the density is zero.
longitudinal_profile = polygonal(xpoints=[x_begin_upramp,x_begin_plateau,x_end_plateau,x_end_downramp],xvalues=[0.,n0,n0,0.])
def plasma_density(x,r):
	profile_r     = np.zeros_like(r)
	profile_r     = np.where((r<Radius_plasma),1.,profile_r)
	return profile_r*longitudinal_profile(x,r)

###### Define the plasma electrons
Species(
 name                      = "plasmaelectrons",
 position_initialization   = "regular",
 momentum_initialization   = "cold",
 particles_per_cell        = 1,       # macro-particles per cell: this number must be equal to the product of the elements of regular_number
 regular_number            = [1,1,1], # distribution of macro-particles per cell in each direction [x,r,theta]
 mass                      =  1.0,    # units of electron mass
 charge                    = -1.0,    # units of electron charge
 number_density            = plasma_density,
 mean_velocity             = [0.0, 0.0, 0.0],
 temperature               = [0.,0.,0.],
 pusher                    = "ponderomotive_borisBTIS3",
 time_frozen               = 0.0,
 boundary_conditions       = [ ["remove", "remove"],["remove", "remove"], ],
)

######################## Define the electron bunch

###### electron bunch parameters
Q_bunch                    = -configuration["bunch_charge_pC"] * pC   # Total charge of the electron bunch
sigma_x                    = 1.  * um                                 # initial longitudinal rms size
sigma_r                    = 1.5 * um                                 # initial transverse/radial rms size (cylindrical symmetry)
bunch_energy_spread        = 0.01                                     # initial rms energy spread / average energy (not in percent)
bunch_normalized_emittance = 2.  * um                                 # initial rms emittance, same emittance for both transverse planes
delay_behind_laser         = configuration["electron_bunch_delay_behind_laser_um"] * um  # distance between x_center_laser and center_bunch
center_bunch               = x_center_laser-delay_behind_laser        # initial position of the electron bunch in the window   
gamma_bunch                = 30.                                      # initial relativistic Lorentz factor of the bunch

n_bunch_particles          = 5000                                     # number of macro-particles to model the electron bunch 
normalized_species_charge  = -1                                       # For electrons
Q_part                     = Q_bunch/n_bunch_particles                # charge for every macro-particle in the electron bunch
weight                     = Q_part/((c/omega0)**3*ncrit*normalized_species_charge)

#### initialize the bunch using numpy arrays
#### the bunch will have n_bunch_particles particles, so an array of n_bunch_particles elements is used to define the x coordinate of each particle and so on ...
array_position             = np.zeros((4,n_bunch_particles))          # positions x,y,z, and weight
array_momentum             = np.zeros((3,n_bunch_particles))          # momenta x,y,z

#### The electron bunch is supposed at waist. To make it convergent/divergent, transport matrices can be used.
#### For the coordinates x,y,z, and momenta px,py,pz of each macro-particle, 
#### a random number is drawn from a Gaussian distribution with appropriate average and rms spread. 
array_position[0,:]        = np.random.normal(loc=center_bunch, scale=sigma_x                           , size=n_bunch_particles)
array_position[1,:]        = np.random.normal(loc=0.          , scale=sigma_r                           , size=n_bunch_particles)
array_position[2,:]        = np.random.normal(loc=0.          , scale=sigma_r                           , size=n_bunch_particles)
array_momentum[0,:]        = np.random.normal(loc=gamma_bunch , scale=bunch_energy_spread*gamma_bunch   , size=n_bunch_particles)
array_momentum[1,:]        = np.random.normal(loc=0.          , scale=bunch_normalized_emittance/sigma_r, size=n_bunch_particles)
array_momentum[2,:]        = np.random.normal(loc=0.          , scale=bunch_normalized_emittance/sigma_r, size=n_bunch_particles)

#### This last array element contains the statistical weight of each macro-particle, 
#### proportional to the total charge of all the electrons it contains
array_position[3,:]        = np.multiply(np.ones(n_bunch_particles),weight)

#### define the electron bunch
Species( 
 name                      = "electronbunch",
 position_initialization   = array_position,
 momentum_initialization   = array_momentum,
 mass                      =  1.0, # units of electron mass
 charge                    = -1.0, # units of electron charge
 pusher                    = "ponderomotive_borisBTIS3", 
 boundary_conditions       = [ ["remove", "remove"],["remove", "remove"],],
)

######################### Diagnostics

##### 1D Probe diagnostic on the x axis
DiagProbe(
    every                  = int(100*um/dt),
    origin                 = [0., 1.*dr, 1.*dr],
    corners                = [ [Main.grid_length[0], 1.*dr, 1.*dr]],
    number                 = [nx],
    fields                 = ['Ex','Ey','Rho','Env_E_abs','BzBTIS3']
)

##### 2D Probe diagnostics on the xy plane
DiagProbe(
    every                  = int(100*um/dt),
    origin                 = [0., -nr*dr,0.],
    corners                = [ [nx*dx,-nr*dr,0.], [0,nr*dr,0.] ],
    number                 = [nx, int(2*nr)],
    fields                 = ['Ex','Ey','Rho','Env_E_abs','BzBTIS3']
)

##### Diagnostic for the electron bunch macro-particles
DiagTrackParticles(
   species                 = "electronbunch",
   every                   = int(100*um/dt),
   attributes              = ["x", "y", "z", "px", "py", "pz", "w"]
)


###### Load balancing (for parallelization when running with more than one MPI process)                                                                                                                                                     
LoadBalancing(
    initial_balance      = False,
    every                = 40,
    cell_load            = 1.,
    frozen_particle_load = 0.1
)