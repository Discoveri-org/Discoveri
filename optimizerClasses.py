##### Author      : Francesco Massimo (Laboratoire de Physique des Gaz et des Plasmas, CNRS)
##### Purpose     : define optimizer classes 
#####               available optimization algorithms: Random Search, Particle Swarm Optimization ("classic", IAPSO, PSO-TPME), Bayesian Optimization 



import random
import numpy as np
import os,sys
from scipy.stats import qmc
import math

# this import is needed for Adaptive Particle Swarm Optimization
from generalUtilities import *

# these imports are necessary for the Bayesian Optimization
import warnings
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel


####### Classes for samples

class Sample:
    def __init__(self):
        self.position = None
        self.velocity = None

## Inherits from the class Sample
class RandomSearchSample(Sample):
    def __init__(self, position):
        self.position = position
        self.optimum_position = position.copy()
        self.optimum_function_value = float('-inf')

## Inherits from the class Sample but has also a velocity 
class SwarmParticle(Sample):
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.optimum_position = position.copy()
        self.optimum_function_value = float('-inf')

class Optimizer:
    def __init__(self, name = "", number_of_samples_per_iteration = 1, number_of_dimensions = 1, search_interval = [], number_of_iterations = 1, **kwargs ):
        
        self.name                            = name                             # name of the optimizer
        self.number_of_iterations            = number_of_iterations             # maximum number of iterations
        self.number_of_samples_per_iteration = number_of_samples_per_iteration  # number of samples drawn at each iteration
        self.number_of_dimensions            = number_of_dimensions             # number of dimensions of the parameter space to explore
        self.optimizer_hyperparameters       = kwargs
        self.search_interval                 = search_interval                  # list of lists, with the boundaries of the space to explore
        self.optimum_position                = None                             # optimum position found by the optimizer, yielding the swarm_optimum_function_value
        self.optimum_function_value          = float('-inf')                    # maximum value found of the function to optimize, in all the the swarm
        self.samples                         = []                               # array containing the explored samples at the present iterations
        self.iteration_number                = 0                                # present iteration number 
        # history of the positions traveled by the particles
        # dimension 0 : iteration
        # dimension 1 : sample number
        # dimension 2 : the first number_of_dimensions indices give the number_of_dimensions components of the isample position at that iteration,
        #               while the last index gives the value of that function found by isample at that iteration
        self.history_samples_positions_and_function_values = np.zeros(shape=(number_of_iterations,number_of_samples_per_iteration,number_of_dimensions+1)) 
        
        # this is used mostly for normalizations
        self.search_interval_size                          = [self.search_interval[idim][1]-self.search_interval[idim][0] for idim in range(0,self.number_of_dimensions)]
        
        
        self.initialPrint(**kwargs)   
        
    def initialPrint(self,**kwargs):
        print("\nOptimizer:                          ", self.name)
        print("Number of iterations:                 ",self.number_of_iterations)
        print("Number of dimensions:                 ",self.number_of_dimensions)
        print("Number of samples per iteration       ",self.number_of_samples_per_iteration)
        print("Search interval:                      ",self.search_interval)
        print("Hyperparameters provided by the user: ",kwargs,"\n")
    
    def updateSamplesForExploration(self):
        print("In an optimizer, you must define a way to pick new positions to explore for each sample")
    
    def operationsAfterUpdateOfOptimumFunctionValueAndPosition(self):
        pass
        
    def updateOptimumFunctionValueAndPosition(self):
        function_values_of_samples   = [self.samples[isample].optimum_function_value for isample in range(0,self.number_of_samples_per_iteration)]
        self.optimum_function_value  = max(function_values_of_samples)
        self.optimum_position        = self.samples[function_values_of_samples.index(self.optimum_function_value)].optimum_position
        
        self.operationsAfterUpdateOfOptimumFunctionValueAndPosition()
    
    def updateSamplePositionAndFunctionHistory(self,iteration,isample,function_value):
        self.history_samples_positions_and_function_values[iteration,isample,0:self.number_of_dimensions]     = self.samples[isample].position[:]
        self.history_samples_positions_and_function_values[iteration,isample,self.number_of_dimensions]       = function_value
    
    def printOptimumFunctionValueAndOptimumPosition(self):
        print("\n# Optimum value    found by the "+self.name+" = ",self.optimum_function_value)
        print("\n# Optimum position found by the "+self.name+" = ",self.optimum_position)
        
    def printSamplesPositions(self):
        #plt.figure(2)
        #plt.colorbar()
        print("\n")
        self.printOptimumFunctionValueAndOptimumPosition()
        for isample in range(self.number_of_samples_per_iteration):
            position = self.samples[isample].position
            print("\n=== Sample", isample) #, "Position:", position)
            print("\n=== optimum position found by sample ",isample," until now: ",self.samples[isample].optimum_position)
            print("\n=== optimum function value found by sample ",isample," until now: ",self.samples[isample].optimum_function_value)
            
    def updateSampleOptimumFunctionValueAndOptimumPosition(self,iparticle,function_value,new_optimum_position):
        self.samples[iparticle].optimum_function_value  = function_value
        self.samples[iparticle].optimum_position[:]     = new_optimum_position[:]
        
    def updateHistoryAndCheckIfFunctionValueIsBetter(self,iteration,isample,function_value):
        # store position and function value in history
        self.updateSamplePositionAndFunctionHistory(iteration,isample,function_value)
        # update the optimum value found by the individual swarm particle if necessary 
        sample = self.samples[isample]
        print("\n ---> Sample", isample, "Position:", sample.position," --> function value at this iteration = ",function_value)
        if function_value>sample.optimum_function_value:
            self.updateSampleOptimumFunctionValueAndOptimumPosition(isample,function_value,sample.position[:])          
            
class RandomSearch(Optimizer):
    def __init__(self, name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs):
        super().__init__(name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs)
        self.sampler_for_position     = qmc.Halton(d=self.number_of_dimensions, scramble=True)
        self.total_sample_index       = 0
        self.random_position_sequence = self.sampler_for_position.random(n=self.number_of_samples_per_iteration*self.number_of_iterations)
        
        
        default_value_use_Halton_sequence  = True
        self.use_Halton_sequence           = kwargs.get('use_Halton_sequence', default_value_use_Halton_sequence)

        # Initialize each sample
        for isample in range(self.number_of_samples_per_iteration):
            position = np.zeros(self.number_of_dimensions)
            if (self.use_Halton_sequence==True):
                # use a Halton sequence to sample more uniformly the parameter space
                for idim in range(0,self.number_of_dimensions):
                    position[idim] = self.search_interval[idim][0]+self.random_position_sequence[isample][idim]*(self.search_interval[idim][1]-self.search_interval[idim][0]) #np.random.uniform(self.search_interval[dimension][0], self.search_interval[dimension][1])
            else: 
                # do not use Halton sequence
                for idim in range(0,self.number_of_dimensions):
                    random_number  = np.random.uniform(0., 1.)
                    position[idim] = self.search_interval[idim][0]+random_number*(self.search_interval[idim][1]-self.search_interval[idim][0]) 
            random_sample          = RandomSearchSample(position) 
            self.samples.append(random_sample)
            self.total_sample_index = self.total_sample_index+1
            print("\n ---> Sample", isample, "Position:", position)  
            self.history_samples_positions_and_function_values[0,isample,0:self.number_of_dimensions] = position[:]
        print("\n Random Search initialized")
        
    def pseudorandomlyExtractNewPositionsToExplore(self):
        for isample in range(0,self.number_of_samples_per_iteration):
            if (self.use_Halton_sequence==True):
                # use a Halton sequence to sample more uniformly the parameter space
                for idim in range(self.number_of_dimensions):
                    # extract random number
                    self.samples[isample].position[idim] = self.search_interval[idim][0]+self.random_position_sequence[self.total_sample_index][idim]*(self.search_interval[idim][1]-self.search_interval[idim][0]) #np.random.uniform(self.search_interval[dimension][0],self.search_interval[dimension][1],1)[0]
            else:
                # do not use Halton sequence
                for idim in range(0,self.number_of_dimensions):
                    random_number                        = np.random.uniform(0., 1.)
                    self.samples[isample].position[idim] = self.search_interval[idim][0]+random_number*(self.search_interval[idim][1]-self.search_interval[idim][0]) 
            self.total_sample_index = self.total_sample_index+1
            
    def updateSamplesForExploration(self):
        # randomly choose new samples, store position history
        self.pseudorandomlyExtractNewPositionsToExplore()

class GridSearch(Optimizer):
    def __init__(self, name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs):
        super().__init__(name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs)
        
        if ( self.number_of_iterations != 1 ):
            print("ERROR: Grid Search only supports number_of_iterations = 1")
            sys.exit()
            
        default_samples_per_dimension  = [self.number_of_samples_per_iteration]
        self.samples_per_dimension     = kwargs.get('samples_per_dimension', default_samples_per_dimension)
        
        self.samples_per_dimension     = np.asarray(self.samples_per_dimension)
        
        if ( np.size(self.samples_per_dimension) != self.number_of_dimensions):
            print("ERROR: samples_per_dimension must be a list of number_of_dimensions integers")
            sys.exit()
        
        if ( np.prod(self.samples_per_dimension) != self.number_of_samples_per_iteration ):
            print("ERROR: samples_per_dimension must be a list of integers, whose product is number_of_samples_per_iteration")
            sys.exit()
        
        print("\n -- hyperparameters used by the optimizer -- ")
        print("samples_per_dimension : ",self.samples_per_dimension)
        
        # create population of samples 
        for isample in range(self.number_of_samples_per_iteration):
            random_sample          = RandomSearchSample(np.zeros(self.number_of_dimensions))
            self.samples.append(random_sample)
        
        # fill the population with the correct position, print and save
        position_arrays_along_dimensions = [np.linspace(self.search_interval[idim][0], self.search_interval[idim][1], num=self.samples_per_dimension[idim]) for idim in range(self.number_of_dimensions)]
        mesh_grid       = np.meshgrid(*position_arrays_along_dimensions, indexing='ij')
        flat_mesh_grid  = np.array(mesh_grid).reshape(number_of_dimensions, -1).T; #print(flat_mesh_grid)
        
        for isample in range(0,self.number_of_samples_per_iteration):
            for idim in range(self.number_of_dimensions):
                self.samples[isample].position[idim]=flat_mesh_grid[isample, idim] 
            print("\n ---> Sample", isample, "Position:", self.samples[isample].position)  
            self.history_samples_positions_and_function_values[0,isample,0:self.number_of_dimensions] = self.samples[isample].position[:]
            
        print("\n Grid Search initialized")

    
class ParticleSwarmOptimization(Optimizer):
    # in this Optimizer, each sample is a particle of the swarm
    def __init__(self, name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs):
        super().__init__(name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs)
        
        
        # c1 (cognitive parameter): It determines the weight or influence of the particle's personal optimum position on its velocity update
        # A higher value of c1 gives more importance to the particle's historical optimum position and encourages exploration.
        default_value_c1  = 2.
        self.c1           = kwargs.get('c1', default_value_c1)
        
        # c2 (social parameter): It determines the weight or influence of the swarm's swarm optimum position on the particle's velocity update.
        # A higher value of c2 gives more importance to the swarm optimum position and encourages its exploitation.
        default_value_c2  = 2.
        self.c2           = kwargs.get('c2', default_value_c2)
        
        if ((self.c1+self.c2)>4.):
            print("ERROR: c1+c2 must be < 4.")
            sys.exit()
        
        # maximum speed for a particle, it must be a vector with number_of_dimensions elements
        default_max_speed = np.zeros(self.number_of_dimensions)
        for idim in range(0,self.number_of_dimensions):
            default_max_speed[idim]   = 0.3*(self.search_interval[idim][1]-self.search_interval[idim][0])
                
        self.max_speed                                     = kwargs.get('max_speed', default_max_speed)
        
        # # To avoid too quick particles, this parameter is used to make velocity components 
        # proportional to the search space size in each dimension
        default_value_initial_speed_over_search_space_size = 0.1
        self.initial_speed_over_search_space_size          = kwargs.get('initial_speed_over_search_space_size', default_value_initial_speed_over_search_space_size)
        
        
        if ((self.initial_speed_over_search_space_size)>1.):
            print("ERROR: initial_speed_over_search_space_size must be < 1.")
            sys.exit()
        
        print("\n -- hyperparameters used by the optimizer -- ")
        
        print("initial_speed_over_search_space_size     = ",self.initial_speed_over_search_space_size)
        print("max_speed                                = ",self.max_speed )
        
        if (self.name=="Particle Swarm Optimization"):

            # "classic" version of Particle Swarm Optimization, but using an inertia term to avoid divergence
            # as described in Y. Shi, R.C. Eberhart, 1998, https://ieeexplore.ieee.org/document/69914
            # the acceleration coefficients and the inertia weight are kept constant
            
            # w (inertia weight): It controls the impact of the particle's previous velocity on the current velocity update. 
            # A higher value of w emphasizes the influence of the particle's momentum, promoting exploration.
            # On the other hand, a lower value of w emphasizes the influence of the current optimum positions, promoting exploitation.
            default_value_w  = 0.9
            self.w           = kwargs.get('w', default_value_w)
            
            if ((self.w)>1.):
                print("ERROR: w must be < 1.")
                sys.exit()
            
            print("c1                                       = ",self.c1)
            print("c2                                       = ",self.c2)
            print("w                                        = ",self.w)
            print("")

        elif (self.name=="Adaptive Particle Swarm Optimization"):
            # from Z.-H. Zhan, J. Zhang, Y. Li; H. S.-H. Chung
            # IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) 39, 6 (2009)
            # https://ieeexplore.ieee.org/document/4812104
            # in this version of PSO: 
            #     the c1,c2,w coefficient are adapted based on the evolutionary state of the swarm
            #     compared to the version in that reference, no fuzzy transition rule state, i.e. only the values of mu_Sx will be used
            
            # w1 (initial inertia weight)
            default_value_w    = 0.9
            self.w             = kwargs.get('w', default_value_w)
            
            if ((self.w)>1.):
                print("ERROR: w must be < 1.")
                sys.exit()
            
            self.history_w     = np.zeros(self.number_of_iterations)
            self.history_c1    = np.zeros(self.number_of_iterations)
            self.history_c2    = np.zeros(self.number_of_iterations)
            self.history_f     = np.zeros(self.number_of_iterations)
            
            self.history_w[0]  = self.w 
            self.history_c1[0] = self.c1
            self.history_c2[0] = self.c2
            self.f             = 0.      # evolutionary factor of the swarm
            
            
            self.history_evolutionary_state = []
            
            # if True, when the swarm evolutionary state is convergence
            # the global best particle position will be "mutated" along one random dimension
            # and the resulting mutated position will be assigned to the worst particle of the swarm. 
            # This is to avoid falling on local minima; if the new position is better than the swarm optimum position, then the swarm will be able to exit a local minimum
            default_value_perturbation_global_best_particle = True
            self.perturbation_global_best_particle = kwargs.get('perturbation_global_best_particle', default_value_perturbation_global_best_particle)
            
            print("c1 (initial)                          = ",self.c1                               )
            print("c2 (initial)                          = ",self.c2                               )
            print("w  (initial)                          = ",self.w                                )
            print("perturbation_global_best_particle     = ",self.perturbation_global_best_particle)
            print("")
            
        elif (self.name=="PSO-TPME"):
            # parameters of the  Particle Swarm Optimization variant described in
            # T. Shaquarin, B. R. Noack, International Journal of Computational Intelligence Systems (2023) 16:6, https://doi.org/10.1007/s44196-023-00183-z
            # changes compared to that article:
            # - the level from which the levels for bad, fair, good are computed will not decrease \
            #   over the iterations: i.e. the maximum between the average of the function to optimize and the previous level is used ;
            #   probably this prevents particles from going all towards "fair" as described in that article
            # - to reinitialize the particles closer to the optimum one, the mutated_amplitude is linearly decreasing,
            #   and the distribution of the coordinates near the optimum particle is a gaussian proportional to the search space size
            #   in that dimension and the mutated amplitude
            
            # w1 (initial inertia weight)
            default_value_w1  = 0.9
            self.w1           = kwargs.get('w1', default_value_w1)
            
            if ( self.w1 >1. ):
                print("ERROR: w1 must be < 1.")
                sys.exit()
            
            # w2 (final inertia weight)
            default_value_w2  = 0.4
            self.w2           = kwargs.get('w2', default_value_w2)
            
            if ( self.w2 > 1. ):
                print("ERROR: w must be < 1.")
                sys.exit()
                
            if ( self.w2 > self.w1) :
                print("ERROR: w2 must be < w1.")
                sys.exit()
                
            
            # maximum number of iterations in which "bad" particles are allowed to explore (Ne in the original paper)
            default_Number_of_iterations_bad_particles         = 3
            self.Number_of_iterations_bad_particles            = kwargs.get('Number_of_iterations_bad_particles', default_Number_of_iterations_bad_particles)
            
            if (self.Number_of_iterations_bad_particles < 1):
                print("ERROR: Number_of_iterations_bad_particles must be >= 1.")
                sys.exit()
            
            # percentage p of the mean to define the classification levels mean*(1-p), mean*(1+p)
            default_portion_of_mean_classification_levels      = 0.02
            self.portion_of_mean_classification_levels         = kwargs.get('portion_of_mean_classification_levels', default_portion_of_mean_classification_levels )
            
            if ( self.portion_of_mean_classification_levels > 1. ):
                print("ERROR: portion_of_mean_classification_levels must be < 1.")
                sys.exit()
                
                
            # "bad" particles that remain "bad" for more than Number_of_iterations_bad_particlesiterations
            # are relocated around the best swarm particle, within an interval (1-a) and (1+a) in all dimensions
            # in this version it will decrease from value a1 to value a2 
            default_amplitude_mutated_range_1                  = 0.4
            self.amplitude_mutated_range_1                     = kwargs.get('amplitude_mutated_range_1', default_amplitude_mutated_range_1 )
            
            if (self.amplitude_mutated_range_1 > 1.):
                print("ERROR: amplitude_mutated_range_1 must be < 1.")
                sys.exit()
            
            default_amplitude_mutated_range_2                  = 0.01
            self.amplitude_mutated_range_2                     = kwargs.get('amplitude_mutated_range_2', default_amplitude_mutated_range_2 )
            
            if (self.amplitude_mutated_range_2 > 1.):
                print("ERROR: amplitude_mutated_range_2 must be < 1.")
                sys.exit() 
                
            if (self.amplitude_mutated_range_2 > self.amplitude_mutated_range_1):
                print("ERROR: amplitude_mutated_range_2 must be < amplitude_mutated_range_1.")
                sys.exit() 
            
            ### Initialize inertia and amplitude of the mutated range
            # w (inertia weight): It controls the impact of the particle's previous velocity on the current velocity update. A higher value of w emphasizes the influence of the particle's momentum, promoting exploration. On the other hand, a lower value of w emphasizes the influence of the current optimum positions, promoting exploitation.
            self.w                                             = self.w1
            # the rms width of the gaussian distribution used to relocate hopeless particles is proportional to this amplitude_mutated_range
            self.amplitude_mutated_range                       = self.amplitude_mutated_range_1
            # value of the last average_function_value_of_swarm
            self.best_average_function_value                   = float('-inf')
            
            print("c1                                       = ",self.c1)
            print("c2                                       = ",self.c2)
            print("w1                                       = ",self.w1)
            print("w2                                       = ",self.w2)
            print("portion_of_mean_classification_levels    = ",self.portion_of_mean_classification_levels)
            print("Number_of_iterations_bad_particles       = ",self.Number_of_iterations_bad_particles)
            print("amplitude_mutated_range_1                = ",self.amplitude_mutated_range_1)
            print("amplitude_mutated_range_2                = ",self.amplitude_mutated_range_2)
            print("")
        
        # if PSO-TPME is used, we need an array with the classification of each particle
        # and the total number of iterations in which that particle has remained in the same category
        self.particle_category                                           = []
        self.particles_number_of_iterations_remained_in_current_category = []
            
        # use scrambled Halton sampler to extract initial positions
        self.halton_sampler_position                 = qmc.Halton(d=self.number_of_dimensions, scramble=True)
        halton_sampler_random_position               = self.halton_sampler_position.random(n=self.number_of_samples_per_iteration)
        
        # Initialize each particle the swarm
        for iparticle in range(0,self.number_of_samples_per_iteration):
            position = np.zeros(self.number_of_dimensions)
            velocity = np.zeros(self.number_of_dimensions)
            
            for idim in range(0,self.number_of_dimensions):
                # use a scrambled Halton sequence to sample more uniformly the parameter space
                position[idim] = self.search_interval[idim][0]+halton_sampler_random_position[iparticle][idim]*self.search_interval_size[idim] #np.random.uniform(search_interval[dimension][0], search_interval[dimension][1])
                # use initial velocity proportional to the search_space size in this dimension
                velocity[idim] = self.initial_speed_over_search_space_size*np.random.uniform(-1, 1)*self.search_interval_size[idim]
                particle       = SwarmParticle(position,velocity) 

            self.samples.append(particle)
            if (self.name=="PSO-TPME"):
                self.particle_category.append("bad")
                self.particles_number_of_iterations_remained_in_current_category.append(0)
                
            print("Particle", iparticle, "Position:", position)  
            self.history_samples_positions_and_function_values[0,iparticle,0:self.number_of_dimensions] = position[:]
            self.samples[iparticle].optimum_position[:]                                           = position[:]
            del position, velocity
            
            
        print("\n"+self.name+" initialized")
        
            
    def updateParticlePositionAndVelocity(self):
                    
        if (self.name=="PSO-TPME"):
            self.w  = self.w1-(self.iteration_number+2)*(self.w1-self.w2)/self.number_of_iterations
            self.amplitude_mutated_range  = self.amplitude_mutated_range_1-(self.iteration_number+2)*(self.amplitude_mutated_range_1-self.amplitude_mutated_range_2)/self.number_of_iterations
            print("\n ",self.name," activated; w = ",self.w,", mutation range =",self.amplitude_mutated_range )
            
        
        for iparticle in range(0,self.number_of_samples_per_iteration):

            if ((self.name=="Particle Swarm Optimization") or (self.name=="Adaptive Particle Swarm Optimization")):
                for idim in range(self.number_of_dimensions):
                    # extract two random numbers
                    r1                  = random.random()
                    r2                  = random.random()
                    # compute cognitive velocity, based on the individual particle's exploration
                    cognitive_velocity  = self.c1 * r1 * (self.samples[iparticle].optimum_position[idim] - self.samples[iparticle].position[idim])
                    # compute social velocity, based on the swarm exploration
                    social_velocity     = self.c2 * r2 * (self.optimum_position[idim] - self.samples[iparticle].position[idim])
                    # update velocity
                    self.samples[iparticle].velocity[idim] = self.w * self.samples[iparticle].velocity[idim] + cognitive_velocity + social_velocity    
                    # limit the velocity to the interval [-max_speed,max_speed]
                    self.samples[iparticle].velocity[idim] = np.clip(self.samples[iparticle].velocity[idim],-self.max_speed[idim],self.max_speed[idim])
                # Update individual particle position
                self.samples[iparticle].position += self.samples[iparticle].velocity
                
            
            elif (self.name=="PSO-TPME"):
                if (self.particle_category[iparticle]=="good"): # update position only using exploitation of personal best
                    for idim in range(self.number_of_dimensions):
                        # extract one random numbers
                        r1                  = random.random()
                        # compute cognitive velocity, based on the individual particle's exploration
                        cognitive_velocity  = self.c1 * r1 * (self.samples[iparticle].optimum_position[idim] - self.samples[iparticle].position[idim])
                        # update velocity
                        self.samples[iparticle].velocity[idim] = self.w * self.samples[iparticle].velocity[idim] + cognitive_velocity
                        # limit the velocity to the interval [-max_speed,max_speed]
                        self.samples[iparticle].velocity[idim] = np.clip(self.samples[iparticle].velocity[idim],-self.max_speed[idim],self.max_speed[idim])
                    # Update individual particle position
                    self.samples[iparticle].position           += self.samples[iparticle].velocity
                
                elif (self.particle_category[iparticle]=="fair"): # update position as in a classic PSO
                    for idim in range(self.number_of_dimensions):
                        # extract two random numbers
                        r1                  = random.random()
                        r2                  = random.random()
                        # compute cognitive velocity, based on the individual particle's exploration
                        cognitive_velocity  = self.c1 * r1 * (self.samples[iparticle].optimum_position[idim] - self.samples[iparticle].position[idim])
                        # compute cognitive velocity, based on the swarm exploration
                        social_velocity     = self.c2 * r2 * (self.optimum_position[idim]     - self.samples[iparticle].position[idim])
                        # update velocity
                        self.samples[iparticle].velocity[idim]      = self.w * self.samples[iparticle].velocity[idim] + cognitive_velocity + social_velocity
                        # limit the velocity to the interval [-max_speed,max_speed]
                        self.samples[iparticle].velocity[idim]      = np.clip(self.samples[iparticle].velocity[idim],-self.max_speed[idim],self.max_speed[idim])
                    # Update individual particle position    
                    self.samples[iparticle].position           += self.samples[iparticle].velocity
                    
                elif (self.particle_category[iparticle]=="bad"):
                    for idim in range(self.number_of_dimensions):
                        # keep converging towards the best position of the swarm
                        # extract one random number
                        r2                  = random.random()
                        # compute social velocity, based on the swarm exploration
                        social_velocity     = self.c2 * r2 * (self.optimum_position[idim] - self.samples[iparticle].position[idim])
                        # Update individual particle position
                        self.samples[iparticle].velocity[idim]      = self.w*self.samples[iparticle].position [idim] + social_velocity
                        # limit the velocity to the interval [-max_speed,max_speed]
                        self.samples[iparticle].velocity[idim]      = np.clip(self.samples[iparticle].velocity[idim],-self.max_speed[idim],self.max_speed[idim])
                    # Update individual particle position
                    self.samples[iparticle].position           += self.samples[iparticle].velocity
                        
                elif (self.particle_category[iparticle]=="hopeless"): 
                    # extract one random number
                    # eta                  = random.random()
                    for idim in range(self.number_of_dimensions):
                        # case of "hopeless particle" ---> relocate it arounf the best position so far
                        #mutation_amplitude   = 2*eta*self.amplitude_mutated_range+(1-self.amplitude_mutated_range)
                        random_number_gaussian = np.random.normal(0, 1, 1)[0]
                        mutation_amplitude     = random_number_gaussian*self.amplitude_mutated_range*(self.search_interval_size[idim])
                        mutation_amplitude     = np.minimum(mutation_amplitude, 3.*self.amplitude_mutated_range*(self.search_interval_size[idim]))
                        mutation_amplitude     = np.maximum(mutation_amplitude,-3.*self.amplitude_mutated_range*(self.search_interval_size[idim]))
                        # relocate particle around the swarm's optimum position
                        #self.samples[iparticle].position [idim] = self.optimum_position[idim]*mutation_amplitude
                        self.samples[iparticle].position [idim] = self.optimum_position[idim]*(1+mutation_amplitude)
                        # reinitialize velocity to avoid being stuck if new position will become best position
                        # use initial velocity proportional to the search_space size in this dimension
                        self.samples[iparticle].velocity[idim] = self.initial_speed_over_search_space_size*np.random.uniform(-1, 1)*self.search_interval_size[idim] 
                    print("\n PSO-TPE mutation of sample ",iparticle,"to position ",self.samples[iparticle].position)
                                    
            # Boundary condition on position:
            # if the particle exits the search_interval in one of its position coordinates,
            # reassign that coordinate randomly
            # don't change the velocity
            for dimension in range(self.number_of_dimensions):
                if ((self.samples[iparticle].position[dimension] < self.search_interval[dimension][0]) or (self.samples[iparticle].position[dimension] > self.search_interval[dimension][1])):
                    self.samples[iparticle].position[dimension] = self.search_interval[dimension][0]+np.random.uniform(0., 1.)*(self.search_interval[dimension][1]-self.search_interval[dimension][0])
                    #self.samples[iparticle].velocity[dimension] = self.initial_speed_over_search_space_size*np.random.uniform(-1, 1)*search_interval_size[dimension]
                
    
    def updateSamplesForExploration(self):
        if (self.name=="Adaptive Particle Swarm Optimization"):
            self.evaluateEvolutionStateAndAdaptHyperparameters()
        # update position and velocity of particles, store position history
        self.updateParticlePositionAndVelocity()
        self.iteration_number = self.iteration_number+1
        
    def evaluateEvolutionStateAndAdaptHyperparameters(self):
        # based on the implementation in the pymoo project https://pymoo.org 
        # (which is probably more efficient than the one you will find in the next lines of code)
        
        # compute for each particle its mean distance from the other particles
        #print(self.history_samples_positions_and_function_values)
        average_distance_from_other_particles = np.zeros(self.number_of_samples_per_iteration)
        for iparticle1 in range(0,self.number_of_samples_per_iteration):
            for iparticle2 in range(0,self.number_of_samples_per_iteration):
                if (iparticle2==iparticle1):
                    normalized_distance_1_from_2 = 0.
                else:
                    normalized_distance_1_from_2 = normalized_euclidean_distance(self.samples[iparticle1].position,self.samples[iparticle2].position,self.search_interval_size)
                #print("normalized distance from particle ",iparticle1," to particle ",iparticle2,"is ",normalized_distance_1_from_2)
                average_distance_from_other_particles[iparticle1] = average_distance_from_other_particles[iparticle1] + normalized_distance_1_from_2
            average_distance_from_other_particles[iparticle1] = average_distance_from_other_particles[iparticle1]/(self.number_of_samples_per_iteration-1.)
            #print("average normalized distances from other particles: ",average_distance_from_other_particles[iparticle1])
        
        # compute average distance from the other particles of the globally best particle
        index_globally_best_particle = np.argmax(self.history_samples_positions_and_function_values[self.iteration_number,:,self.number_of_dimensions])
        d_g                         = average_distance_from_other_particles[index_globally_best_particle]
        
        # compute evolutionary factor f
        d_min  = np.amin(average_distance_from_other_particles)
        d_max  = np.amax(average_distance_from_other_particles)
        self.f = (d_g-d_min)/(d_max-d_min+1e-15) # add a small constant to avoid division by 0.
        self.history_f[self.iteration_number] = self.f

        # compute value of the membership functions mu_Sx for x=1,2,3,4
        mu_values = np.array([mu_S1(self.f), mu_S2(self.f), mu_S3(self.f), mu_S4(self.f)])
        
        # find the highest value for mu_Sx and extract evolutionary state of the Swarm
        # simple evaluation of the evolutionary state, without the transition rule of the original paper
        self.evolutionary_state = mu_values.argmax() + 1
        
        dictionary_evolutionary_state_swarm = {1:"Exploration",2:"Exploitation",3:"Convergence",4:"Jumping Out"}
        
        # set inertia weight based on evolutionary factor f
        self.w = 1. / (1. + 1.5 * np.exp(-2.6 * self.f))
        
        # change acceleration coefficients based on the evolutionary state of the swarm
        delta = 0.05*(1+np.random.random())
        
        if self.evolutionary_state == 1: # Exploration
            self.c1 = self.c1 + delta
            self.c2 = self.c2 - delta
        elif self.evolutionary_state == 2: # Exploitation
            self.c1 = self.c1 + 0.5 * delta
            self.c2 = self.c2 - 0.5 * delta
        elif self.evolutionary_state == 3: # Convergence
            self.c1 = self.c1 + 0.5 * delta
            self.c2 = self.c2 + 0.5 * delta
        elif self.evolutionary_state == 4: # Jumping Out
            self.c1 = self.c1 - delta
            self.c2 = self.c2 + delta
            
        self.c1 = max(1.5, min(2.5, self.c1))
        self.c2 = max(1.5, min(2.5, self.c2))
        
        if (self.c1 + self.c2) > 4.0:
            self.c1 = 4.0 * (self.c1 / (self.c1 + self.c2))
            self.c2 = 4.0 * (self.c2 / (self.c1 + self.c2))
            
        # store some history of the varying terms    
        self.history_c1[self.iteration_number] = self.c1
        self.history_c2[self.iteration_number] = self.c2
        self.history_w [self.iteration_number] = self.w
        self.history_evolutionary_state.append(self.evolutionary_state)
        
        print("\n",self.name,", f = ",self.f,"--> evolutionary state: ",dictionary_evolutionary_state_swarm[self.evolutionary_state],"; c1 = ",self.c1,"; c2 = ",self.c2,"; w = ",self.w,"\n")


        if ( (self.perturbation_global_best_particle==True) and ( self.evolutionary_state == 3) ): # if the swarm is at convergence and perturbation is activated
        
            # find the particle with worst function value
            index_worst_particle      = np.argmin(self.history_samples_positions_and_function_values[self.iteration_number,:,self.number_of_dimensions])
            
            # find the particle with the best function value
            index_best_particle       = np.argmax(self.history_samples_positions_and_function_values[self.iteration_number,:,self.number_of_dimensions])
            position_best_particle    = self.samples[index_best_particle].position[:]
            
            # randomly draw the index of the coordinate to mutate 
            index_dimension_to_mutate = random.randint(0, self.number_of_dimensions-1)
            
            # standard deviation of the gaussian for the mutation
            sigma_max                 = 1.0 
            sigma_min                 = 0.1
            rms_width_mutation        = sigma_max-(self.iteration_number+2)*(sigma_max-sigma_min)/self.number_of_iterations
            rms_width_mutation        = rms_width_mutation*(self.search_interval_size[index_dimension_to_mutate])
            
            # start from the best particle position
            # generate new coordinates until one is found within the search_interval in that dimension,
            # by adding to this best position a float drawn from a Gaussian distribution
            candidate_mutated_coordinate = float('-inf')
            while ( (candidate_mutated_coordinate > self.search_interval[index_dimension_to_mutate][1]) \
                or (candidate_mutated_coordinate < self.search_interval[index_dimension_to_mutate][0]) ):
                candidate_mutated_coordinate = position_best_particle[index_dimension_to_mutate]+rms_width_mutation*np.random.normal(0, 1, 1)[0]
            
            # and substitute this mutated position to the position of the worst particle of the swarm
            # do not change its velocity
            self.samples[index_worst_particle].position[:]                         = position_best_particle[:]
            self.samples[index_worst_particle].position[index_dimension_to_mutate] = candidate_mutated_coordinate
            
            print("Particle ",index_worst_particle," mutated to position ",self.samples[index_worst_particle].position[:])
            
            
        
    def operationsAfterUpdateOfOptimumFunctionValueAndPosition(self):
        
        if (self.name=="PSO-TPME"):
            average_function_value_of_swarm = np.average(self.history_samples_positions_and_function_values[self.iteration_number,:,self.number_of_dimensions])
            average_function_value_of_swarm = np.maximum(average_function_value_of_swarm,self.best_average_function_value)
            if (average_function_value_of_swarm > self.best_average_function_value):
                self.best_average_function_value = average_function_value_of_swarm
            bad_function_value_level        = average_function_value_of_swarm*(1.-self.portion_of_mean_classification_levels)
            good_function_value_level       = average_function_value_of_swarm*(1.+self.portion_of_mean_classification_levels)
                
            print("\n PSO-TPME: Bad, average and good function value levels :",bad_function_value_level,average_function_value_of_swarm,good_function_value_level)
            old_particle_category       = self.particle_category
            
            for isample in range(0,self.number_of_samples_per_iteration):
                    
                if self.history_samples_positions_and_function_values[self.iteration_number,isample,self.number_of_dimensions] > good_function_value_level:
                    self.particle_category[isample] = "good"
                    self.particles_number_of_iterations_remained_in_current_category[isample] = 1
                elif self.history_samples_positions_and_function_values[self.iteration_number,isample,self.number_of_dimensions]< bad_function_value_level:
                    if (self.particles_number_of_iterations_remained_in_current_category[isample]>self.Number_of_iterations_bad_particles-1):
                        self.particle_category[isample] = "hopeless"
                        self.particles_number_of_iterations_remained_in_current_category[isample] = 1
                    else:
                        if (self.iteration_number==0):
                            self.particle_category[isample] = "bad" 
                            self.particles_number_of_iterations_remained_in_current_category[isample] = 1
                        else:
                            if (old_particle_category[isample]=="bad"):
                                self.particle_category[isample] = "bad" 
                                self.particles_number_of_iterations_remained_in_current_category[isample] = self.particles_number_of_iterations_remained_in_current_category[isample]+1
                            else:
                                self.particle_category[isample] = "bad" 
                                self.particles_number_of_iterations_remained_in_current_category[isample] = 1
                else:
                    self.particle_category[isample] = "fair"
                    self.particles_number_of_iterations_remained_in_current_category[isample] = 1
     
                print("Particle ",isample,"with function value ","{:.3f}".format(self.history_samples_positions_and_function_values[self.iteration_number,isample,self.number_of_dimensions])," is classified as ",self.particle_category[isample], \
                            ", remained in this category for",self.particles_number_of_iterations_remained_in_current_category[isample]," iterations")
                            
    def APSOLastEvaluationAndDumpHyperparameters(self): # used only for Adaptive Particle Swarm Optimization
        self.evaluateEvolutionStateAndAdaptHyperparameters()
        with open('history_evolutionary_factor_f.npy', 'wb') as f:
            np.save( f, self.history_f)
        with open('history_c1.npy', 'wb') as f:
            np.save( f, self.history_c1)
        with open('history_c2.npy', 'wb') as f:
            np.save( f, self.history_c2)
        with open('history_w.npy', 'wb') as f:
            np.save( f, self.history_w)
            
    def APSOSavePartialHyperparametersHistory(self):
        with open('history_evolutionary_factor_f_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.history_f[0:self.iteration_number])
        with open('history_c1_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.history_c1[0:self.iteration_number])
        with open('history_c2_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.history_c2[0:self.iteration_number])
        with open('history_w_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.history_w[0:self.iteration_number])
        
                     

class BayesianOptimization(Optimizer):
    def __init__(self, name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs):
        super().__init__(name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs)
        self.num_tests     = 100    # number of points to test through the surrogate function when choosing new samples to draw
        # Define the model for the kernel of the Gaussian process
        # the kernel scales are normalized, so the input data to fit and predict must be normalized
        self.kernel        = ConstantKernel(1.0, constant_value_bounds="fixed")*Matern(nu=1.5) 
        self.model         = GaussianProcessRegressor(optimizer="fmin_l_bfgs_b",kernel=self.kernel) #(kernel=self.kernel,optimizer="fmin_l_bfgs_b")
        self.Xsamples      = np.zeros(shape=(self.number_of_samples_per_iteration, self.number_of_dimensions))   # new samples for the new iteration iteration
        # Initial sparse sample
        self.halton_sampler_position   = qmc.Halton(d=self.number_of_dimensions, scramble=True)
        halton_sampler_random_position = self.halton_sampler_position.random(n=self.number_of_samples_per_iteration)
        
        
        # NOTE: X is normalized by the search_interval_size in each dimension
        self.X = np.zeros(shape=(self.number_of_samples_per_iteration, self.number_of_dimensions)) # all normalized positions explored by the Bayesian Optimization
        self.y = np.zeros((self.number_of_samples_per_iteration,1))                                # all the function values found by the Bayesian Optimization
        # Initialize each sample
        for isample in range(self.number_of_samples_per_iteration):
            position = np.zeros(self.number_of_dimensions)
            # use a Halton sequence to sample more uniformly the parameter space
            for idim in range(0,self.number_of_dimensions):
                position[idim]       = search_interval[idim][0]+halton_sampler_random_position[isample][idim]*(search_interval[idim][1]-search_interval[idim][0]) #np.random.uniform(self.search_interval[dimension][0], self.search_interval[dimension][1])
                self.X[isample,idim] = position[idim]/self.search_interval_size[idim] # normalize data
            random_sample            = RandomSearchSample(position) 
            self.samples.append(random_sample)
            print("\n ---> Sample", isample, "Position:", position)  
            self.history_samples_positions_and_function_values[0,isample,0:self.number_of_dimensions] = position[:]
                
        # for isample in range(self.number_of_samples_per_iteration):
        #     for idim in range(0,self.number_of_dimensions):
        #         self.X[isample,idim] = np.random.uniform(self.search_interval[idim][0], self.search_interval[idim][1])   #, size=number_of_dimensions)
        #     self.y[isample] = objective(X[isample], noise=0.)
        # 
        # # Initial Fit the model to the initial sparse data
        # self.model.fit(self.X, self.y)
        
        print("\n Bayesian Optimization initialized") 
    # Surrogate or approximation for the objective function
    def predictFunctionValueWithSurrogateModel(self, input_position_array):
        # Catch any warning generated when making a prediction
        with warnings.catch_warnings():
            # Ignore generated warnings
            warnings.filterwarnings("ignore")
            return self.model.predict(input_position_array, return_std=True)
        
    # Expected improvement acquisition function
    def getAcquisitionFunctionResult(self, Xtest):
        # Xtest are test positions that are evaluated to pick new sample positions
        
        # Calculate the best surrogate score found so far
        yhat, _ = self.predictFunctionValueWithSurrogateModel(self.X)
        best = np.max(yhat)
    
        # Calculate mean and standard deviation via surrogate model
        mu  = np.zeros(shape=np.shape(Xtest[:,0]))
        std = np.zeros(shape=np.shape(Xtest[:,0]))
        ei  = np.zeros(shape=np.shape(Xtest[:,0]))
        # this should be probably vectorized
        for i in range(0,np.size(Xtest[:,0])):
            mu[i], std[i] = self.predictFunctionValueWithSurrogateModel(  Xtest[i,:].reshape(1,np.size(Xtest[0,:]))    );
            z = (mu[i] - best) / (std[i] + 1e-9)
            ei[i] = (mu[i] - best) * norm.cdf(z) + std[i] * norm.pdf(z)
        return ei

    # Optimize the acquisition function
    def optimizeAcquisitionFunction(self):
        X_test   = np.zeros(shape=(self.num_tests, self.number_of_dimensions)) # normalized
        for itest in range(0,self.num_tests): # remember that you need to feed normalized inputs to the model
    	       for idim in range(0,self.number_of_dimensions):
    		             X_test[itest,idim] = np.random.uniform(self.search_interval[idim][0]/self.search_interval_size[idim], self.search_interval[idim][1]/self.search_interval_size[idim]) #np.random.uniform(self.search_interval[idim][0], self.search_interval[idim][1])#, size=(num_tests, number_of_dimensions))
        scores = self.getAcquisitionFunctionResult(X_test)  # Calculate acquisition scores on test points
        for isample in range(self.number_of_samples_per_iteration):
            best_index         = np.argmax(scores)   # Find the index with the highest acquisition score
            scores[best_index] = float('-inf')       # Set the acquisition score of the selected index to negative infinity
            self.Xsamples[isample]  = X_test[best_index]  # Select the corresponding sample
        return self.Xsamples   
    
    def chooseNewPositionsToExplore(self):
        Xsamples = self.optimizeAcquisitionFunction()
        # denormalize because Xsamples is normalized, but the positions are not
        for isample in range(self.number_of_samples_per_iteration):
            for idim in range(0,self.number_of_dimensions):
                self.samples[isample].position[idim] = Xsamples[isample,idim]*self.search_interval_size[idim] 
            
    def updateSamplesForExploration(self):
        # pick new samples
        self.chooseNewPositionsToExplore()
        self.iteration_number = self.iteration_number+1
    
    def operationsAfterUpdateOfOptimumFunctionValueAndPosition(self):
        # perform some special operations for the data structure needed by the Bayesian Optimization kernel
        # X is normalized
        if self.iteration_number == 0:
            for isample in range(0,self.number_of_samples_per_iteration):
                self.y[isample] = self.history_samples_positions_and_function_values[self.iteration_number,isample,self.number_of_dimensions]
            self.model.fit(self.X, self.y)
        else:
            for isample in range(0,self.number_of_samples_per_iteration):
                self.X = np.vstack((self.X, self.Xsamples[isample]))
                self.y = np.vstack((self.y, self.history_samples_positions_and_function_values[self.iteration_number,isample,self.number_of_dimensions]))
            self.model.fit(self.X, self.y)
        
class GeneticAlgorithm(Optimizer):
    def __init__(self, name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs):
        super().__init__(name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs)
        
        # Implementation of a basic genetic algorithm
        # to avoid function re-evaluation, the samples stored by this optimizer 
        # represent only the children generated at the present iteration.
        # The set of children + parents is called population.
        # At the first iteration, the population is made of the initial samples, 
        # while in the other iterations the population is made of 
        # population = children (which are number_of_samples_per_iteration) + the parents (which are number_of_parents)
        # The parents are selected from the population of the previous iteration.
        
        self.population_positions       = []
        self.population_function_values = []
        
        default_number_of_parents       = int(0.3*self.number_of_samples_per_iteration)
        self.number_of_parents          = kwargs.get('number_of_parents', default_number_of_parents)
        if (self.number_of_parents > self.number_of_samples_per_iteration):
            print("ERROR: the number_of_parents must be smaller than the number_of_samples_per_iteration")
            sys.exit()
        if (self.number_of_parents < 2):
            print("ERROR: the total number_of_parents must be at least 2")
            sys.exit()
        
        default_probability_of_mutation = 0.1
        self.probability_of_mutation    = kwargs.get('probability_of_mutation', default_probability_of_mutation)
        if ( (self.probability_of_mutation < 0.) or (self.probability_of_mutation > 1.) ):
            print("ERROR: probability_of_mutation must be a float between 0. and 1.")
            sys.exit()
        
        # Initialize each sample
        self.halton_sampler_position   = qmc.Halton(d=self.number_of_dimensions, scramble=True)
        halton_sampler_random_position = self.halton_sampler_position.random(n=self.number_of_samples_per_iteration)
        for isample in range(self.number_of_samples_per_iteration):
            position = np.zeros(self.number_of_dimensions)
            # use a Halton sequence to sample more uniformly the parameter space
            for idim in range(0,self.number_of_dimensions):
                position[idim]       = search_interval[idim][0]+halton_sampler_random_position[isample][idim]*(search_interval[idim][1]-search_interval[idim][0]) 
                
            random_sample            = RandomSearchSample(position) 
            self.samples.append(random_sample)
            
            # at the first iteration, the population is made just by the initial samples
            self.population_positions.append(position)
            
            print("\n ---> Sample", isample, "Position:", position)  
            self.history_samples_positions_and_function_values[0,isample,0:self.number_of_dimensions] = position[:]
    
    def updateSamplesForExploration(self):
        
        # Select parents, i.e. the number_of_parents positions among the present population 
        # which obtained the highest values of the function to optimize
        # remove the other positions from the population
        self.addChildrenFunctionValuesToPopulationFunctionValues()
        self.selectParents()
        
        # Generate new children
        children_positions = self.crossover()
        children_positions = self.mutateChildren(children_positions)
        
        # Build population fot the next iteration
        # population = parents + new children
        self.addChildrenToPopulation(children_positions)
        
        # The new samples of the optimizer will be the children that were just generated
        self.addChildrenToSamples(children_positions)
        
        # update iteration number
        self.iteration_number = self.iteration_number+1
        
    def addChildrenFunctionValuesToPopulationFunctionValues(self):
        for isample in range(0,self.number_of_samples_per_iteration):
            self.population_function_values.append(self.history_samples_positions_and_function_values[self.iteration_number,isample,self.number_of_dimensions])
            
    def selectParents(self):
        # Create a temporary list of tuples (element, index)
        temp_population_function_values = [(element, index) for index, element in enumerate(self.population_function_values)]
        # Sort the temporary list based on the elements (descending order)
        temp_population_function_values.sort(reverse=True)
        # Get the first n indices from the sorted list
        indices_parents = [index for _, index in temp_population_function_values[:self.number_of_parents]]
        # Remove the samples which are not selected as parents from the population list
        self.population_positions       = [elem for i, elem in enumerate(self.population_positions)       if i in indices_parents]
        self.population_function_values = [elem for i, elem in enumerate(self.population_function_values) if i in indices_parents]
        
        
    def crossover(self):
        children_positions = []
        position = np.zeros(self.number_of_dimensions)

        # for each child
        for ichild in range(0,self.number_of_samples_per_iteration):
            # draw a random couple of parents from the selected parents
            
            # index of parent 1
            random_index_1     = random.randint(0, len(self.population_positions)-1)
            random_index_2     = random_index_1 
            
            while ( random_index_2 == random_index_1 ):
                # index of parent 2
                random_index_2 = random.randint(0, len(self.population_positions)-1)
                

            parent_1_position  = self.population_positions[random_index_1]
            parent_2_position  = self.population_positions[random_index_2]
            
            # Arithmetic Crossover applied to all genes of the children and parents:
            # for each dimension the value for the child is the average of the values of the two parents
            # this part can change in different variants of genetic algorithms
            position = 0.5 * (parent_1_position + parent_2_position)
            
            
            children_positions.append(position)

        return children_positions
    
    def mutateChildren(self,children_positions):
        for child_position in children_positions:
            for idim in range(0,self.number_of_dimensions):
                if (np.random.rand() < self.probability_of_mutation):
                    child_position[idim] = np.random.uniform(self.search_interval[idim][0], self.search_interval[idim][1])
        return children_positions
        
    def addChildrenToPopulation(self,children_positions):
        print("Parents:")
        for parent_position in self.population_positions:
            print(" ",parent_position)
        print("Children:")
        for child_position in children_positions:
            print(" ",child_position)
            self.population_positions.append(child_position)
    
    def addChildrenToSamples(self,children_positions):
        for isample in range(0,len(children_positions)):
            self.samples[isample].position[:] = children_positions[isample][:]
            
     
        
        
        
        
                  
     