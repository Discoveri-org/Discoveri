##### Author      : Francesco Massimo (Laboratoire de Physique des Gaz et des Plasmas, CNRS)
##### Purpose     : define optimizer classes 
#####               available optimization algorithms: Random Search, Particle Swarm Optimization ("classic", IAPSO, PSO-TPME), Bayesian Optimization 



import random
import numpy as np
import os,sys
from scipy.stats import qmc
import math

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
        
        self.name                        = name                   # name of the optimizer
        self.number_of_iterations              = number_of_iterations         # maximum number of iterations
        self.number_of_samples_per_iteration                 = number_of_samples_per_iteration            # number of samples drawn at each iteration
        self.number_of_dimensions              = number_of_dimensions         # number of dimensions of the parameter space to explore
        self.optimizer_hyperparameters   = kwargs
        self.search_interval             = search_interval        # list of lists, with the boundaries 
        self.optimum_position            = None                   # optimum position found by the optimizer, yielding the swarm_optimum_function_value
        self.optimum_function_value      = float('-inf')          # maximum value found of the function to optimize, in all the the swarm
        self.samples                     = []                     # array containing the explored samples at the present iterations
        self.iteration_number            = 0                      # present iteration number 
        # history of the positions traveled by the particles
        # dimension 0 : iteration
        # dimension 1 : sample number
        # dimension 2 : the first number_of_dimensions indices give the number_of_dimensions components of the isample position at that iteration,
        #               while the last index gives the value of that function found by isample at that iteration
        self.history_samples_positions_and_function_values        = np.zeros(shape=(number_of_iterations,number_of_samples_per_iteration,number_of_dimensions+1)) 
        
        self.initialPrint(**kwargs)   
        
    def initialPrint(self,**kwargs):
        print("\nOptimizer:                          ", self.name)
        print("Number of iterations:                 ",self.number_of_iterations)
        print("Number of dimensions:                 ",self.number_of_dimensions)
        print("Search interval:                      ",self.search_interval)
        print("Hyperparameters provided by the user: ",kwargs,"\n")
    
    def updateSamplesForExploration():
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
        
        self.use_Halton_sequence = kwargs.get('use_Halton_sequence', True)
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
        print("\n RandomSearch initialized")
        
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
    
class ParticleSwarmOptimization(Optimizer):
    # in this Optimizer, each sample is a particle of the swarm
    def __init__(self, name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs):
        super().__init__(name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs)
        
        
        if (self.name=="Particle Swarm Optimization"):

            # "classic" version of Particle Swarm Optimization, but using an inertia term to avoid divergence
            # as described in Y. Shi, R.C. Eberhart, 1998, https://ieeexplore.ieee.org/document/69914
            
            # c1 (cognitive parameter): It determines the weight or influence of the particle's personal optimum position on its velocity update
            # A higher value of c1 gives more importance to the particle's historical optimum position and encourages exploration.
            default_value_c1 = 0.5
            self.c1          = kwargs.get('c1', default_value_c1)
            # c2 (social parameter): It determines the weight or influence of the swarm's swarm optimum position on the particle's velocity update.
            # A higher value of c2 gives more importance to the swarm optimum position and encourages its exploitation.
            default_value_c2 = 0.5
            self.c2          = kwargs.get('c2', default_value_c2)
            # w (inertia weight): It controls the impact of the particle's previous velocity on the current velocity update. 
            # A higher value of w emphasizes the influence of the particle's momentum, promoting exploration.
            # On the other hand, a lower value of w emphasizes the influence of the current optimum positions, promoting exploitation.
            default_value_w  = 0.5
            self.w           = kwargs.get('w', default_value_w)
            # # To avoid too quick particles, this parameter is used to make velocity components 
            # proportional to the search space size in each dimension
            default_value_initial_speed_over_search_space_size = 0.1
            self.initial_speed_over_search_space_size          = kwargs.get('initial_speed_over_search_space_size', default_value_initial_speed_over_search_space_size)
            
            # maximum speed for a particle, it must be a vector with number_of_dimensions elements
            default_max_speed = np.zeros(self.number_of_dimensions)
            for idim in range(0,self.number_of_dimensions):
                default_max_speed[idim]   = self.search_interval[idim][1]-self.search_interval[idim][0]
                    
            self.max_speed  = kwargs.get('max_speed', default_max_speed)
            
            print("\n -- hyperparameters used by the optimizer -- ")
            print("c1                                       = ",self.c1)
            print("c2                                       = ",self.c2)
            print("w                                        = ",self.w)
            print("initial_speed_over_search_space_size  = ",self.initial_speed_over_search_space_size)
            print("max_speed                                = ",self.max_speed )
            print("")

        elif (self.name=="IAPSO"):
            # from Wanli Yang et al 2021 J. Phys.: Conf. Ser. 1754 012195, doi:10.1088/1742-6596/1754/1/012195
            # in this version of PSO: 
            #     the c1 coefficient decreases from 2 to 0 with a sin^2 function;
            #     the c2 coefficient increases from 0 to 2 with a sin^2 function;
            #     the inertia weight decreases exponentially from w1 to w2
            # improvement: the velocities are updated and then the positions, like in a PSO, 
            # and not like in a SPSO, where the position is directly updated (hence the name used in the article IASPSO and the name IAPSO used here).
            
            # w1 (initial inertia weight)
            default_value_w1  = 0.9
            self.w1           = kwargs.get('w1', default_value_w1)
            
            # w2 (final inertia weight)
            default_value_w2  = 0.4
            self.w2           = kwargs.get('w2', default_value_w2)
            
            # w2 (final inertia weight)
            default_value_m   = 10
            self.w2           = kwargs.get('w2', default_value_m)
            
            # # To avoid too quick particles, this parameter is used to make velocity components 
            # proportional to the search space size in each dimension
            default_value_initial_speed_over_search_space_size = 0.1
            self.initial_speed_over_search_space_size          = kwargs.get('initial_speed_over_search_space_size', default_value_initial_speed_over_search_space_size)
            
            # maximum speed for a particle, it must be a vector with number_of_dimensions elements
            default_max_speed = np.zeros(self.number_of_dimensions)
            for idim in range(0,self.number_of_dimensions):
                default_max_speed[idim]   = self.search_interval[idim][1]-self.search_interval[idim][0]
                    
            self.max_speed  = kwargs.get('max_speed', default_max_speed)
            
            
            #### Initialize the acceleration coefficients and the inertia
            # c1 (cognitive parameter): It determines the weight or influence of the particle's personal optimum position on its velocity update. A higher value of c1 gives more importance to the particle's historical optimum position and encourages exploration.
            self.c1                                      = 2.
            # c2 (social parameter): It determines the weight or influence of the swarm's swarm optimum position on the particle's velocity update. A higher value of c2 gives more importance to the swarm optimum position and encourages exploitation.
            self.c2                                      = 1.
            # w (inertia weight): It controls the impact of the particle's previous velocity on the current velocity update. A higher value of w emphasizes the influence of the particle's momentum, promoting exploration. On the other hand, a lower value of w emphasizes the influence of the current optimum positions, promoting exploitation.
            self.w                                       = self.w1 
            
            print("\n -- hyperparameters used by the optimizer -- ")
            print("w1                                       = ",self.w1)
            print("w2                                       = ",self.w2)
            print("m                                        = ",self.m)
            print("initial_speed_over_search_space_size  = ",self.initial_speed_over_search_space_size)
            print("max_speed                                = ",self.max_speed )
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
            
            # c1 (cognitive parameter): It determines the weight or influence of the particle's personal optimum position on its velocity update
            # A higher value of c1 gives more importance to the particle's historical optimum position and encourages exploration.
            default_value_c1 = 1.5
            self.c1          = kwargs.get('c1', default_value_c1)
            # c2 (social parameter): It determines the weight or influence of the swarm's swarm optimum position on the particle's velocity update.
            # A higher value of c2 gives more importance to the swarm optimum position and encourages its exploitation.
            default_value_c2 = 1.5
            self.c2          = kwargs.get('c2', default_value_c2)
            
            # w1 (initial inertia weight)
            default_value_w1  = 0.9
            self.w1           = kwargs.get('w1', default_value_w1)
            
            # w2 (final inertia weight)
            default_value_w2  = 0.4
            self.w2           = kwargs.get('w2', default_value_w2)
            
            # # To avoid too quick particles, this parameter is used to make velocity components 
            # proportional to the search space size in each dimension
            default_value_initial_speed_over_search_space_size = 0.5
            self.initial_speed_over_search_space_size          = kwargs.get('initial_speed_over_search_space_size', default_value_initial_speed_over_search_space_size)
            
            # maximum speed for a particle, it must be a vector with number_of_dimensions elements
            default_max_speed = np.zeros(self.number_of_dimensions)
            for idim in range(0,self.number_of_dimensions):
                default_max_speed[idim]   = self.search_interval[idim][1]-self.search_interval[idim][0]
                    
            self.max_speed  = kwargs.get('max_speed', default_max_speed)
            
            # maximum number of iterations in which "bad" particles are allowed to explore (Ne in the original paper)
            default_Number_of_iterations_bad_particles = 3
            self.Number_of_iterations_bad_particles   = kwargs.get('Number_of_iterations_bad_particles', default_Number_of_iterations_bad_particles)
            
            # percentage p of the mean to define the classification levels mean*(1-p), mean*(1+p)
            default_portion_of_mean_classification_levels = 0.02
            self.portion_of_mean_classification_levels    = kwargs.get('portion_of_mean_classification_levels', default_portion_of_mean_classification_levels )
            
            # "bad" particles that remain "bad" for more than Number_of_iterations_bad_particlesiterations
            # are relocated around the best swarm particle, within an interval (1-a) and (1+a) in all dimensions
            # in this version it will decrease from value a1 to value a2 
            default_amplitude_mutated_range_1    = 0.4
            self.amplitude_mutated_range_1       = kwargs.get('amplitude_mutated_range_1', default_amplitude_mutated_range_1 )
            
            default_amplitude_mutated_range_2    = 0.01
            self.amplitude_mutated_range_2       = kwargs.get('amplitude_mutated_range_2', default_amplitude_mutated_range_2 ) 
            
            ### Initialize inertia and amplitude of the mutated range
            # w (inertia weight): It controls the impact of the particle's previous velocity on the current velocity update. A higher value of w emphasizes the influence of the particle's momentum, promoting exploration. On the other hand, a lower value of w emphasizes the influence of the current optimum positions, promoting exploitation.
            self.w                                       = self.w1
            # the rms width of the gaussian distribution used to relocate hopeless particles is proportional to this amplitude_mutated_range
            self.amplitude_mutated_range                 = self.amplitude_mutated_range_1
            # value of the last average_function_value_of_swarm
            self.best_average_function_value             = float('-inf')
            
            print("\n -- hyperparameters used by the optimizer -- ")
            print("c1                                       = ",self.c1)
            print("c2                                       = ",self.c2)
            print("w1                                       = ",self.w1)
            print("w2                                       = ",self.w2)
            print("initial_speed_over_search_space_size  = ",self.initial_speed_over_search_space_size)
            print("max_speed                                = ",self.max_speed )
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
        self.search_interval_size                    = [search_interval[idim][1]-search_interval[idim][0] for idim in range(0,self.number_of_dimensions)]
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
        search_interval_size = [self.search_interval[idim][1]-self.search_interval[idim][0] for idim in range(0,self.number_of_dimensions)]
        
        if (self.name=="IAPSO"):
            self.c1 = 2.*np.sin(np.pi*(self.number_of_iterations-self.iteration_number)/2./self.number_of_iterations)**2
            self.c2 = 2.*np.sin(np.pi*(self.iteration_number)/2./self.number_of_iterations)**2
            self.w  = self.w2*(self.w1/self.w2)**(1/(1+self.m*self.iteration_number/self.number_of_iterations))
            print("\n",self.name," activated; c1 = ",self.c1,"; c2 = ",self.c2,"; w = ",self.w,"\n")
            
        elif (self.name=="PSO-TPME"):
            self.w  = self.w1-(self.iteration_number+2)*(self.w1-self.w2)/self.number_of_iterations
            self.amplitude_mutated_range  = self.amplitude_mutated_range_1-(self.iteration_number+2)*(self.amplitude_mutated_range_1-self.amplitude_mutated_range_2)/self.number_of_iterations
            print("\n ",self.name," activated; w = ",self.w,", mutation range =",self.amplitude_mutated_range )
            
        
        for iparticle in range(0,self.number_of_samples_per_iteration):
            particle = self.samples[iparticle]
            velocity = particle.velocity
        
            if ((self.name=="Particle Swarm Optimization") or (self.name=="IAPSO")):
                for idim in range(self.number_of_dimensions):
                    # extract two random numbers
                    r1                  = random.random()
                    r2                  = random.random()
                    # compute cognitive velocity, based on the individual particle's exploration
                    cognitive_velocity  = self.c1 * r1 * (particle.optimum_position[idim] - particle.position[idim])
                    # compute social velocity, based on the swarm exploration
                    social_velocity     = self.c2 * r2 * (self.optimum_position[idim] - particle.position[idim])
                    
                    velocity[idim] = self.w * particle.velocity[idim] + cognitive_velocity + social_velocity

                # Update individual particle position
                # limit the velocity to the interval [-max_speed,max_speed]
                for idim in range(0,self.number_of_dimensions):
                    self.samples[iparticle].velocity[idim]      = np.clip(velocity[idim],-self.max_speed[idim],self.max_speed[idim])
                    self.samples[iparticle].position           += velocity
            
            elif (self.name=="PSO-TPME"):
                if (self.particle_category[iparticle]=="good"): # update position only using exploitation of personal best
                    for idim in range(self.number_of_dimensions):
                        # extract one random numbers
                        r1                  = random.random()
                        # compute cognitive velocity, based on the individual particle's exploration
                        cognitive_velocity  = self.c1 * r1 * (particle.optimum_position[idim] - particle.position[idim])
                    
                        velocity[idim] = self.w * particle.velocity[idim] + cognitive_velocity

                    # Update individual particle position
                    # limit the velocity to the interval [-max_speed,max_speed]
                    for idim in range(0,self.number_of_dimensions):
                        self.samples[iparticle].velocity[idim]      = np.clip(velocity[idim],-self.max_speed[idim],self.max_speed[idim])
                        self.samples[iparticle].position           += velocity
                
                elif (self.particle_category[iparticle]=="fair"): # update position as in a classic PSO
                    for idim in range(self.number_of_dimensions):
                        # extract two random numbers
                        r1                  = random.random()
                        r2                  = random.random()
                        # compute cognitive velocity, based on the individual particle's exploration
                        cognitive_velocity  = self.c1 * r1 * (particle.optimum_position[idim] - particle.position[idim])
                        # compute cognitive velocity, based on the swarm exploration
                        social_velocity     = self.c2 * r2 * (self.optimum_position[idim] - particle.position[idim])
                    
                        velocity[idim]      = self.w * particle.velocity[idim] + cognitive_velocity + social_velocity

                    # Update individual particle position
                    # limit the velocity to the interval [-max_speed,max_speed]
                    for idim in range(0,self.number_of_dimensions):
                        self.samples[iparticle].velocity[idim]      = np.clip(velocity[idim],-self.max_speed[idim],self.max_speed[idim])
                        self.samples[iparticle].position           += velocity
                    
                elif (self.particle_category[iparticle]=="bad"):
                    for idim in range(self.number_of_dimensions):
                        # keep converging towards the best position of the swarm
                        # extract one random number
                        r2                  = random.random()
                        # compute social velocity, based on the swarm exploration
                        social_velocity     = self.c2 * r2 * (self.optimum_position[idim] - particle.position[idim])
                        # Update individual particle position
                        velocity[idim]      = self.w*self.samples[iparticle].position [idim] + social_velocity
                    
                    # Update individual particle position
                    # limit the velocity to the interval [-max_speed,max_speed]
                    for idim in range(0,self.number_of_dimensions):
                        self.samples[iparticle].velocity[idim]      = np.clip(velocity[idim],-self.max_speed[idim],self.max_speed[idim])
                        self.samples[iparticle].position           += velocity
                        
                elif (self.particle_category[iparticle]=="hopeless"): 
                    # extract one random number
                    # eta                  = random.random()
                    for idim in range(self.number_of_dimensions):
                        # case of "hopeless particle" ---> relocate it arounf the best position so far
                        #mutation_amplitude   = 2*eta*self.amplitude_mutated_range+(1-self.amplitude_mutated_range)
                        random_number_gaussian = np.random.normal(0, 1, 1)[0]
                        mutation_amplitude     = random_number_gaussian*self.amplitude_mutated_range*(self.search_interval_size[idim])
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
        # update position and velocity of particles, store position history
        self.updateParticlePositionAndVelocity()
        self.iteration_number = self.iteration_number+1
        
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
                        
                     
                     
                     
            
        

class BayesianOptimization(Optimizer):
    def __init__(self, name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs):
        super().__init__(name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs)
        self.num_tests     = 100    # number of points to test through the surrogate function when choosing new samples to draw
        # Define the model for the kernel of the Gaussian process
        # For multidimensional search spaces it is important to use an anisotropic kernel, i.e. with a different scale in each dimension
        self.length_scales = [self.search_interval[idim][1] - self.search_interval[idim][0] for idim in range(self.number_of_dimensions)] # length scales of the kernel in all dimensions
        self.kernel        = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=self.length_scales, nu=1.5)
        self.model         = GaussianProcessRegressor(kernel=self.kernel,optimizer="fmin_l_bfgs_b")
        self.Xsamples      = np.zeros(shape=(self.number_of_samples_per_iteration, self.number_of_dimensions))   # new samples for the new iteration iteration
        # Initial sparse sample
        self.halton_sampler_position   = qmc.Halton(d=self.number_of_dimensions, scramble=True)
        halton_sampler_random_position = self.halton_sampler_position.random(n=self.number_of_samples_per_iteration)
        

        self.X = np.zeros(shape=(self.number_of_samples_per_iteration, self.number_of_dimensions))    # all positions explored by the Bayesian optimization
        self.y = np.zeros((self.number_of_samples_per_iteration,1))                             # all the function values found by the Bayesian optimization
        # Initialize each sample
        for isample in range(self.number_of_samples_per_iteration):
            position = np.zeros(self.number_of_dimensions)
            # use a Halton sequence to sample more uniformly the parameter space
            for idim in range(0,self.number_of_dimensions):
                position[idim]       = search_interval[idim][0]+halton_sampler_random_position[isample][idim]*(search_interval[idim][1]-search_interval[idim][0]) #np.random.uniform(self.search_interval[dimension][0], self.search_interval[dimension][1])
                self.X[isample,idim] = position[idim]
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
        mu, std = self.predictFunctionValueWithSurrogateModel(Xtest)
    
        # Calculate the expected improvement
        z = (mu - best) / (std + 1e-9)
        ei = (mu - best) * norm.cdf(z) + std * norm.pdf(z)
    
        return ei

    # Optimize the acquisition function
    def optimizeAcquisitionFunction(self):
        X_test   = np.zeros(shape=(self.num_tests, self.number_of_dimensions))
        for itest in range(0,self.num_tests):
    	       for idim in range(0,self.number_of_dimensions):
    		             X_test[itest,idim] = np.random.uniform(self.search_interval[idim][0], self.search_interval[idim][1])#, size=(num_tests, number_of_dimensions))
        scores = self.getAcquisitionFunctionResult(X_test)  # Calculate acquisition scores on test points
    
        for isample in range(self.number_of_samples_per_iteration):
            best_index         = np.argmax(scores)   # Find the index with the highest acquisition score
            scores[best_index] = float('-inf')       # Set the acquisition score of the selected index to negative infinity
            self.Xsamples[isample]  = X_test[best_index]  # Select the corresponding sample
    
        return self.Xsamples   
    
    def chooseNewPositionsToExplore(self):
        Xsamples = self.optimizeAcquisitionFunction()
        for isample in range(self.number_of_samples_per_iteration):
            self.samples[isample].position[:] = Xsamples[isample]
            
    def updateSamplesForExploration(self):
        # pick new samples
        self.chooseNewPositionsToExplore()
        self.iteration_number = self.iteration_number+1
    
    def operationsAfterUpdateOfOptimumFunctionValueAndPosition(self):
        # perform some special operations for the data structure needed by the Bayesian Optimization kernel
        if self.iteration_number == 0:
            for isample in range(0,self.number_of_samples_per_iteration):
                self.y[isample] = self.history_samples_positions_and_function_values[self.iteration_number,isample,self.number_of_dimensions]
            self.model.fit(self.X, self.y)
        else:
            for isample in range(0,self.number_of_samples_per_iteration):
                self.X = np.vstack((self.X, self.Xsamples[isample]))
                self.y = np.vstack((self.y, self.history_samples_positions_and_function_values[self.iteration_number,isample,self.number_of_dimensions]))
            self.model.fit(self.X, self.y)
        
            
     