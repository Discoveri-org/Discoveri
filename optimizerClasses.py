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
        self.search_interval_size            = [self.search_interval[idim][1]-self.search_interval[idim][0] for idim in range(0,self.number_of_dimensions)]
        
        # used for optimizers with a predictive model e.g. Bayesian Optimization
        self.number_of_tests                 = 1
        self.model                           = None
        self.xi                              = 0.
        self.X                               = np.zeros(1)
        self.y                               = np.zeros(1)
        
        
        self.initialPrint(**kwargs)   
        
    def initialPrint(self,**kwargs):
        print("")
        print("Optimizer                                = ", self.name)
        print("Number of iterations                     = ",self.number_of_iterations)
        print("Number of dimensions                     = ",self.number_of_dimensions)
        print("Number of samples per iteration          = ",self.number_of_samples_per_iteration)
        print("Search interval                          = ",self.search_interval)
        print("Hyperparameters provided by the user     = ",kwargs,"\n")
    
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
            
    # Optimize the acquisition function
    def optimizeAcquisitionFunction(self,number_of_new_samples_to_choose):
        
        # this function will pick the (maybe) best number_new_samples positions
        # to try, found by the a acquisition function 
        
        # Vectorized version
        # Generate normalized test points
        X_test_with_surrogate_model = np.random.uniform(
            [interval[0] / size for interval, size in zip(self.search_interval, self.search_interval_size)],
            [interval[1] / size for interval, size in zip(self.search_interval, self.search_interval_size)],
            size=(self.number_of_tests, self.number_of_dimensions)
        )

        # Calculate acquisition scores on test points
        scores = getExpectedImprovementAcquisitionFunctionResult(self.model,X_test_with_surrogate_model,self.X, self.xi)

        # Select top samples based on acquisition scores
        best_indices  = np.argsort(scores)[::-1][:number_of_new_samples_to_choose]
        XsamplesChosenWithSurrogateModel = X_test_with_surrogate_model[best_indices]

        return XsamplesChosenWithSurrogateModel         
            
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
        print("samples_per_dimension                    = ",self.samples_per_dimension)
        
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
        
        if (self.number_of_samples_per_iteration < 2):
            print("ERROR: in any variant of Particle Swarm Optimization there should be at least two particles in the swarm,")
            print("i.e. number_of_samples_per_iteration must be at least equal to 2.")
            sys.exit()
        
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
            default_max_speed[idim]   = 0.4*(self.search_interval[idim][1]-self.search_interval[idim][0])
                
        self.max_speed                                     = kwargs.get('max_speed', default_max_speed)
        
        print("\n -- hyperparameters used by the optimizer -- ")
        
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
            
            print("max_speed                                = ",self.max_speed )
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
            #     compared to the version in that reference, no fuzzy classification is used for the evolutionary state.
            #     the intervals for this classification based on the parameter f are just 
            #     Convergence: [0,0.25), Exploitation: [0.25,0.5), Exploration: [0.5,0.75), Jumping-Out: [0.75,1.)]
            
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
            
            self.history_mu    = np.zeros(self.number_of_iterations)
            
            self.history_evolutionary_state = []
            
            # if True, when the swarm evolutionary state is convergence
            # the global best particle position will be "mutated" along one random dimension
            # and the resulting mutated position will be assigned to the worst particle of the swarm. 
            # This is to avoid falling on local minima; if the new position is better than the swarm optimum position, then the swarm will be able to exit a local minimum
            default_value_perturbation_global_best_particle = True
            self.perturbation_global_best_particle = kwargs.get('perturbation_global_best_particle', default_value_perturbation_global_best_particle)
            
            # use PSO variant with state-based adaptive velocity limit strategy (PSO-SAVL)
            # described in X. Li et al., Neurocomputing 447 (2021) 64–79, https://doi.org/10.1016/j.neucom.2021.03.077
            # the maximum velocity in each dimension idim, i.e. mu*(search_space_size[idim]) will be adaptively changed
            # depending on the evolutionary state, with mu chosen within the interval [mu_min,mu_max]
            default_mu_min     = 0.4
            self.mu_min        = kwargs.get('mu_min', default_mu_min)
            
            default_mu_max     = 0.7
            self.mu_max        = kwargs.get('mu_max', default_mu_max)
            
            if ( self.mu_max < self.mu_min ):
                print("ERROR: mu_max must be larger than mu_min")
                sys.exit()
                
            self.alpha         = (1./self.mu_min) - 1.
            self.beta          = -math.log( (1./self.mu_max-1.)/self.alpha )
            
            self.mu            = self.mu_max
            self.history_mu[0] = self.mu
            
            # start with the maximum speed initially to promote exploration
            self.max_speed = np.zeros(self.number_of_dimensions)
            for idim in range(0,self.number_of_dimensions):
                self.max_speed[idim] = self.mu*(self.search_interval[idim][1]-self.search_interval[idim][0])
            
            print("max_speed                                = ",self.max_speed )
            print("c1 (initial)                             = ",self.c1                               )
            print("c2 (initial)                             = ",self.c2                               )
            print("w  (initial)                             = ",self.w                                )
            print("perturbation_global_best_particle        = ",self.perturbation_global_best_particle)
            print("mu_min                                   = ",self.mu_min                           )
            print("mu_max                                   = ",self.mu_max                           )
            print("")
        
        # flag to substitute number_of_particles_from_regression particles (those with worst fitness value in the previous iteration) 
        # every number_of_iterations_between_regressions iterations 
        # with positions picked like in a Bayesian Optimization
        default_value_relocateParticlesWithRegression                = False
        self.relocateParticlesWithRegression                         = kwargs.get('relocateParticlesWithRegression', default_value_relocateParticlesWithRegression)
        
        default_value_number_of_particles_from_regression            = 1
        self.number_of_particles_from_regression                     = kwargs.get('number_of_particles_from_regression', default_value_number_of_particles_from_regression)
        
        default_value_number_of_iterations_between_regressions       = 1
        self.number_of_iterations_between_regressions                = kwargs.get('number_of_iterations_between_regressions', default_value_number_of_iterations_between_regressions)
        
        
        if (self.relocateParticlesWithRegression==True):
            ### Define the regression model with a Gaussian process
            self.number_of_tests,self.nu,self.length_scale,self.length_scale_bounds,self.xi,\
            self.kernel,self.model,self.XsamplesChosenWithSurrogateModel \
            = initializePredictiveModelForOptimization(number_of_samples_to_choose=self.number_of_particles_from_regression,number_of_dimensions=self.number_of_dimensions,**kwargs)
            
            print("\n -- hyperparameters used for the regression -- ")
            
            print("relocateParticlesWithRegression          = ",self.relocateParticlesWithRegression)
            print("number_of_particles_from_regression      = ",self.number_of_particles_from_regression)
            print("number_of_iterations_between_regressions = ",self.number_of_iterations_between_regressions)
            print("number_of_tests                          = ",self.number_of_tests)
            print("nu                                       = ",self.nu)
            print("length_scale                             = ",self.length_scale)
            print("length_scale_bounds                      = ",self.length_scale_bounds)
            print("")
            
            # NOTE: X is normalized by the search_interval_size in each dimension
            self.X = np.zeros(shape=(self.number_of_samples_per_iteration, self.number_of_dimensions)) # all normalized positions explored by the Bayesian Optimization
            self.y = np.zeros((self.number_of_samples_per_iteration,1))                                # all the function values found by the Bayesian Optimization
            
            
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
                velocity[idim] = self.max_speed[idim]*np.random.uniform(-1, 1)
                particle       = SwarmParticle(position,velocity)
            
            self.samples.append(particle)
            
            if(self.relocateParticlesWithRegression==True):    
                self.X[iparticle,idim] = position[idim]/self.search_interval_size[idim] # normalize data
                
            print("Particle", iparticle, "Position:", position)  
            self.history_samples_positions_and_function_values[0,iparticle,0:self.number_of_dimensions] = position[:]
            self.samples[iparticle].optimum_position[:]                                                 = position[:]
            del position, velocity
            
        
            
        print("\n"+self.name+" initialized")
        
            
    def updateParticlePositionAndVelocity(self):    
        
        for iparticle in range(0,self.number_of_samples_per_iteration):
            
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
                                    
            # Boundary condition on position:
            # if the particle exits the search_interval in one of its position coordinates,
            # reassign that coordinate randomly within the search_interval boundaries;
            # don't change the velocity
            for idim in range(self.number_of_dimensions):
                if ((self.samples[iparticle].position[idim] < self.search_interval[idim][0]) or (self.samples[iparticle].position[idim] > self.search_interval[idim][1])):
                    self.samples[iparticle].position[idim] = self.search_interval[idim][0]+np.random.uniform(0., 1.)*(self.search_interval[idim][1]-self.search_interval[idim][0])
                    #self.samples[iparticle].velocity[dimension] = self.max_speed[idim]*np.random.uniform(-1, 1)
                
    
    def updateSamplesForExploration(self):
        if (self.name=="Adaptive Particle Swarm Optimization"):
            self.evaluateEvolutionStateAndAdaptHyperparameters()
        # update position and velocity of particles, store position history
        self.updateParticlePositionAndVelocity()
        self.iteration_number = self.iteration_number+1
        
    def evaluateEvolutionStateAndAdaptHyperparameters(self):
        
        # Vectorized version 
        normalized_distances = np.zeros((self.number_of_samples_per_iteration, self.number_of_samples_per_iteration))
        for i in range(self.number_of_samples_per_iteration):
            for j in range(self.number_of_samples_per_iteration):
                if j != i:
                    normalized_distances[i, j] = normalized_euclidean_distance(
                        self.samples[i].position,
                        self.samples[j].position,
                        np.asarray(self.search_interval_size)
                    )
        sum_distances = np.sum(normalized_distances, axis=1)
        average_distance_from_other_particles = sum_distances / (self.number_of_samples_per_iteration - 1)
        
        # compute average distance from the other particles of the globally best particle
        index_globally_best_particle = np.argmax(self.history_samples_positions_and_function_values[self.iteration_number,:,self.number_of_dimensions])
        d_g                         = average_distance_from_other_particles[index_globally_best_particle]
        
        # compute evolutionary factor f
        d_min  = np.amin(average_distance_from_other_particles)
        d_max  = np.amax(average_distance_from_other_particles)
        self.f = (d_g-d_min)/(d_max-d_min+1e-15) # add a small constant to avoid division by 0.
        self.history_f[self.iteration_number] = self.f

        # simple evaluation of the evolutionary state, without the transition rule of the original paper
        self.evolutionary_state = evolutionary_state(self.f)
        
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
            
        # change the maximum speed following X. Li et al., Neurocomputing 447 (2021) 64–79, https://doi.org/10.1016/j.neucom.2021.03.077
        self.mu = (  1. / ( 1.+self.alpha*np.exp(-self.beta*self.f) ) )
        
        for idim in range(0,self.number_of_dimensions):
            self.max_speed[idim] = self.mu*(self.search_interval[idim][1]-self.search_interval[idim][0])
            
        # store some history of the varying terms    
        self.history_c1[self.iteration_number] = self.c1
        self.history_c2[self.iteration_number] = self.c2
        self.history_w [self.iteration_number] = self.w
        self.history_mu[self.iteration_number] = self.mu
        self.history_evolutionary_state.append(self.evolutionary_state)
        
        print("\n",self.name,", f = ",self.f,"--> evolutionary state: ",dictionary_evolutionary_state_swarm[self.evolutionary_state],"; c1 = ",self.c1,"; c2 = ",self.c2,"; w = ",self.w,"; mu = ",self.mu,"\n")


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
        
        if ( (self.relocateParticlesWithRegression==True) and ( (self.iteration_number % self.number_of_iterations_between_regressions) == 0)):
            self.relocateParticlesUsingRegression()

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
        with open('history_mu.npy', 'wb') as f:
            np.save( f, self.history_mu)
            
    def APSOSavePartialHyperparametersHistory(self):
        with open('history_evolutionary_factor_f_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.history_f[0:self.iteration_number])
        with open('history_c1_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.history_c1[0:self.iteration_number])
        with open('history_c2_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.history_c2[0:self.iteration_number])
        with open('history_w_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.history_w[0:self.iteration_number])
        with open('history_mu_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.history_mu[0:self.iteration_number])
            
    def operationsAfterUpdateOfOptimumFunctionValueAndPosition(self):
        if (self.relocateParticlesWithRegression==True):
            # fit the regressor model to the data X,y 
            # IMPORTANT: X is normalized
            if self.iteration_number == 0: # if it is the first iteration
                # fit to the initial observation
                for isample in range(0,self.number_of_samples_per_iteration):
                    self.y[isample] = self.history_samples_positions_and_function_values[self.iteration_number,isample,self.number_of_dimensions]
                self.model.fit(self.X, self.y)
            else: # if it is one of the next iterations
                # add the new samples to the dataset
                for isample in range(0,self.number_of_samples_per_iteration):
                    sampled_positions = self.history_samples_positions_and_function_values[self.iteration_number,isample,0:self.number_of_dimensions]
                    normalized_sampled_positions = np.divide(sampled_positions,np.asarray(self.search_interval_size))
                    self.X = np.vstack((self.X, normalized_sampled_positions))
                    self.y = np.vstack((self.y, self.history_samples_positions_and_function_values[self.iteration_number,isample,self.number_of_dimensions]))
                # fit to the whole dataset including the new observations
                self.model.fit(self.X, self.y)
        else:
            pass
    
    def relocateParticlesUsingRegression(self):
        self.XsamplesChosenWithSurrogateModel = self.optimizeAcquisitionFunction(self.number_of_particles_from_regression)
        
        # Get the indices that would sort the array
        sorted_indices = np.argsort(self.history_samples_positions_and_function_values[self.iteration_number,:,self.number_of_dimensions])

        # Get the indices of the self.number_of_particles_from_regression lowest function values
        indices_worst_particles = sorted_indices[:self.number_of_particles_from_regression]
        
        # denormalize because XsamplesChosenWithSurrogateModel is normalized, but the positions are not
        index_particle_from_regression = 0
        for index_from_worst_particles in indices_worst_particles:
            for idim in range(0,self.number_of_dimensions):
                self.samples[index_from_worst_particles].position[idim] = self.XsamplesChosenWithSurrogateModel[index_particle_from_regression,idim]*self.search_interval_size[idim]
            index_particle_from_regression = index_particle_from_regression + 1
            print("Particle ",index_from_worst_particles," relocated with regression to position ",self.samples[index_from_worst_particles].position[:])
        
                     

class BayesianOptimization(Optimizer):
    def __init__(self, name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs):
        super().__init__(name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs)

        # Implementation of Bayesian Optimization based on a Matern Kernel (see https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html)
        ### Define the regression model with a Gaussian process
        self.number_of_tests,self.nu,self.length_scale,self.length_scale_bounds,self.xi,\
        self.kernel,self.model,self.XsamplesChosenWithSurrogateModel \
        = initializePredictiveModelForOptimization(number_of_samples_to_choose=self.number_of_samples_per_iteration,number_of_dimensions=self.number_of_dimensions,**kwargs)
        
        print("\n -- hyperparameters used by the optimizer -- ")
        print("number_of_tests                          = ",self.number_of_tests)
        print("nu                                       = ",self.nu)
        print("length_scale                             = ",self.length_scale)
        print("length_scale_bounds                      = ",self.length_scale_bounds)
        print("")
        
        
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
        
        print("\n Bayesian Optimization initialized")    
    
    def chooseNewPositionsToExplore(self):
        
        self.XsamplesChosenWithSurrogateModel = self.optimizeAcquisitionFunction(self.number_of_samples_per_iteration)
        # denormalize because XsamplesChosenWithSurrogateModel is normalized, but the positions are not
        for isample in range(self.number_of_samples_per_iteration):
            for idim in range(0,self.number_of_dimensions):
                self.samples[isample].position[idim] = self.XsamplesChosenWithSurrogateModel[isample,idim]*self.search_interval_size[idim] 
            
    def updateSamplesForExploration(self):
        # pick new samples
        self.chooseNewPositionsToExplore()
        self.iteration_number = self.iteration_number+1
    
    def operationsAfterUpdateOfOptimumFunctionValueAndPosition(self):
        # fit the regressor model to the data X,y 
        # IMPORTANT: X is normalized
        if self.iteration_number == 0: # if it is the first iteration
            # fit to the initial observation
            for isample in range(0,self.number_of_samples_per_iteration):
                self.y[isample] = self.history_samples_positions_and_function_values[self.iteration_number,isample,self.number_of_dimensions]
            self.model.fit(self.X, self.y)
        else: # if it is one of the next iterations
            # add the new samples to the dataset
            for isample in range(0,self.number_of_samples_per_iteration):
                self.X = np.vstack((self.X, self.XsamplesChosenWithSurrogateModel[isample]))
                self.y = np.vstack((self.y, self.history_samples_positions_and_function_values[self.iteration_number,isample,self.number_of_dimensions]))
            # fit to the whole dataset including the new observations
            self.model.fit(self.X, self.y)
            
# Structure for genetic algorithm - need to find good crossover and mutation operators       
# class GeneticAlgorithm(Optimizer):
#     def __init__(self, name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs):
#         super().__init__(name, number_of_samples_per_iteration, number_of_dimensions, search_interval, number_of_iterations, **kwargs)
# 
#         # Implementation of a basic genetic algorithm
#         # to avoid function re-evaluation, the samples stored by this optimizer 
#         # represent only the children generated at the present iteration.
#         # The set of children + parents is called population.
#         # At the first iteration, the population is made of the initial samples, 
#         # while in the other iterations the population is made of 
#         # population = children (which are number_of_samples_per_iteration) + the parents (which are number_of_parents)
#         # The parents are selected from the population of the previous iteration.
# 
#         if (self.number_of_samples_per_iteration < 2):
#             print("ERROR: for the Genetic Algorithm number_of_samples_per_iteration must be at least equal to 2.")
#             sys.exit()
# 
#         default_number_of_parents       = int(0.3*self.number_of_samples_per_iteration)
#         self.number_of_parents          = kwargs.get('number_of_parents', default_number_of_parents)
#         if (self.number_of_parents > self.number_of_samples_per_iteration):
#             print("ERROR: the number_of_parents must be smaller than the number_of_samples_per_iteration")
#             sys.exit()
#         if (self.number_of_parents < 2):
#             print("ERROR: the total number_of_parents must be at least 2")
#             sys.exit()
# 
#         # Compute the binomial coefficient C(number_of_parents, 2)
#         number_possible_children = math.comb(self.number_of_parents, 2)
#         if (self.number_of_samples_per_iteration<=number_possible_children):
#             print("ERROR: with number_of_parents parents, the number of children number_of_samples_per_iteration must be larger than ",number_possible_children,".")
#             sys.exit()
# 
#         default_probability_of_mutation = 0.1
#         self.probability_of_mutation    = kwargs.get('probability_of_mutation', default_probability_of_mutation)
#         if ( (self.probability_of_mutation < 0.) or (self.probability_of_mutation > 1.) ):
#             print("ERROR: probability_of_mutation must be a float between 0. and 1.")
#             sys.exit()
# 
#         print("number_of_parents                        = ",self.number_of_parents)
#         print("probability_of_mutation                  = ",self.probability_of_mutation)
#         print("")
# 
# 
#         # Lists containing the positions and function values of the population from which parents are selected.
#         # At the first iteration, the population from which parents are selected is made only of the initial samples (which are `number_of_samples_per_iteration`). 
#         # At the next iterations, the population from which parents are selected is made of the parents selected at the previous iteration + their children.
#         self.population_positions       = []
#         self.population_function_values = []
# 
#         # Initialize each sample
#         self.halton_sampler_position   = qmc.Halton(d=self.number_of_dimensions, scramble=True)
#         halton_sampler_random_position = self.halton_sampler_position.random(n=self.number_of_samples_per_iteration)
#         for isample in range(self.number_of_samples_per_iteration):
#             position = np.zeros(self.number_of_dimensions)
#             # use a Halton sequence to sample more uniformly the parameter space
#             for idim in range(0,self.number_of_dimensions):
#                 position[idim]       = search_interval[idim][0]+halton_sampler_random_position[isample][idim]*(search_interval[idim][1]-search_interval[idim][0]) 
# 
#             random_sample            = RandomSearchSample(position) 
#             self.samples.append(random_sample)
# 
#             # at the first iteration, the population is made just by the initial samples
#             self.population_positions.append(position)
# 
#             print("\n ---> Sample", isample, "Position:", position)  
#             self.history_samples_positions_and_function_values[0,isample,0:self.number_of_dimensions] = position[:]
# 
#         print("\n Genetic Algorithm initialized") 
# 
#     def updateSamplesForExploration(self):
# 
#         # Select parents, i.e. the number_of_parents positions among the present population 
#         # which obtained the highest values of the function to optimize
#         # remove the other positions from the population
#         self.addChildrenFunctionValuesToPopulationFunctionValues()
#         self.selectParents()
# 
#         # Generate new children
#         children_positions = self.crossover()
#         children_positions = self.mutateChildren(children_positions)
# 
#         # Build population fot the next iteration
#         # population = parents + new children
#         self.addChildrenToPopulation(children_positions)
# 
#         # The new samples of the optimizer will be the children that were just generated
#         self.addChildrenToSamples(children_positions)
# 
#         # update iteration number
#         self.iteration_number = self.iteration_number+1
# 
#     def addChildrenFunctionValuesToPopulationFunctionValues(self):
#         for isample in range(0,self.number_of_samples_per_iteration):
#             self.population_function_values.append(self.history_samples_positions_and_function_values[self.iteration_number,isample,self.number_of_dimensions])
# 
#     def selectParents(self):
#         # Create a temporary list of tuples (element, index)
#         temp_population_function_values = [(element, index) for index, element in enumerate(self.population_function_values)]
#         # Sort the temporary list based on the elements (descending order)
#         temp_population_function_values.sort(reverse=True)
#         # Get the first n indices from the sorted list
#         indices_parents = [index for _, index in temp_population_function_values[:self.number_of_parents]]
#         # Remove the samples which are not selected as parents from the population list
#         self.population_positions       = [elem for i, elem in enumerate(self.population_positions)       if i in indices_parents]
#         self.population_function_values = [elem for i, elem in enumerate(self.population_function_values) if i in indices_parents]
# 
# 
#     def crossover(self):
#         children_positions = []
# 
#         # for each child
#         for ichild in range(0,self.number_of_samples_per_iteration):
#             # draw a random couple of parents from the selected parents
# 
#             # index of parent 1
#             random_index_1     = random.randint(0, len(self.population_positions)-1)
#             random_index_2     = random_index_1
# 
#             while ( random_index_2 == random_index_1 ):
#                 # index of parent 2
#                 random_index_2 = random.randint(0, len(self.population_positions)-1)
# 
#             parent_1_position  = self.population_positions[random_index_1]
#             parent_2_position  = self.population_positions[random_index_2]
# 
#             position = np.zeros(self.number_of_dimensions)
#             for idim in range(0,self.number_of_dimensions):
#                 position[idim] = random.uniform( min(parent_1_position[idim],parent_2_position[idim]), max(parent_1_position[idim],parent_2_position[idim]) )
# 
#             children_positions.append(position)
# 
# 
#         return children_positions
# 
#     def mutateChildren(self,children_positions):
#         for child_position in children_positions:
#             for idim in range(0,self.number_of_dimensions):
#                 if (np.random.rand() < self.probability_of_mutation):
#                     child_position[idim] = np.random.uniform(self.search_interval[idim][0], self.search_interval[idim][1])
#         return children_positions
# 
#     def addChildrenToPopulation(self,children_positions):
#         print("Parents:")
#         for parent_position in self.population_positions:
#             print(" ",parent_position)
#         print("Children:")
#         for child_position in children_positions:
#             print(" ",child_position)
#             self.population_positions.append(child_position)
# 
#     def addChildrenToSamples(self,children_positions):
#         for isample in range(0,len(children_positions)):
#             self.samples[isample].position[:] = children_positions[isample][:]
            
     
        
        
        
        
                  
     