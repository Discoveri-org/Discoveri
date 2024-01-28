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

# this import is needed for the FST Particle Swarm Optimization
from FuzzyLogicUtilitiesForFSTPSO import *

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
        self.use_multiple_swarms             = False                            # use or not multiple swarms if Particle Swarm Optimization or FST-PSO are used
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
        
        # Boundary conditions for particles crossing the domain boundaries
        default_boundary_conditions   = "damping"
        self.boundary_conditions      = kwargs.get('boundary_conditions', default_boundary_conditions)
        
        if ( (self.boundary_conditions!="relocating") and (self.boundary_conditions!="damping") ):
            print("ERROR: boundary_conditions for ",self.name,"can be either 'relocating' or 'damping' " )
            sys.exit()
            
        
        print("\n -- hyperparameters used by the optimizer -- ")
        
        print("boundary_conditions                      = ",self.boundary_conditions )
        
        # Choose if using multiple (almost) independent swarms searching in parallel for the optimum
        default_use_multiple_swarms   = False
        self.use_multiple_swarms      = kwargs.get('use_multiple_swarms', default_use_multiple_swarms)
        print("use_multiple_swarms                      = ",self.use_multiple_swarms )
        
        # Choose if how to distribute the subswarms 
        # if "by_search_space_subdomains", each subswarm will be initially assigned to a different part of the domain
        # if "all_the_search_space", the subswarms will be distributed through the whole domain
        default_subswarms_distribution= "all_the_search_space"
        self.subswarms_distribution   = kwargs.get('subswarms_distribution', default_subswarms_distribution)
        if (self.use_multiple_swarms==True):
            print("subswarms_distribution                   = ",self.subswarms_distribution )
        
        default_subswarm_size         = self.number_of_samples_per_iteration
        self.number_of_subswarms      = 1
        if (self.use_multiple_swarms==True):
            self.subswarm_size        = kwargs.get('subswarm_size', default_subswarm_size)
            if (self.subswarm_size == self.number_of_samples_per_iteration):
                print("ERROR: if use_multiple_swarms=True, you have to choose a subswarm size < number_of_samples_per_iteration")
                sys.exit()
            if (self.number_of_samples_per_iteration%self.subswarm_size!=0):
                print("ERROR: subswarm_size must divide number_of_samples_per_iteration evenly")
                sys.exit()
            print("subswarm_size                            = ",self.subswarm_size )
            self.number_of_subswarms  = int(self.number_of_samples_per_iteration/self.subswarm_size)
            print("number_of_subswarms                      = ",self.number_of_subswarms )
        
        # If yes, the particles are randomly exchanged between subswarms every iterations_beween_subswarm_regrouping iterations
        # similarly to J.J. Liang, P.N. Suganthan, Proceedings 2005 IEEE Swarm Intelligence Symposium, 2005. SIS 2005,
        default_subswarm_regrouping   = False
        self.subswarm_regrouping      = kwargs.get('subswarm_regrouping', default_subswarm_regrouping)
        
        default_iterations_beween_subswarm_regrouping = 5
        self.iterations_beween_subswarm_regrouping    = kwargs.get('subswarm_regrouping', default_iterations_beween_subswarm_regrouping)
        
        if (self.name=="Particle Swarm Optimization"):

            # "classic" version of Particle Swarm Optimization, but using an inertia term to avoid divergence
            # as described in Y. Shi, R.C. Eberhart, 1998, https://ieeexplore.ieee.org/document/69914
            # the acceleration coefficients and the inertia weight are kept constant
            
            # w (inertia weight): It controls the impact of the particle's previous velocity on the current velocity update. 
            # A higher value of w emphasizes the influence of the particle's momentum, promoting exploration.
            # On the other hand, a lower value of w emphasizes the influence of the current optimum positions, promoting exploitation.
            default_value_w  = 0.8
            self.w           = kwargs.get('w', default_value_w)
            
            if ((self.w)>1.):
                print("ERROR: w must be < 1.")
                sys.exit()
            
            # The inertia will be decreased from w1 to w2.
            # If no value for w1 and w2 is provided, the inertia will remain constant
            default_value_w1 = self.w
            self.w1          = kwargs.get('w1', default_value_w1)
            
            default_value_w2 = self.w1
            self.w2          = kwargs.get('w2', default_value_w2)
            
            
            if ( (self.w1 > 1) or (self.w2 > 1) ):
                print("ERROR: w1 and w2 should be smaller than 1")
                sys.exit()
                
            if ( self.w2 > self.w1 ):
                print("ERROR: w2 should be smaller than w")
                sys.exit()
            
            print("max_speed                                = ",self.max_speed )
            print("c1                                       = ",self.c1)
            print("c2                                       = ",self.c2)
            print("w                                        = ",self.w)
            if (self.w2 != self.w):
                print("w1                                       = ",self.w1)
                print("w2                                       = ",self.w2)
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
            
            if (self.use_multiple_swarms==True):
                print("ERROR: multiple swarms are not supported with ",self.name)
                sys.exit()
            
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
            
            print("max_speed                                = ",self.max_speed                        )
            print("c1 (initial)                             = ",self.c1                               )
            print("c2 (initial)                             = ",self.c2                               )
            print("w  (initial)                             = ",self.w                                )
            print("perturbation_global_best_particle        = ",self.perturbation_global_best_particle)
            print("mu_min                                   = ",self.mu_min                           )
            print("mu_max                                   = ",self.mu_max                           )
            print("")
        
        elif (self.name=="FST-PSO"):
            # Version of Particle Swarm Optimization described in
            # M. Nobile et al., Swarm and Evolutionary Computation 39 (2018) 70–85
            # where all the parameters of the PSO, independently for each particle,
            # are adapted using fuzzy rules
            
            # Parameters for the crisp values of the fuzzy rules, taken from the paper
            
            # maximum normalized distance in the multidimensional space, i.e. the diagonal of the multidimensional cube of the normalized search_interval
            self.delta_max    = max(1.,np.sqrt(self.number_of_dimensions)) # 1 if number_of_dimensions=1, sqrt(number_of_dimensions) otherwise
            self.delta1       = 0.2 * self.delta_max
            self.delta2       = 0.4 * self.delta_max
            self.delta3       = 0.6 * self.delta_max

            # inertia w
            self.w_low        = 0.3
            self.w_medium     = 0.5
            self.w_high       = 1.0

            # acceleration coefficient c1
            self.c1_low       = 0.1
            self.c1_medium    = 1.5
            self.c1_high      = 3.0

            # acceleration coefficient c2
            self.c2_low       = 1.0
            self.c2_medium    = 2.0
            self.c2_high      = 3.0

            # Minimum absolute value of velocity
            self.L_low        = 0.
            self.L_medium     = 0.001
            self.L_high       = 0.01

            # Maximum absolute value of velocity
            self.U_low        = 0.1
            self.U_medium     = 0.15
            self.U_high       = 0.2                           
            
            # arrays to store the hyperparameters for each particle and each iteration
            self.Phi_FSTPSO   = np.zeros(shape=(self.number_of_samples_per_iteration,self.number_of_iterations))
            self.delta_FSTPSO = np.zeros(shape=(self.number_of_samples_per_iteration,self.number_of_iterations))
            self.w_FSTPSO     = np.zeros(shape=(self.number_of_samples_per_iteration,self.number_of_iterations))
            self.c1_FSTPSO    = np.zeros(shape=(self.number_of_samples_per_iteration,self.number_of_iterations))
            self.c2_FSTPSO    = np.zeros(shape=(self.number_of_samples_per_iteration,self.number_of_iterations))
            self.U_FSTPSO     = np.zeros(shape=(self.number_of_samples_per_iteration,self.number_of_iterations))
            self.L_FSTPSO     = np.zeros(shape=(self.number_of_samples_per_iteration,self.number_of_iterations))
            
            
        # use scrambled Halton sampler to extract initial positions
        self.halton_sampler_position                 = qmc.Halton(d=self.number_of_dimensions, scramble=True)
        self.halton_sampler_random_position          = self.halton_sampler_position.random(n=self.number_of_samples_per_iteration)

        # Initialize each particle the swarm
        for iparticle in range(0,self.number_of_samples_per_iteration):
            position = np.zeros(self.number_of_dimensions)
            velocity = np.zeros(self.number_of_dimensions)
            
            for idim in range(0,self.number_of_dimensions):
                # use a scrambled Halton sequence to sample more uniformly the parameter space
                position[idim] = self.search_interval[idim][0]+self.halton_sampler_random_position[iparticle][idim]*self.search_interval_size[idim] #np.random.uniform(search_interval[dimension][0], search_interval[dimension][1])
                # use initial velocity proportional to the search_space size in this dimension
                velocity[idim] = self.max_speed[idim]*np.random.uniform(-1, 1)
                particle       = SwarmParticle(position,velocity)
            
            self.samples.append(particle)
                
            print("Particle", iparticle, "Position:", position)  
            self.history_samples_positions_and_function_values[0,iparticle,0:self.number_of_dimensions] = position[:]
            self.samples[iparticle].optimum_position[:]                                                 = position[:]
            del position, velocity
            
        self.particles_indices_in_subswarm = []
        if (self.use_multiple_swarms==True):
            # first dimension: iteration number
            # second dimension: number of swarm
            # third dimension: the coordinates of the optimum position for that swarm at that iteration + the corresponding function value
            self.history_subswarm_optimum_position_and_optimum_function_values = np.zeros(shape=(self.number_of_iterations,self.number_of_subswarms,self.number_of_dimensions+1))
            print("\nAssigning the swarm particles to subswarms")
            if (self.subswarms_distribution=="all_the_search_space"):
                for iswarm in range(0,self.number_of_subswarms):
                    self.particles_indices_in_subswarm.append(np.asarray([iswarm*self.subswarm_size+iparticle for iparticle in range(0,self.subswarm_size) ]))
                    print("Particles in subswarm ",iswarm,": ",self.particles_indices_in_subswarm[iswarm])
            elif (self.subswarms_distribution=="search_space_subdomains"):
                for iswarm in range(0,self.number_of_subswarms):
                    self.particles_indices_in_subswarm.append(np.asarray([iswarm+iparticle*self.number_of_subswarms for iparticle in range(0,self.subswarm_size) ]))
                    print("Particles in subswarm ",iswarm,": ",self.particles_indices_in_subswarm[iswarm])
            
            
        print("\n"+self.name+" initialized")

            
    def updateParticlePositionAndVelocity(self):    
        
        if (self.name == "Particle Swarm Optimization"):
            self.w = self.w1-(self.w1-self.w2)/(self.number_of_iterations+1)*(self.iteration_number)
            print("Inertia weight w = ",self.w)
        
        for iparticle in range(0,self.number_of_samples_per_iteration):
            
            if (self.use_multiple_swarms == False ):
                # choose as global best position the best one from the entire swarm
                global_best_position = self.optimum_position
            elif (self.use_multiple_swarms == True ):
                # choose as global best position the best one from the particle's subswarm
                if (self.subswarms_distribution=="all_the_search_space"):
                    subswarm_index   = int(iparticle/self.subswarm_size)
                elif (self.subswarms_distribution=="search_space_subdomains"):    
                    subswarm_index   = int(iparticle%self.number_of_subswarms)
                global_best_position = self.history_subswarm_optimum_position_and_optimum_function_values[self.iteration_number,subswarm_index,0:self.number_of_dimensions]
                
            if (self.name != "FST-PSO"): # not using the Fuzzy Self Tuning PSO
                
                for idim in range(self.number_of_dimensions):
                    # extract two random numbers
                    r1                  = random.random()
                    r2                  = random.random()
                    # compute cognitive velocity, based on the individual particle's exploration
                    cognitive_velocity  = self.c1 * r1 * (self.samples[iparticle].optimum_position[idim] - self.samples[iparticle].position[idim])
                    # compute social velocity, based on the swarm exploration
                    social_velocity     = self.c2 * r2 * (global_best_position[idim] - self.samples[iparticle].position[idim])
                    # update velocity
                    self.samples[iparticle].velocity[idim] = self.w * self.samples[iparticle].velocity[idim] + cognitive_velocity + social_velocity    
                    # limit the velocity to the interval [-max_speed,max_speed]
                    self.samples[iparticle].velocity[idim] = np.clip(self.samples[iparticle].velocity[idim],-self.max_speed[idim],self.max_speed[idim])
                # Update individual particle position
                self.samples[iparticle].position += self.samples[iparticle].velocity
            
            else: # using the Fuzzy Self Tuning PSO, each parameter for the particles (w,c1,c2,L,u) is adaptively changed at each iteration
            
                for idim in range(self.number_of_dimensions):
                    # extract two random numbers
                    r1                  = random.random()
                    r2                  = random.random()
                    # compute cognitive velocity, based on the individual particle's exploration
                    cognitive_velocity  = self.c1_FSTPSO[iparticle,self.iteration_number] * r1 * (self.samples[iparticle].optimum_position[idim] - self.samples[iparticle].position[idim])
                    # compute social velocity, based on the swarm exploration
                    social_velocity     = self.c2_FSTPSO[iparticle,self.iteration_number] * r2 * (global_best_position[idim] - self.samples[iparticle].position[idim])
                    # update velocity
                    self.samples[iparticle].velocity[idim] = self.w_FSTPSO[iparticle,self.iteration_number] * self.samples[iparticle].velocity[idim] + cognitive_velocity + social_velocity    
                    # limit the absolute value of velocity to the interval [L,U]
                    min_absolute_value_velocity = self.L_FSTPSO[iparticle,self.iteration_number]*self.search_interval_size[idim]
                    max_absolute_value_velocity = self.U_FSTPSO[iparticle,self.iteration_number]*self.search_interval_size[idim]
                    self.samples[iparticle].velocity[idim] = np.sign(self.samples[iparticle].velocity[idim])*np.clip(np.abs(self.samples[iparticle].velocity[idim]),min_absolute_value_velocity,max_absolute_value_velocity)
                # Update individual particle position
                self.samples[iparticle].position += self.samples[iparticle].velocity
                            
            # Boundary condition on position:
            for idim in range(self.number_of_dimensions):
                if ((self.samples[iparticle].position[idim] < self.search_interval[idim][0]) or (self.samples[iparticle].position[idim] > self.search_interval[idim][1])):
                    if (self.boundary_conditions!="relocating"):
                        # if the particle exits the search_interval in one of its position coordinates,
                        # reassign that coordinate randomly within the search_interval boundaries;
                        # don't change the velocity
                        self.samples[iparticle].position[idim] = self.search_interval[idim][0]+np.random.uniform(0., 1.)*(self.search_interval[idim][1]-self.search_interval[idim][0])
                    elif (self.boundary_conditions!="damping"):
                        # if the particle exits the search_interval in one of its position coordinates,
                        # reassign that coordinate at the search_interval boundary that was crossed;
                        # the velocity is inverted and multiplied by a random number within [0,1)
                        self.samples[iparticle].velocity[idim] = -np.random.uniform(0., 1.)*self.samples[iparticle].velocity[idim]
                        if   (self.samples[iparticle].position[idim] < self.search_interval[idim][0]):
                            self.samples[iparticle].position[idim] = self.search_interval[idim][0]
                        elif (self.samples[iparticle].position[idim] > self.search_interval[idim][1]):
                            self.samples[iparticle].position[idim] = self.search_interval[idim][1]
                   
        if ((self.use_multiple_swarms == True) and (self.subswarm_regrouping==True) and (self.iteration_number!=0) and ((self.iteration_number+1)%self.iterations_beween_subswarm_regrouping==0)):
            print("Regrouping the subswarms")
            self.regroup_subswarms()
    
    def updateSamplesForExploration(self):
        if (self.name=="Adaptive Particle Swarm Optimization"):
            self.evaluateEvolutionStateAndAdaptHyperparameters()
        if (self.name=="FST-PSO"):
            self.FSTPSOAdaptationOfHyperparameters()
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
            
    def multiSwarmSaveHistoryBestPositionsAndBestFunctionValues(self):
        with open('history_subswarm_optimum_position_and_optimum_function_values.npy', 'wb') as f:
            np.save( f, self.history_subswarm_optimum_position_and_optimum_function_values)
            
    def multiSwarmSavePartialHistoryBestPositionsAndBestFunctionValues(self):
        with open('history_subswarm_optimum_position_and_optimum_function_values_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.history_subswarm_optimum_position_and_optimum_function_values[0:self.iteration_number,:,:])
    
    def computePhiFSTPSO(self,iparticle):
        # normalized improvement as described in M. Nobile et al., Swarm and Evolutionary Computation 39 (2018) 70–85
        # updated to follow the definition of version FST-PSO2b of their library
        
        # compute distance_factor
        present_position                                   = self.samples[iparticle].position
        past_position                                      = self.history_samples_positions_and_function_values[self.iteration_number-1,iparticle,0:self.number_of_dimensions]
        normalized_distance_present_past_position          = normalized_euclidean_distance(present_position,past_position,self.search_interval_size)
        distance_factor                                    = normalized_distance_present_past_position/self.delta_max
        
        # compute improvement_factor
        function_improvement_factor = 0.
        present_function_value      = min(-self.history_samples_positions_and_function_values[self.iteration_number,iparticle,self.number_of_dimensions],-self.FSTPSO_worst_function_value)
        past_function_value         = 0.
        if (self.iteration_number == 0):
            past_function_value     = -self.FSTPSO_worst_function_value
        else:
            past_function_value     = min(-self.history_samples_positions_and_function_values[self.iteration_number-1,iparticle,self.number_of_dimensions],-self.FSTPSO_worst_function_value) 
        
        if ( abs(self.optimum_function_value-self.FSTPSO_worst_function_value) == 0 ):
            print("ERROR: division by zero in Phi calculation.")
            print(" This can only happen if the function value of all the particles at the first iteration is zero")
            sys.exit()
        else:
            function_improvement_factor = (present_function_value-past_function_value)/abs(self.optimum_function_value-self.FSTPSO_worst_function_value)
                                                                                                        
        return distance_factor*function_improvement_factor
        
    def FSTPSOAdaptationOfHyperparameters(self):
        for iparticle in range(0,self.number_of_samples_per_iteration):
            
            if (self.use_multiple_swarms == False ):
                # choose as global best position the best one from the entire swarm
                global_best_position = self.optimum_position
            else:
                # choose as global best position the best one from the particle's subswarm
                if (self.subswarms_distribution=="all_the_search_space"):
                    subswarm_index   = int(iparticle/self.subswarm_size)
                elif (self.subswarms_distribution=="search_space_subdomains"):    
                    subswarm_index   = int(iparticle%self.number_of_subswarms)
                global_best_position = self.history_subswarm_optimum_position_and_optimum_function_values[self.iteration_number,subswarm_index,0:self.number_of_dimensions]
                
                
            # normalized distance between the particle and the optimum
            self.delta_FSTPSO[iparticle,self.iteration_number] = normalized_euclidean_distance(self.samples[iparticle].position,global_best_position,self.search_interval_size)
                                                           
            self.Phi_FSTPSO[iparticle,self.iteration_number]   = self.computePhiFSTPSO(iparticle)
                                                                                   
            self.w_FSTPSO[iparticle,self.iteration_number],  \
            self.c1_FSTPSO[iparticle,self.iteration_number], \
            self.c2_FSTPSO[iparticle,self.iteration_number], \
            self.L_FSTPSO[iparticle,self.iteration_number],  \
            self.U_FSTPSO[iparticle,self.iteration_number]   \
            = self.FSTPSO_adapt_hyperparameters( \
                                            self.delta_FSTPSO[iparticle,self.iteration_number],\
                                            self.Phi_FSTPSO[iparticle,self.iteration_number]\
                                               )
        # print all the adapted Parameters
        print("FSTPSO parameters adapted.")
        print("- delta  : ")
        print(self.delta_FSTPSO[:,self.iteration_number])
        print("- Phi  : ")
        print(self.Phi_FSTPSO[:,self.iteration_number])
        print("- w  : ")
        print(self.w_FSTPSO[:,self.iteration_number])
        print("- c1 : ")
        print(self.c1_FSTPSO[:,self.iteration_number])
        print("- c2 : ")
        print(self.c2_FSTPSO[:,self.iteration_number])
        print("- L  : ")
        print(self.L_FSTPSO[:,self.iteration_number])
        print("- U  : ")
        print(self.U_FSTPSO[:,self.iteration_number])
        print("")
        
    def FSTPSOSavePartialHyperparametersHistory(self):    
        # save all the adapted parameters
        with open('history_FSTPSO_Phi_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.Phi_FSTPSO[:,0:self.iteration_number])
        with open('history_FSTPSO_delta_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.delta_FSTPSO[:,0:self.iteration_number])
        with open('history_FSTPSO_w_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.w_FSTPSO[:,0:self.iteration_number])
        with open('history_FSTPSO_c1_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.c1_FSTPSO[:,0:self.iteration_number])
        with open('history_FSTPSO_c2_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.c2_FSTPSO[:,0:self.iteration_number])
        with open('history_FSTPSO_L_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.L_FSTPSO[:,0:self.iteration_number])
        with open('history_FSTPSO_U_up_to_iteration_'+str(self.iteration_number).zfill(5)+'.npy', 'wb') as f:
            np.save( f, self.U_FSTPSO[:,0:self.iteration_number])
            
    def FSTPSO_adapt_hyperparameters(self,delta,phi):
        # Dictionaries for the membership functions 
        phi_membership   = {                                                                              \
                            'worse'  : membership_phi_worse(phi),                                         \
                            'same'  : membership_phi_same  (phi),                                         \
                            'better': membership_phi_better(phi)                                          \
                           }
                           
                           
        delta_membership = {                                                                              \
                            'same': membership_delta_same(delta, self.delta1, self.delta2),               \
                            'near': membership_delta_near(delta, self.delta1, self.delta2, self.delta3),  \
                            'far' : membership_delta_far (delta, self.delta2, self.delta3, self.delta_max)\
                            }
                            
        # inertia w      
        weight_low           = phi_membership  ['worse' ] + delta_membership['same']
        weight_medium        = phi_membership  ['same'  ] + delta_membership['near']
        weight_high          = phi_membership  ['better'] + delta_membership['far' ]
        
        w  = (weight_low * self.w_low + weight_medium * self.w_medium + weight_high * self.w_high) / (weight_low + weight_medium + weight_high)
        
        # acceleration coefficient c1
        weight_low           = delta_membership['far'   ]
        weight_medium        = phi_membership  ['worse' ] + phi_membership  ['same'] + delta_membership['same'] + delta_membership['near']
        weight_high          = phi_membership  ['better']
        c1 = (weight_low * self.c1_low + weight_medium * self.c1_medium + weight_high * self.c1_high) / (weight_low + weight_medium + weight_high)
        
        # acceleration coefficient c2
        weight_low           = phi_membership  ['better'] + delta_membership['near']
        weight_medium        = phi_membership  ['same'  ] + delta_membership['same']
        weight_high          = phi_membership  ['worse' ] + delta_membership['far' ]
        c2 = (weight_low * self.c2_low + weight_medium * self.c2_medium + weight_high * self.c2_high) / (weight_low + weight_medium + weight_high)

        # minimum absolute value of velocity L
        weight_low           = phi_membership  ['same'  ] + phi_membership['better'] + delta_membership['far']
        weight_medium        = delta_membership['same'  ] + delta_membership['near']
        weight_high          = phi_membership  ['worse' ]
        L  = (weight_low * self.L_low + weight_medium * self.L_medium + weight_high * self.L_high) / (weight_low + weight_medium + weight_high)
        
        # maximum absolute value of velocity U 
        weight_low           = delta_membership['same'  ]
        weight_medium        = phi_membership  ['same'  ] + phi_membership['better'] + delta_membership['near']
        weight_high          = phi_membership  ['worse' ] +  delta_membership['far']
        U  = (weight_low * self.U_low + weight_medium * self.U_medium + weight_high * self.U_high) / (weight_low + weight_medium + weight_high)    
        
        return w,c1,c2,L,U
        
    def FSTPSOLastEvaluationAndDumpHyperparameters(self):
        self.FSTPSOAdaptationOfHyperparameters()
        # save all the adapted parameters
        with open('history_FSTPSO_Phi.npy', 'wb') as f:
            np.save( f, self.Phi_FSTPSO)
        with open('history_FSTPSO_delta.npy', 'wb') as f:
            np.save( f, self.delta_FSTPSO)
        with open('history_FSTPSO_w.npy', 'wb') as f:
            np.save( f, self.w_FSTPSO)
        with open('history_FSTPSO_c1.npy', 'wb') as f:
            np.save( f, self.c1_FSTPSO)
        with open('history_FSTPSO_c2.npy', 'wb') as f:
            np.save( f, self.c2_FSTPSO)
        with open('history_FSTPSO_L.npy', 'wb') as f:
            np.save( f, self.L_FSTPSO)
        with open('history_FSTPSO_U.npy', 'wb') as f:
            np.save( f, self.U_FSTPSO)
    
    def regroup_subswarms(self):
        # Similarly to J.J. Liang, P.N. Suganthan, Proceedings 2005 IEEE Swarm Intelligence Symposium, 2005. SIS 2005,
        # the particles are redistributed randomly among the swarms periodically.
        # Here their new particle index is simply obtained by shuffling the particle indices
        # and then copying in each slot the particle corresponding to that index in the previous configuration
        
        
        # obtain an array with the new index that each particle will have
        old_particle_indices = list(range(self.number_of_samples_per_iteration))
        random.shuffle(old_particle_indices)
        shuffled_particle_indices = old_particle_indices
        
        # copy the positions of the particles
        temporary_copy_particles_positions = []
        for iparticle in range(0,self.number_of_samples_per_iteration):
            temporary_copy_particles_positions.append(self.samples[iparticle].position[:])
            
        # for each index in the array of particles
        for iparticle in range(0,self.number_of_samples_per_iteration):
            print("Position of particle ",iparticle," substituted by the position of particle ",shuffled_particle_indices[iparticle])
            
            # overwrite the particle position with the one of the corresponding shuffled index
            self.samples[iparticle].position[:] = temporary_copy_particles_positions[shuffled_particle_indices[iparticle]]
            
            # reset the velocity
            for idim in range(0,self.number_of_dimensions):
                self.samples[iparticle].velocity[idim] = random.uniform(-1, 1)*self.search_interval_size[idim]
                
                # limit the absolute value of velocity
                if (self.name != "FST-PSO"):
                    self.samples[iparticle].velocity[idim] = np.clip(self.samples[iparticle].velocity[idim],-self.max_speed[idim],self.max_speed[idim])
                else:
                    # limit the absolute value of velocity to the interval [L,U]
                    min_absolute_value_velocity = self.L_FSTPSO[iparticle,self.iteration_number]*self.search_interval_size[idim]
                    max_absolute_value_velocity = self.U_FSTPSO[iparticle,self.iteration_number]*self.search_interval_size[idim]
                    self.samples[iparticle].velocity[idim] = np.sign(self.samples[iparticle].velocity[idim])*np.clip(np.abs(self.samples[iparticle].velocity[idim]),min_absolute_value_velocity,max_absolute_value_velocity)
    
    def operationsAfterUpdateOfOptimumFunctionValueAndPosition(self):
        
        if (self.use_multiple_swarms==True):
            if (self.iteration_number==0):
                for iswarm in range(0,self.number_of_subswarms):
                    # find best particle in subswarm
                    index_best_particle_in_subswarm = \
                    np.argmax(self.history_samples_positions_and_function_values[0,self.particles_indices_in_subswarm[iswarm],self.number_of_dimensions])
                    # find corresponding index in global swarm
                    index_best_particle_in_swarm = self.particles_indices_in_subswarm[iswarm][index_best_particle_in_subswarm]
                    # store its position and function value as best ones in the subswarm
                    self.history_subswarm_optimum_position_and_optimum_function_values[0,iswarm,:] \
                    = self.history_samples_positions_and_function_values[0,index_best_particle_in_swarm,:]
            else: # store only if the subswarm best value is better than the previous one
                for iswarm in range(0,self.number_of_subswarms):
                    # find best particle in subswarm
                    index_best_particle_in_subswarm = \
                    np.argmax(self.history_samples_positions_and_function_values[self.iteration_number,self.particles_indices_in_subswarm[iswarm],self.number_of_dimensions])
                    # find corresponding index in global swarm
                    index_best_particle_in_swarm = self.particles_indices_in_subswarm[iswarm][index_best_particle_in_subswarm]
                    # store its position and function value as best ones in the subswarm, only if the subswarm best value is better than the previous one
                    if (self.history_samples_positions_and_function_values[self.iteration_number,index_best_particle_in_swarm,self.number_of_dimensions] >self.history_subswarm_optimum_position_and_optimum_function_values[self.iteration_number-1,iswarm,self.number_of_dimensions]):
                        self.history_subswarm_optimum_position_and_optimum_function_values[self.iteration_number,iswarm,:] = self.history_samples_positions_and_function_values[self.iteration_number,index_best_particle_in_swarm,:]
                    else:
                        self.history_subswarm_optimum_position_and_optimum_function_values[self.iteration_number,iswarm,:] = self.history_subswarm_optimum_position_and_optimum_function_values[self.iteration_number-1,iswarm,:]
                #sys.exit()
            print("Optimum positions and optimum function values of the subswarms:")
            print(self.history_subswarm_optimum_position_and_optimum_function_values[self.iteration_number,:,:])
            print("")
        
        # compute the worst function value for FSTO
        if (self.name == "FST-PSO"):
            if (self.iteration_number==0):
                self.FSTPSO_worst_function_value = np.amin(self.history_samples_positions_and_function_values[self.iteration_number,:,self.number_of_dimensions])
            else:
                present_worst_function_value     = np.amin(self.history_samples_positions_and_function_values[self.iteration_number,:,self.number_of_dimensions])
                self.FSTPSO_worst_function_value = min(self.FSTPSO_worst_function_value,present_worst_function_value)


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
            
     
        
        
        
        
                  
     