# Discoveri
## Data-driven Investigation through Simulations on Clusters for the Optimization of the physical Variables' Effects in Regimes of Interest

### About ``Discoveri``
``Discoveri`` is a Python code to optimize/maximize a function with derivative-free methods. This function can be a `numpy` function or the result of a postprocessing function of simulations on a cluster. In both cases the user can define the function to optimize. In the latter case, ``Discoveri`` prepares and launches automatically the simulations that sample the function to optimize. At the moment, the following optimization methods are implemented: `"Random Search"`,`"Bayesian Optimization"`, `"Particle Swarm Optimization"` (and two of its variants called `"IAPSO"` and `"PSO-TPME"`).

### Python libraries used
- `numpy`
- `matplotlib`
- `scipy`
- `sklearn`
- `pandas`
- `seaborn`
- `os`
- `sys`
- `time`
- `math`
- `random`
- `datetime`
If ``Discoveri`` is used to optimize the result of ``Smilei`` simulations, then also the postprocessing library `happi` will be necessary.

### Basic concepts and terminology used in ``Discoveri``
``Discoveri`` optimizes/maximizes the result of `f(X)`, where `f` is a real-valued function of an array `X` called position, with `num_dimensions` dimensions. The elements of `X` can vary continuously in the real numbers space. The whole search space from which `X` is drawn is called `search_interval`.

``Discoveri`` uses derivative-free (also called black-box) optimization methods, used to optimize a function `f` that is costly to evaluate and/or unknown. 
If the gradients of this function are known, probably there are more efficient methods to optimize it.

Once an optimization run of ``Discoveri`` is launched, at each of the `max_iterations` iterations the code will perform `num_samples` evaluations of the function specified by the user, where each sample is characterized by a different array `X`. If the function is a `numpy` function `f`, these evaluations will simply compute the value of `f(X)`. If the function is a function `f` that computes the result of postprocessing of a simulation, the code will automatically launch the required simulations, wait for their results and postprocess them. Each simulation will have `num_dimension` varying physical quantities stored in its own array `X`.

### Input file for ``Discoveri`` (to do)

#### General parameters (to do)

##### Optimization parameters
- `optimization_method`
- `num_samples`
- `num_dimensions`
- `search_interval`
- `max_iterations`
- `input_parameters_names`
- `iterations_between_outputs`
- `optimizer_hyperparameters`

##### Function to optimize
- `use_test_function`                  
- `test_function`  
- `simulation_postprocessing_function`

##### Job preparation and management  in a cluster
- `starting_directory`                     
- `home_directory`
- `path_executable`
- `path_input_namelist`
- `path_submission_script`
- `path_second_submission_script`
- `input_parameters_names`
- `name_input_namelist`
- `command_to_launch_jobs`
- `name_log_file_simulations`
- `word_marking_end_of_simulation_in_log_file`
- `time_to_wait_for_iteration_results`


#### Available optimization methods and their hyperparameters (to do)
- `"Random Search"`, `optimizer_hyperparameters = [use_Halton_sequence]`: in this optimization method, the array `X` for each sample of each iteration is generated pseudo-randomly, with a uniform distribution within the `search_interval`. If `use_Halton_sequence` is `True`, a scrambled Halton sequence (https://en.wikipedia.org/wiki/Halton_sequence) is used to draw the samples. Otherwise, they are drawn using `numpy.random.uniform` scaled and shifted for each dimension to draw `X` frim the `search_interval` of interest.
- `"Bayesian Optimization"`, `optimizer_hyperparameters = []`: based on the implementation of Gaussian process regression (GPR) provided by the `scikit-learn` library, with an anisotropic Matérn kernel with `nu=1.5`. The length scale of the Matérn kernel was optimized by maximizing the log-marginal-likelihood every time new data is fitted to the model. An expected improvement acquisition function is used.
- `"Particle Swarm Optimization"`, `optimizer_hyperparameters = [c1,c2,w,initial_velocity_over_search_space_size]`: the most common version of Particle Swarm Optimization (PSO), where an inertia term is added to the velocity update, as described in Y. Shi, R.C. Eberhart, 1998 https://ieeexplore.ieee.org/document/699146 . The `num_sample` samples represent the particles of the swarm, with position `X` and a velocity. At each iteration the velocity of the particles are updated adding a cognitive velocity component (proportional to a constant coefficient `c1`), a social velocity component (proportional to a constant coefficient `c2`, `c1+c1<4`) and the velocity at the last iteration multiplied by the constant inertia weight `w` (which must be `<1` to avoid velocity divergence). For each particle, high values of `c1` promote the exploration near the best position found by the particle, while high values of `c2` promote the exploitation of the knowledge of the best position found by the entire swarm. Both best positions are updated at each iteration if better positions are found. The particles initial positions are initialized with a scrambled Halton sequence and their velocities in each dimension are drawn from a uniform distribution proportional to `initial_velocity_over_search_space_size` in that dimension. Particles crossing a `search_interval` boundary in a dimension will have their coordinate in that dimension reinitialized, drawn from a uniform distribution in the `search_domain` span in that dimension, their velocity will remain unchanged.
- `"IAPSO"` (Inertia weight and Acceleration coefficients PSO), `optimizer_hyperparameters = [w1,w2,m,initial_velocity_over_search_space_size]`: version of PSO based on Wanli Yang et al 2021 J. Phys.: Conf. Ser. 1754 012195, https://iopscience.iop.org/article/10.1088/1742-6596/1754/1/012195
In this version of PSO the `c1` coefficient decreases from 2 to 0 with a `sin^2` function and the `c2` coefficient increases from 0 to 2 with a `sin^2` function (Eq. 10); the inertia weight decreases exponentially from `w1` (<1) to `w2` (<`w1`) (Eq. 7), with a speed controlled by the number `m` (the highest `m`, the quickest the decrease). Compared to that reference, the velocities are updated and then the positions, like in a PSO and not like in a SPSO, where the position is directly updated (hence the name used in the article IASPSO and the name IAPSO used here). The initial position, velocity of the particles and the boundary conditions are the same of the PSO.
- `"PSO-TPME"` (PSO with Targeted, Position-Mutated Elitism), `optimizer_hyperparameters = [c1,c2,w1,w2,initial_velocity_over_search_space_size,Nmax_iterations_bad_particles,portion_of_mean_classification_levels,amplitude_mutated_range_1,amplitude_mutated_range_2]`: version of the PSO based on T. Shaquarin, B. R. Noack, International Journal of Computational Intelligence Systems (2023) 16:6, https://doi.org/10.1007/s44196-023-00183-z In this version of PSO, the inertia linearly decreases from `w1` (<1) to `w2` (<`w1`) and the coefficients `c1` and `c2` (`c1+c2<4`) are fixed. At each iteration, the mean `mean` of the function values found by the particles is computed. Afterwards, two levels for the function value are defined: `mean*(1+portion_of_mean_classification_levels)` and `mean*(1-portion_of_mean_classification_levels)`, where `portion_of_mean_classification_levels<1`. Depending on the function value they have found compared to these two levels, at each iterations particles are classified as `good` (above the highest level), `bad` (below the lowest level) and `fair` (in between). `good` particles will behave only exploring around their personal optimum position i.e. as if `c2=0`), `bad` particles will behave only converging towards the swarm optimum position (i.e. as if `c1=0`). `fair` particles will behave as the particles of a "classic" PSO. Particles remaining `bad` for `Nmax_iterations_bad_particles` will be marked as `hopeless`, i.e. at the next iteration they will be relocated near the swarm optimum position.
Compared to that reference, the level from which the levels for `bad`, `fair`, `good` particles are computed cannot decrease over the iterations: i.e. the maximum between the mean of the function values found and the mean found at the previous iteration is used as `mean`;
furthermore, to reinitialize the particles closer to the optimum one, the mutated_amplitude scale is linearly decreasing from `amplitude_mutated_range_1` to `amplitude_mutated_range_2` (`<amplitude_mutated_range_1`) and the distribution of the coordinates in a dimension near the optimum particle is a gaussian proportional to the `search_space` size in that dimension and the mutated amplitude.
The initial position, velocity of the particles and the boundary conditions are the same of the PSO.