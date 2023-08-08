# :Discoveri
## Data-driven Investigation through Simulations on Clusters for the Optimization of the physical Variables' Effects in Regimes of Interest

### About ``:Discoveri``

``:Discoveri`` is a Python code to optimize/maximize a function with derivative-free methods. This function can be a `numpy` function or the result of a postprocessing function of simulations on a cluster. In both cases the user can define the function to optimize. In the latter case, ``:Discoveri`` prepares and launches automatically the simulations that sample the function to optimize. At the moment, the following optimization methods are implemented: `"Grid Search"`, `"Random Search"`,`"Bayesian Optimization"`, `"Genetic Algorithm"`, `"Particle Swarm Optimization"` (and two of its variants called `"Adaptive Particle Swarm Optimization"` and `"PSO-TPME"`).

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

### Basic concepts and terminology used in ``:Discoveri``

``Discoveri`` optimizes/maximizes the result of `f(X)`, where `f` is a real-valued function of an array `X` called position, with `number_of_dimensions` dimensions. The elements of `X` can vary continuously in the real numbers space. The whole search space from which `X` is drawn is called `search_interval`.

``Discoveri`` uses derivative-free (also called black-box) optimization methods, used to optimize a function `f` that is costly to evaluate and/or unknown. 
If the gradients of this function are known, probably there are more efficient methods to optimize it.

Once an optimization run of ``Discoveri`` is launched, at each of the `number_of_iterations` iterations the code will perform `number_of_samples_per_iteration` evaluations of the function specified by the user, where each sample is characterized by a different array `X`. If the function is a `numpy` function `f`, these evaluations will simply compute the value of `f(X)`. If the function is a function `f` that computes the result of postprocessing of a simulation, the code will automatically launch the required simulations, wait for their results and postprocess them. Each simulation will have `num_dimension` varying physical quantities stored in its own array `X`.

### Input file for ``:Discoveri``

An input file for ``:Discoveri`` needs to import the library, and set the general optimization parameters, e.g. `number_of_iterations`, the `search_interval`, etc. If you use ``:Discoveri`` to optimize the result of simulations, you will need to provide the paths of your executable, the command to launch jobs on your cluster etc., i.e. all the information needed to prepare, launch and postprocess the simulation jobs.

Afterwards, an `optimizationRun` object is initialized. The method `execute()` of this object will perform the optimization. Optionally, at the end of the run you can easily dump all the optimization run data in a `pickle` file.

See the `example` folder for some examples of optimizations using ``:Discoveri``.

The next two subsections will detail the parameters that can be provided to the initializer of the `optimizationRun`. Many of these parameters, e.g. the hyperparameters of the various optimization methods, have also default values, thus they don't always have to be provided.


##### General optimization parameters

- `optimization_method`(string): the derivative-free optimization techniques that will try to maximise `f(X)`.
At the moment the available options are:
  - `"Grid Search"`
  - `"Random Search"`
  - `"Bayesian Optimization"`
  - `"Genetic Algorithm"`
  - `"Particle Swarm Optimization"`
  - `"Adaptive Particle Swarm Optimization"`
  - `"PSO-TPME"`
- `number_of_iterations`(integer): the number of iterations of the optimization process.
- `number_of_samples_per_iteration`(integer): the number of samples chosen/drawn at each iteration, each evaluating `f(X)` at a different sample `X`.
- `number_of_dimensions` (integer): number of dimensions of the parameter space where the maximum of `f(X)` is searched.
- `search_interval`(list of `number_of_dimensions` lists of 2 elements): in this list, the inferior and superior boundaries of the parameter space to explore (where the different `X` will always belong). The boundaries for each dimension of the explored parameter space must be provided. 
- `iterations_between_outputs`(integer): number of iterations between outputs, i.e. the output files dump and some message prints at screen.

##### Function to optimize

- `use_test_function`: if `True`, the function `f(X)` to optimize, i.e. maximize, will be `test_function`. Otherwise, it will be `simulation_postprocessing_function`. 
In both cases, the users must ensure that the function does not return `nan`,`-inf`,`inf`, and that a real result is always obtained.
- `test_function`: a real-valued `numpy` function of the position `X`.
- `simulation_postprocessing_function`: a real-valued function that returns the result of a postprocessing (defined by the users) of a simulation, e.g. the average energy of tracked particles of a Smilei simulation. ``:Discoveri`` will prepare the simulation directories, launch the simulations corresponding to the sampled `X` positions and postprocess the results. The users must ensure that the namelist of the used code can be modified to use the parameters in `X` as inputs. For more details, see the next section and the examples folder. 

##### Job preparation and management  in a cluster 

The users must ensure that these parameters are coherent. e.g. the template job submission script must set the correct name for the simulation log files, etc.
- `input_parameters_names` (list of `number_of_dimensions` strings): important to modify the namelist to launch simulations. Currently, ``:Discoveri`` assumes that a Python namelist is used by the code, where after a line containing `#External_config` a dictionary will be created by ``:Discoveri``, containing the names of the parameters to explore and their values. The namelist of the code must be prepared in order to use this dictionary.
- `starting_directory`(string):                    
- `home_directory`(string):
- `path_executable`(string): the path of the executable.
- `path_input_namelist`(string): the name of the namelist template to launch simulations. This template will be copied and modified by ``:Discoveri`` in each simulation folder to include the dictionary with the `input_parameters_names` and the elements of `X`.
- `path_submission_script`(string): path of the submission script for the job manager system where the simulations will be launched. 
- `path_second_submission_script`(string): path of the second submission script for the job manager system where the simulations will be launched (f needed).
- `name_input_namelist`(string): name of the file for the input namelist.
- `command_to_launch_jobs` (string): the command used to launch jobs in the job managing system where the simulations will be launched, e.g. `sbatch submission_scipt.sh` for SLURM. The users must ensure that all the required libraries are charged in the submission script and that this file has all the permissions to be used.
- `name_log_file_simulations` (string): name of the log file produced by the simulations.
- `word_marking_end_of_simulation_in_log_file`(string): a word that if present in the `name_log_file_simulations` of a simulation will tell ``:Discoveri`` that the simulation has ended and can be postprocessed.
- `time_to_wait_for_iteration_results` (float): after launching `number_of_samples_per_iteration` simulations at each iteration, ``:Discoveri`` will wait this time in seconds to check if at least one simulation has ended. The simulations that have ended are postprocessed, evaluating the corresponding `f(X)`. If some simulations are still running, ``:Discoveri`` will wait again the same amount of time and check again. This process continues until all the `number_of_samples_per_iteration` simulations have ended and the next iteration can start after all of them are postprocessed. 


#### Available optimization methods and their hyperparameters (to do)
Following are the optimization techniques currently supported by ``:Discoveri``, as well as their hyperparameters. 

- `"Grid Search"`: the arrays `X` of the `number_of_samples_per_iteration` samples are evenly distributed in each dimension `idim` between `search_interval[idim][0]` and `search_interval[idim][1]`, with `samples_per_dimension[idim]` samples.
NOTE: this optimizer only supports `number_of_iterations = 1`.
Optimizer hyperparameters:
  - `samples_per_dimension` a list of `number_of_dimensions` integers, whose product must be equal to `number_of_samples_per_iteration`.
  
- `"Random Search"`: in this optimization method, the array `X` for each sample of each iteration is generated pseudo-randomly, with a uniform distribution within the `search_interval`. 
Optimizer hyperparameter:
  - `use_Halton_sequence` (default value = `True`): if `True`, a scrambled Halton sequence (https://en.wikipedia.org/wiki/Halton_sequence) is used to draw the samples. Otherwise, they are drawn using `numpy.random.uniform` scaled and shifted for each dimension to draw `X` from the `search_interval` of interest.
  
- `"Bayesian Optimization"`: based on the implementation of Gaussian process regression (GPR) provided by the `scikit-learn` library, with an anisotropic Matérn kernel with `nu=1.5`. The length scale of the Matérn kernel was optimized by maximizing the log-marginal-likelihood every time new data is fitted to the model. An expected improvement acquisition function is used. Sometimes convergence can be improved by limiting the orders of magnitude spanned by the function to optimize `f(X)`, e.g. using a `log(f(X))`, instead of `f(X)`. The best technique to limit the orders of magnitude will depend on the considered case.

- `"Genetic Algorithm"`: a genetic algorithm where at each iteration `number_of_parents` biological parents are selected from a population (those with highest `f(X)` are picked). For each child to generate (corresponding to `number_of_samples_per_iteration`), two biological parents are randomly paired and the position `X` of their child is the arithmetic average of the components of their positions `X` (arithmetic crossover). At the moment this is the only crossover method implemented in ``:Discoveri``. For each child, for each dimension, there is a probability `probability_of_mutation` that the corresponding position coordinate is mutated, randomly selected within the `search_interval`. At the first iteration, the population from which parents are selected is made only of the initial samples (which are `number_of_samples_per_iteration`). At the next iterations, the population from which parents are selected is made of the parents selected at the previous iteration and their children.
Optimizer hyperparameters:
  - `probability_of_mutation` (default value = `0.1`): the probability of mutation for each dimension of each child at each iteration.
  - `number_of_parents` (default value = `int(0.3*number_of_samples_per_iteration)`): the number of parents selected from the population at each iteration to generate children. The number of children will always be equal to `number_of_samples_per_iteration`. The `number_of_parents` must be at least 2 (although at least 3 is recommended to improve the genetic diversity of their children).
  
- `"Particle Swarm Optimization"`: the most common version of Particle Swarm Optimization (PSO), where an inertia term is added to the velocity update, as described in Y. Shi, R.C. Eberhart, 1998 https://ieeexplore.ieee.org/document/699146 . The `num_sample` samples represent the particles of the swarm, with position `X` and a velocity. At each iteration `t` the velocity of the particles are updated adding a cognitive velocity component (proportional to a constant coefficient `c1`), a social velocity component (proportional to a constant coefficient `c2`, `c1+c1<4`) and the velocity at the last iteration multiplied by the constant inertia weight `w` (which must be `<1` to avoid velocity divergence). At each iteration, for each dimension `k`, the velocity component `V_k` and the position component `X_k` of each particle `i` at iteration `t+1` will be updated at each iteration using 
`V_k(t+1)=w*V_k(t)+c1*rand1*[Best_X_i_k-X_k(t)]+c2*rand2*[Best_X_swarm_k-X_k(t)]` 
and 
`X_k(t+1)=X_k(t)+V_k(t+1)`,
where `rand1` and `rand2` are two random numbers from between `0` and `1`, and `Best_X_i_k` and `Best_X_swarm` are the best positions `X` found so far respectively by the particle `i` and by the entire swarm.
The optimum positions `Best_X_i` are updated if better positions are found the particles `i`. Coherently, if one of these positions is better than the best one found by the swarm so far,  `Best_X_swarm` is updated. The particles initial positions are initialized with a scrambled Halton sequence. Particles crossing a `search_interval` boundary in a dimension will have their coordinate in that dimension reinitialized, drawn from a uniform distribution in the `search_domain` span in that dimension, their velocity will remain unchanged.
Optimizer hyperparameters:
  - `c1` (cognitive acceleration coefficient, default value = `2.`): high values of `c1` promote the exploitation of positions near the best position found by the individual particle, updated at each iteration if a better position is found. If provided by the users, they must ensure that `c1+c1<4`.
  - `c2` (social acceleration coefficient, default value = `2.`): high values of `c2` promote the exploitation of positions near the best position found by the entire swarm, updated at each iteration if a better position is found. If provided by the users, they must ensure that `c1+c1<4`.
  - `w` (inertia weight, default value = `0.9`): high values of this coefficient reduce the variations of the velocity of the particle. If provided by the users, they must ensure that it is `<1` to avoid velocity divergence.
  - `initial_speed_over_search_space_size` (default value = `0.1`): the initial velocities of particles in each dimension are drawn from a uniform distribution with boundaries proportional to `initial_speed_over_search_space_size` in that dimension and to the `search_interval` size in that dimension.
  - `max_speed` (maximum speed, default value: an array with elements equal to the respective size of `search_interval` in each dimension multiplied by `0.3`).
  
- `"Adaptive Particle Swarm Optimization"` (Adaptive PSO): based on from Z.-H. Zhan et al., IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) 39, 6 (2009) https://ieeexplore.ieee.org/document/4812104 .
Based on the evolutionary state of the swarm, the coefficients `c1`, `c2` and the inertia weight `w` are updated as described in that article. Compared to the description in the original reference, no transition base rule is used.
Optimizer hyperparameters:
  - `c1` (cognitive acceleration coefficient, default value = `2.0`): same as in the `"Particle Swarm Optimization"`, but the value provided by the user is just the initial value of the coefficient.
  - `c2` (social acceleration coefficient, default value = `2.0`): same as in the `"Particle Swarm Optimization"`, but the value provided by the user is just the initial value of the coefficient.
  - `w` (inertia weight, default value = `0.9`): same as in the `"Particle Swarm Optimization"`, but the value provided by the user is just the initial value of the inertia weight.
  - `perturbation_global_best_particle` (default value = `True`): when `True` and the evolutionary state of the swarm is `Convergence`, the position of the best particle (the one with highest `f(X)`) is mutated adding a random perturbation drawn from a Gaussian distribution on one random coordinate (see original reference) and this new position is assigned to the worst particle of the swarm (the one with lowest `f(X)`). This feature allows the swarm to jump out of a local maximum more easily.
  - `initial_speed_over_search_space_size`: same as in the `"Particle Swarm Optimization"`.
  - `max_speed`: same as in the `"Particle Swarm Optimization"`.
  
- `"PSO-TPME"` (PSO with Targeted, Position-Mutated Elitism), `optimizer_hyperparameters = [c1,c2,w1,w2,initial_speed_over_search_space_size,Number_of_iterations_bad_particles,portion_of_mean_classification_levels,amplitude_mutated_range_1,amplitude_mutated_range_2]`: version of the PSO based on T. Shaquarin, B. R. Noack, International Journal of Computational Intelligence Systems (2023) 16:6, https://doi.org/10.1007/s44196-023-00183-z In this version of PSO, the inertia linearly decreases from `w1` (<1) to `w2` (<`w1`) and the coefficients `c1` and `c2` (`c1+c2<4`) are fixed. At each iteration, the mean `mean` of the function values found by the particles is computed. Afterwards, two levels for the function value are defined: `mean*(1+portion_of_mean_classification_levels)` and `mean*(1-portion_of_mean_classification_levels)`, where `portion_of_mean_classification_levels<1`. Depending on the function value they have found compared to these two levels, at each iterations particles are classified as `good` (above the highest level), `bad` (below the lowest level) and `fair` (in between). `good` particles will behave only exploring around their personal optimum position i.e. as if `c2=0`), `bad` particles will behave only converging towards the swarm optimum position (i.e. as if `c1=0`). `fair` particles will behave as the particles of a "classic" PSO. Particles remaining `bad` for `Number_of_iterations_bad_particles` will be marked as `hopeless`, i.e. at the next iteration they will be relocated near the swarm optimum position.
Compared to that reference, the level from which the levels for `bad`, `fair`, `good` particles are computed cannot decrease over the iterations: i.e. the maximum between the mean of the function values found and the mean found at the previous iteration is used as `mean`;
furthermore, to reinitialize the particles closer to the optimum one, the mutated_amplitude scale is linearly decreasing from `amplitude_mutated_range_1` to `amplitude_mutated_range_2` (`<amplitude_mutated_range_1`) and the distribution of the coordinates in a dimension near the optimum particle is a gaussian proportional to the `search_space` size in that dimension and the mutated amplitude.
The initial position, velocity of the particles and the boundary conditions are the same of the PSO.
Optimizer hyperparameters:
  - `w1` (initial inertia weight, default value = `0.9`).
  - `w2` (initial inertia weight, default value = `0.4`).
  - `portion_of_mean_classification_levels` (default value = `0.02`).
  - `amplitude_mutated_range_1` (default value = `0.4`).
  - `amplitude_mutated_range_2` (default value = `0.01`).
  - `Number_of_iterations_bad_particles` (default value = `2`).
  - `initial_speed_over_search_space_size`: same as for the `"Particle Swarm Optimization"`.
  - `max_speed`: same as in the `"Particle Swarm Optimization"`.
  - `c1`: same as for the `"Particle Swarm Optimization"`.
  - `c2`: same as for the `"Particle Swarm Optimization"`.
  
#### Postprocessing:
In the folder `postprocessing_scripts` several scripts are available to have an insight on the optimization run(s) made with ``:Discoveri``:
- `readAndPlotOptimizationHistory`: to be used inside the folder of an optimization run, the script tells the best position `X` found by the run, the corresponding value of the function and the corresponding id of the function evaluation. This is particularly useful if you want to postprocess in detail the outputs in the simulation folder with this id.
The script plots the evolution of the best function value found over the optimization, by all the optimizer, by each sample, the time needed to perform the iterations, all in different figures. In case the `number_of_dimensions` is equal to 1 or 2, additional visualizations will be plotted.
- `extractAllData`: to be used in a folder containing the folders of one or more optimization runs. The script extract all the datas (positions `X` and function values `f(X)`) explored during the optimization, stores them in a unique `pandas.DataFrame` and plots the `seaborn.pairplot` of these data. Having all the data in a single `DataFrame` is particularly useful if one wants to use machine learning techniques to create a surrogate model of `f(X)`.
- `compareOptimizationHistories.py`: to be used in a folder containing the folders of two or more optimization runs. The script will plot the convergence plot of the optimization runs in the folder where it was called. The average best function value and the standard deviation of the best function values found by the runs will be plotted. This is useful to compare different sets of runs, e.g. performed with a different optimizer, because multiple runs are necessary to study the statistical behavior of the different optimizers for a given problem.
- `readAndPlotAdaptiveParticleSwarmOptimizationHistoryHyperparameters.py`: to be used inside the folder of an optimization run performed with `"Adaptive Particle Swarm Optimization"`. The script plots the evolution of the hyperparameters (inertia weight, evolutionary factor, acceleration coefficient) that are computed and tuned automatically by this optimizer.
  
  