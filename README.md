[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10139399.svg)](https://doi.org/10.5281/zenodo.10139399)

## ``:Discoveri``: Data-driven Investigation through Simulations on Clusters for the Optimization of the physical Variables' Effects in Regimes of Interest

### About ``:Discoveri``

``:Discoveri`` is a Python code to optimize/maximize a function with derivative-free methods. This function can be a `numpy` function or the result of a postprocessing function of simulations on a cluster. In both cases the user can define the function to optimize. In the latter case, ``:Discoveri`` prepares and launches automatically the simulations that sample the function to optimize. At the moment, the following optimization methods are implemented: `"Grid Search"`, `"Random Search"`,`"Bayesian Optimization"`, `"Particle Swarm Optimization"` (and its variants called `"Adaptive Particle Swarm Optimization"`, `"FST-PSO"`).

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

### Contributors
Francesco Massimo (LPGP, CNRS, Université Paris Saclay) started the repository, 
which benefitted from contributions from Paul Kigaya (University of Michigan).

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


#### General optimization parameters

- `optimization_method`(string): the derivative-free optimization techniques that will try to maximise `f(X)`.
At the moment the available options are:
  - `"Grid Search"`
  - `"Random Search"`
  - `"Bayesian Optimization"`
  - `"Particle Swarm Optimization"`
  - `"Adaptive Particle Swarm Optimization"`
  - `"FST-PSO"`
- `number_of_iterations`(integer): the number of iterations of the optimization process.
- `number_of_samples_per_iteration`(integer): the number of samples chosen/drawn at each iteration, each evaluating `f(X)` at a different sample `X`.
- `number_of_dimensions` (integer): number of dimensions of the parameter space where the maximum of `f(X)` is searched.
- `search_interval`(list of `number_of_dimensions` lists of 2 elements): in this list, the inferior and superior boundaries of the parameter space to explore (where the different `X` will always belong). The boundaries for each dimension of the explored parameter space must be provided. 
- `iterations_between_outputs`(integer): number of iterations between outputs, i.e. the output files dump and some message prints at screen.

#### Function to optimize

- `use_test_function`: if `True`, the function `f(X)` to optimize, i.e. maximize, will be `test_function`. Otherwise, it will be `simulation_postprocessing_function`. 
In both cases, the users must ensure that the function does not return `nan`,`-inf`,`inf`, and that a real result is always obtained.
- `test_function`: a real-valued `numpy` function of the position `X`.
- `simulation_postprocessing_function`: a real-valued function that returns the result of a postprocessing (defined by the users) of a simulation, e.g. the average energy of tracked particles of a Smilei simulation. ``:Discoveri`` will prepare the simulation directories, launch the simulations corresponding to the sampled `X` positions and postprocess the results. The users must ensure that the namelist of the used code can be modified to use the parameters in `X` as inputs. For more details, see the next section and the examples folder. 

#### Job preparation and management  in a cluster 

The users must ensure that these parameters are coherent. e.g. the template job submission script must set the correct name for the simulation log files, etc.
- `input_parameters_names` (list of `number_of_dimensions` strings): important to modify the namelist to launch simulations. Currently, ``:Discoveri`` assumes that a Python namelist is used by the code, where after a line containing `# Configuration to simulate` a line defining a dictionary `configuration` will be added by ``:Discoveri``, containing the names of the parameters to explore and their values. The namelist of the code must be prepared in order to use this dictionary.
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


### Available optimization methods and their hyperparameters
Following are the optimization techniques currently supported by ``:Discoveri``, as well as their hyperparameters. 

- `"Grid Search"`: the arrays `X` of the `number_of_samples_per_iteration` samples are evenly distributed in each dimension `idim` between `search_interval[idim][0]` and `search_interval[idim][1]`, with `samples_per_dimension[idim]` samples.
NOTE: this optimizer only supports `number_of_iterations = 1`.
Optimizer hyperparameters:
  - `samples_per_dimension` a list of `number_of_dimensions` integers, whose product must be equal to `number_of_samples_per_iteration`.
  
- `"Random Search"`: in this optimization method, the array `X` for each sample of each iteration is generated pseudo-randomly, with a uniform distribution within the `search_interval`. 
Optimizer hyperparameter:
  - `use_Halton_sequence` (default value = `True`): if `True`, a scrambled Halton sequence (https://en.wikipedia.org/wiki/Halton_sequence) is used to draw the samples. Otherwise, they are drawn using `numpy.random.uniform` scaled and shifted for each dimension to draw `X` from the `search_interval` of interest.
  
- `"Bayesian Optimization"`: based on the implementation of Gaussian Process Regression  provided by the `scikit-learn` library (https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor), with a Matérn kernel characterized by its `nu` and `length_scale`. An expected improvement acquisition function is used.
Optimizer hyperparameters:
  - `number_of_tests` (default value: `2000*number_of_dimensions`): the number of points evaluated with the surrogate model function to pick the most promising ones through the expected improvement acquisition function. 
  - `nu` (default value = `1.5`): parameter for the Matérn kernel, related to its differentiability. See its definition in https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html
  - `length_scale` (default value = `1.0`): ideally it should be of the order of the variation scale/s (normalized by the `search_interval` size) of the function to optimize. The Gaussian Process Regressor will try to optimize it/them by maximizing the log-marginal-likelihood every time new data is fitted to the model, i.e. at each iteration. See its definition in https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html
  See also https://scikit-learn.org/stable/modules/gaussian_process.html#gp-kernels (Example 1.7.2.1) to understand how a good choice of `length_scale` can improve the surrogate model function predictions.
  - `length_scale_bounds` (default value = `(1e-05, 100000.0)`): bounds of `length_scale` during their optimization. See its definition in https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html
  - `xi` (default value = `0.`): parameter used in the acquisition function (expected improvement) 
  to tune the balance between exploitation of good points already found and exploration of the possible `X` positions. When equal to 0, exploitation is privileged. High values instead privilege exploration.

<!-- - `"Genetic Algorithm"`: a genetic algorithm where at each iteration `number_of_parents` biological parents are selected from a population (those with highest `f(X)` are picked). For each child to generate (corresponding to `number_of_samples_per_iteration`), two biological parents are randomly paired and the position `X` of their child in each dimension is a random number drawn from a uniform distribution between the minimum and maximum value of the coordinate in that dimension of its parents. At the moment this is the only crossover method implemented in ``:Discoveri``. For each child, for each dimension, there is a probability `probability_of_mutation` that the corresponding position coordinate is mutated, randomly selected within the `search_interval`. At the first iteration, the population from which parents are selected is made only of the initial samples (which are `number_of_samples_per_iteration`). At the next iterations, the population from which parents are selected is made of the parents selected at the previous iteration and their children. 
Optimizer hyperparameters:
  - `probability_of_mutation` (default value = `0.1`): the probability of mutation for each dimension of each child at each iteration.
  - `number_of_parents` (default value = `int(0.3*number_of_samples_per_iteration)`): the number of parents selected from the population at each iteration to generate children. The number of children will always be equal to `number_of_samples_per_iteration`. The `number_of_parents` must be at least 2 (although at least 3 is recommended to improve the genetic diversity of their children). IMPORTANT: since the number of possible pairing of `number_of_parents` parents is `number_of_parents!/[2*(number_of_parents-2)!]`, and since each pairing will generate a child, the `number_of_samples_per_iteration` should be larger than this number or two identical or similar children may be generated. This can result in choosing these children as parents in the next iteration. If mutations do not occur, their offspring would be identical or similar to their parents and the optimizer would quickly get stuck. -->
  
- `"Particle Swarm Optimization"`: the most common version of Particle Swarm Optimization (PSO), where an inertia term is added to the velocity update, as described in Y. Shi, R.C. Eberhart, 1998 https://ieeexplore.ieee.org/document/699146 . The `num_sample` samples represent the particles of the swarm, with position `X` and a velocity. At each iteration `t` the velocity of the particles are updated adding a cognitive velocity component (proportional to a constant coefficient `c1`), a social velocity component (proportional to a constant coefficient `c2`, `c1+c1<4`) and the velocity at the last iteration multiplied by the constant inertia weight `w` (which must be `<1` to avoid velocity divergence). At each iteration, for each dimension `k`, the velocity component `V_k` and the position component `X_k` of each particle `i` at iteration `t+1` will be updated at each iteration using 
`V_k(t+1)=w*V_k(t)+c1*rand1*[Best_X_i_k-X_k(t)]+c2*rand2*[Best_X_swarm_k-X_k(t)]` 
and 
`X_k(t+1)=X_k(t)+V_k(t+1)`,
where `rand1` and `rand2` are two random numbers from between `0` and `1`, and `Best_X_i_k` and `Best_X_swarm` are the best positions `X` found so far respectively by the particle `i` and by the entire swarm.
The optimum positions `Best_X_i` are updated if better positions are found the particles `i`. Coherently, if one of these positions is better than the best one found by the swarm so far,  `Best_X_swarm` is updated. The particles initial positions are initialized with a scrambled Halton sequence. Particles crossing a `search_interval` boundary in a dimension will have their coordinate in that dimension reinitialized, drawn from a uniform distribution in the `search_domain` span in that dimension, their velocity will remain unchanged.
Optimizer hyperparameters:
  - `c1` (cognitive acceleration coefficient, default value = `2.`): high values of `c1` promote the exploitation of positions near the best position found by the individual particle, updated at each iteration if a better position is found. If provided by the users, they must ensure that `c1+c1<4`.
  - `c2` (social acceleration coefficient, default value = `2.`): high values of `c2` promote the exploitation of positions near the best position found by the entire swarm, updated at each iteration if a better position is found. If provided by the users, they must ensure that `c1+c1<4`.
  - `w` (inertia weight, default value = `0.8`): high values of this coefficient reduce the variations of the velocity of the particle. If provided by the users, they must ensure that it is `<1` to avoid velocity divergence.
  - `w1`, `w2` (initial and final inertia weight, default value equal to `w`): if these values are specified, the inertia
  weight will linearly decrease from `w1` to `w2`, otherwise it will remain constant.
  - `max_speed` (maximum speed, default value: an array with elements equal to the respective size of `search_interval` in each dimension multiplied by `0.4`). This array is also the maximum absolute value for the initial velocities of the particles in each dimension (which are drawn randomly).
  - `boundary_conditions` (default: `"damping"`): if `"relocating"` and a particle exits the `search_interval` in one of its position coordinates, that coordinate is reassigned randomly within the `search_interval` boundaries, the velocity of the particle remains unchanged. If `"damping"` and a particle exits the `search_interval` in one of its position coordinates, that coordinate is reassigned at the crossed boundary, the velocity in that dimension is inverted and multiplied by a random number within the interval [0,1).
  - `use_multiple_swarms` (default = False)`: if `True`, the swarm is divides in `int(number_of_samples_per_iteration/subswarm_size)` independent swarms
  - `subswarm_size`: the size of the swarm if `use_multiple_swarms=True`. This number must divide evenly the `number_of_samples_per_iteration`.
  - `subswarms_distribution` (default: `"all_the_search_space"`): if `"all_the_search_space"`, the subswarms will be distributed through the whole search space; if `"search_space_subdomains"`, each subswarm will be initially assigned to a different part of the search space.
  - `subswarm_regrouping` (default: `False`): if `True`, every `iterations_beween_subswarm_regrouping` iterations the particles are reassigned randomly to a different subswarm, similarly to J.J. Liang, P.N. Suganthan, Proceedings 2005 IEEE Swarm Intelligence Symposium, 2005. SIS 2005 (https://doi.org/10.1109/SIS.2005.1501611).
  - `iterations_beween_subswarm_regrouping` (default: 5): number of iterations between subswarm regrouping (if activated).
  
- `"Adaptive Particle Swarm Optimization"` (Adaptive PSO): based on from Z.-H. Zhan et al., IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) 39, 6 (2009) https://ieeexplore.ieee.org/document/4812104 .
Based on the evolutionary state of the swarm, the coefficients `c1`, `c2` and the inertia weight `w` are updated as described in that article. Compared to the description in the original reference, no fuzzy classification is used. 
Optimizer hyperparameters:
  - `c1` (cognitive acceleration coefficient, default value = `2.0`): same as in the `"Particle Swarm Optimization"`, but the value provided by the user is just the initial value of the coefficient.
  - `c2` (social acceleration coefficient, default value = `2.0`): same as in the `"Particle Swarm Optimization"`, but the value provided by the user is just the initial value of the coefficient.
  - `w` (inertia weight, default value = `0.9`): same as in the `"Particle Swarm Optimization"`, but the value provided by the user is just the initial value of the inertia weight.
  - `boundary_conditions`: same as in the `"Particle Swarm Optimization"`.
  - `perturbation_global_best_particle` (default value = `True`): when `True` and the evolutionary state of the swarm is `Convergence`, the position of the best particle (the one with highest `f(X)`) is mutated adding a random perturbation drawn from a Gaussian distribution on one random coordinate (see original reference) and this new position is assigned to the worst particle of the swarm (the one with lowest `f(X)`). This feature allows the swarm to jump out of a local maximum more easily.
  - `max_speed`: this hyperparameter is ignored in this version of PSO, since the maximum speed for the particles is adaptively changed following X. Li et al., Neurocomputing 447 (2021) 64–79, https://doi.org/10.1016/j.neucom.2021.03.077 . The ratio between the maximum speed in a given dimension and the `search_interval` size in that dimension is the value `mu`, which will be chosen within the interval [`mu_min`,`mu_max`]. The initial `mu` for the initial speed of the particles is `mu_max`.
  - `mu_min` (default value = `0.4`).
  - `mu_max` (default value = `0.7`), must be larger than `mu_min`.
- `"FST-PSO"` (Fuzzy Self-Tuning PSO): at each iteration, the hyperparameters of each particle (`c1`,`c2`,`w`,`L`,`U`), where `L` and `U` are the minimum and maximum absolute values for the normalized velocity, are adapted following the set of fuzzy rules described in M. Nobile et al., Swarm and Evolutionary Computation 39 (2018) 70–85. This way the user only has to set the `boundary_conditions` of the optimizer.
  - `boundary_conditions`: same as in the `"Particle Swarm Optimization"`.
  - `use_multiple_swarms`: same as in the `"Particle Swarm Optimization"`.
  - `subswarm_size`: same as in the `"Particle Swarm Optimization"`.
  - `subswarms_distribution`: same as in the `"Particle Swarm Optimization"`
  - `subswarm_regrouping` (default: `False`): same as in the `"Particle Swarm Optimization"`. 
  - `iterations_beween_subswarm_regrouping` (default: 5): same as in the `"Particle Swarm Optimization"`.
  - `subswarm_regrouping` (default: `False`): same as in the `"Particle Swarm Optimization"`. 
  - `iterations_beween_subswarm_regrouping` (default: 5): same as in the `"Particle Swarm Optimization"`.
  
#### Postprocessing:
In the folder `postprocessing_scripts` several scripts are available to have an insight on the optimization run(s) made with ``:Discoveri``:
- `readAndPlotOptimizationHistory`: Probably the first script that should be used to have an overview of an optimization run. This script must be used inside the folder of the optimization run to analyse. The script tells the best position `X` found by the run, the corresponding value of the function `f(X)` and the corresponding id of the function evaluation. This is particularly useful if you want to postprocess in detail the outputs in the simulation folder with this id.
The script also plots the evolution of the best function value found during the optimization run, by all the optimizer, by each sample, as function the time needed to perform the iterations, or of the number of iterations, all in different figures. In case the `number_of_dimensions` is equal to 1 or 2, additional visualizations will be plotted.
- `extractAllData`: to be used in a folder containing the folders of one or more optimization runs. The script extract all the datas (positions `X` and function values `f(X)`) explored during the optimization, stores them in a unique `pandas.DataFrame` and plots the `seaborn.pairplot` of these data. Having all the data in a single `DataFrame` is particularly useful if one wants to use machine learning techniques to create a surrogate model of `f(X)`.
- `compareOptimizationHistories.py`: to be used in a folder containing the folders of two or more optimization runs. The script will plot the convergence plot of the optimization runs in the folder where it was called. The average best function value and the standard deviation of the best function values found by the runs will be plotted. This is useful to compare different sets of runs, e.g. performed with a different optimizer, because multiple runs are necessary to study the statistical behavior of the different optimizers for a given problem.
- `readAndPlotAdaptiveParticleSwarmOptimizationHistoryHyperparameters.py`: to be used inside the folder of an optimization run performed with `"Adaptive Particle Swarm Optimization"`. The script plots the evolution of the hyperparameters (inertia weight, evolutionary factor, acceleration coefficients) that are computed and tuned automatically by this optimizer at each iteration.
- `readAndPlotFSTPSOHistoryHyperparameters.py`: to be used inside the folder of an optimization run performed with `"FST-PSO"`. The script plots the evolution of the hyperparameters (inertia weight, acceleration coefficients, minimum and maximum absolute value for velocity) as well as the parameters delta and Phi that are computed aby this optimizer at each iteration.
  
  
