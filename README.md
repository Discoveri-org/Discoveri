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

#### Available optimization methods and their hyperparameters (to do)
- `"Random Search"`: in this optimization method, the array `X` for each sample of each iteration is generated pseudo-randomly, with a uniform distribution within the `search_interval`.
  - `use_Halton_sequence` (boolean): if `True`, a scrambled Halton sequence (https://en.wikipedia.org/wiki/Halton_sequence) is used to draw the samples. Otherwise, they are drawn using `numpy.random.uniform` scaled and shifted for each dimension to draw `X` frim the `search_interval` of interest.
- `"Bayesian Optimization"`: based on the implementation of Gaussian process regression (GPR) provided by the `scikit-learn` library, with an anisotropic Matérn kernel with `nu=1.5`. The length scale of the Matérn kernel was optimized by maximizing the log-marginal-likelihood every time new data is fitted to the model. An expected improvement acquisition function is used.
- `"Particle Swarm Optimization"`: the most common version of Particle Swarm Optimization, where an inertia term is added to the velocity update, as described in (Y. Shi, R.C. Eberhart, 1998 https://ieeexplore.ieee.org/document/699146). The `num_sample` samples represent the particles of the swarm, with position `X` and a velocity. At each iteration the velocity of the particles are updated adding a cognitive velocity component (proportional to a constant coefficient `c1`), a social velocity component (proportional to a constant coefficient `c2`, `c1+c1<4`) and the velocity at the last iteration multiplied by the constant inertia weight `w` (which must be `<1` to avoid velocity divergence). For each particle, high values of `c1` promote the exploration near the best position found by the particle, while high values of `c2` promote the exploitation of the knowledge of the best position found by the entire swarm. Both best positions are updated at each iteration if better positions are found.
- `"IAPSO"`:
- `"PSO-TPME"` (Particle Swarm Optimization with Targeted, Position-Mutated Elitism):
