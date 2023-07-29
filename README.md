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
- `"Bayesian Optimization"`:
- `"Particle Swarm Optimization"`:
- `"IAPSO"`:
- `"PSO-TPME"`:
