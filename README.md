![](https://github.com/mikkelbue/tinyDA/blob/main/misc/tinyDA.png)

# tinyDA
Multilevel Delayed Acceptance MCMC sampler with finite-length subchain sampling and adaptive error modelling. This is intended as a simple, lightweight implementation, with minimal dependencies, i.e. nothing beyond the SciPy stack and ArviZ. It is fully imperative and easy to use!

For instructions, have a look at the [documentation](https://tinyda.readthedocs.io/en/latest/), the [examples](https://github.com/mikkelbue/tinyDA/tree/main/examples) or the [usage section below](#usage).

## Installation
tinyDA can be installed from PyPI:
```
pip install tinyDA
```

## Dependencies
* NumPy
* SciPy
* ArviZ
* tqdm
* [Ray](https://docs.ray.io/en/master/) (multiprocessing, optional)

## Features

### Samplers
* Metropolis-Hastings
* Delayed Acceptance (Christen & Fox, 2005)
* Multilevel Delayed Acceptance (Lykkegaard et al. 2022)

### Proposals
* Random Walk Metropolis Hastings (RWMH) - Metropolis et al. (1953), Hastings (1970)
* preconditioned Crank-Nicolson (pCN) - Cotter et al. (2013)
* Adaptive Metropolis (AM) - Haario et al. (2001)
* Operator-weighted pCN - Law (2014)
* DREAM(Z) - Vrugt (2016)
* Multiple-Try Metropolis (MTM) - Liu et al. (2000)

### Adaptive Error Models
* State independent - Cui et al. (2018)
* State dependent - Cui et al. (2018)

### Diagnostics
* Convert a tinyDA chain to an ArviZ InferenceData object for near-unlimited diagnostics!

## Usage
Documentation is available at [Read the Docs](https://tinyda.readthedocs.io/en/latest/). A few illustrative examples are available as Jupyter Notebooks in the root directory. Below is a short summary of the core features.

### Distributions
The prior and likelihood can be defined using standard `scipy.stats` classes:
```python
import tinyDA as tda

from scipy.stats import multivariate_normal

# set the prior mean and covariance.
mean_prior = np.zeros(n_dim)
cov_prior = np.eye(n_dim)

# set the covariance of the likelihood.
cov_likelihood = sigma**2*np.eye(data.shape[0])

# initialise the prior distribution and likelihood.
my_prior = multivariate_normal(mean_prior, cov_prior)
my_loglike = tda.GaussianLogLike(data, cov_likelihood)
```
If using a Gaussian likelihood, we recommend using the `tinyDA` implementation, since it is unnormalised and plays along well with `tda.AdaptiveLogLike` used for the Adaptive Error Model. Home-brew distributions can easily be defined, and must have a `.rvs()` method for drawing random samples and a `logpdf(x)` method for computing the log-likelihood, as per the `SciPy` implementation.

### tinyDA.Posterior
The heart of the TinyDA sampler is the `tinyDA.Posterior`, which is responsible for:
1. Calling the model with some parameters (a proposal) and collecting the model output.
2. Evaluating the prior density of the parameters, and the likelihood of the data, given the parameters.
3. Constructing `tda.Link` instances that hold information for each sample.

![](https://github.com/mikkelbue/tinyDA/blob/main/misc/flowchart.png)

The `tinyDA.Posterior` takes as input the prior, the likelihood, and a forward model. Therefore, a forward model must be defined. This model can be either a function `model_output = my_function(parameters)` or a class instance with a `.__call__(self, parameters)` method. The function or `__call__` method must return either just the model output or a tuple of `(model_output, qoi)`. In this example, we define a class that performs simple linear regression on whatever inputs `x` we have.

```python
class MyLinearModel:
    def __init__(self, x):

        self.x = x
        
    def __call__(self, parameters):
        
        # the model output is a simple linear regression
        model_output = parameters[0] + parameters[1]*self.x
        
        # no quantity of interest beyond the parameters.
        qoi = None
        
        # return both.
        return model_output, qoi

my_model = MyLinearModel(x)
my_posterior = tda.Posterior(my_prior, my_loglike, my_model)
```

### Proposals
A proposal is simply initialised with its parameters:
```python
# set the covariance of the proposal distribution.
am_cov = np.eye(n_dim)

# set the number of iterations before starting adaptation.
am_t0 = 1000

# set some adaptive metropolis tuning parameters.
am_sd = 1
am_epsilon = 1e-6

# initialise the proposal.
my_proposal = tda.AdaptiveMetropolis(C0=am_cov, t0=am_t0, sd=am_sd, epsilon=am_epsilon)
```

### Sampling
After defining a proposal, a coarse posterior `my_posterior_coarse`, and a fine posterior `my_posterior_fine`, the Delayed Acceptance sampler can be run using `tinyDA.sample()`:
```python
my_chains = tda.sample([my_posterior_coarse, my_posterior_fine], 
                       my_proposal, 
                       iterations=12000, 
                       n_chains=2, 
                       subsampling_rate=10)
```

If using a hirarchy with more than two models, a Multilevel Delayed Acceptance sampler can be run by supplying a list of posteriors in ascending order and a correponsing list of subsampling rates:
```python
my_chains = tda.sample([my_posterior_level0, 
                        my_posterior_level1, 
                        my_posterior_level2, 
                        my_posterior_level3], 
                       my_proposal, 
                       iterations=12000, 
                       n_chains=2, 
                       subsampling_rate=[10, 5, 5])
```

### Postprocessing
The entire sampling history is now stored in `my_chains` in the form of a dictionary with tinyDA.Link instances. You can convert the output of `tinyDA.sample()` to an ArviZ InferenceData object with 
```python
idata = tda.to_inference_data(my_chains, burnin=2000)
```
If you want to have a look at the coarse samples, you can pass an additional argument:
```python
idata = tda.to_inference_data(my_chains, level='coarse', burnin=20000)
```

The `idata` object can then be used with the ArviZ diagnostics suite to e.g. get MCMC statistics, plot the traces and so on.

## Contributing
I (mikkelbue) am currently the sole contributor to this package. I have been using it in my own research, and have so far been developing it (mostly) for myself. 

If you feel that it is missing some features, or that something could be improved, please do not hesitate to create a fork and submit a PR! If you want to help me improve the package, please have a look at the [issues](https://github.com/mikkelbue/tinyDA/issues) and consider if something seems doable to you. I am currently working on a scientific paper that I plan to submit to Journal of Statistical Software, and would be happy for other people to get involved.

If you would like to contribute, please consider the following:
* I am hoping to keep the list of dependencies **short**, and would rather not include any additional large libraries, unless it is strongly warranted. Great things can be achieved using NumPy!
* I am pretty good at converting theoretical methods into computer code, but I am not a software engineer. Any kind of CI, tests and improvements to the software infrastructure would be greatly appreciated!

## TODO
* ~~Parallel multi-chain sampling~~
* ~~More user-friendly diagnostics~~
* ~~Multilevel Delayed Acceptance~~
* Variance Reduction
* MALA proposal
* Wrapper for framework-agnostic adaptive coarse model
* Embedded spaces for hierachical models
* Tests


