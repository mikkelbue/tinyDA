![](https://github.com/mikkelbue/tinyDA/blob/main/misc/tinyDA.png)

# tinyDA
Delayed Acceptance (Christen & Fox, 2005) MCMC sampler with finite-length subchain sampling and adaptive error modelling. 

This is intended as a simple, lightweight implementation, with minimal dependencies, i.e. nothing beyond the SciPy stack. 

It is fully imperative and easy to use!

## Installation
tinyDA can be installed from PyPI:
```
pip install tinyDA
```

## Features

### Proposals
* Random Walk Metropolis Hastings (RWMH) - Metropolis et al. (1953), Hastings (1970)
* preconditioned Crank-Nicolson (pCN) - Cotter et al. (2013)
* Adaptive Metropolis (AM) - Haario et al. (2001)
* Adaptive pCN - Hu and Yao (2016)
* DREAM(Z) - Vrugt (2016)
* Multiple-Try Metropolis (MTM) - Liu et al. (2000)

### Adaptive Error Models
* State independent - Cui et al. (2018)
* State dependent - Cui et al. (2018)

### Diagnostics
* A bunch of plotting functions
* Rank-normalised split-<img src="https://latex.codecogs.com/gif.latex?\hat{R} " />  and ESS - Vehtari et al. (2020)

### Dependencies:
* NumPy
* SciPy
* tqdm
* [pyDOE](https://pythonhosted.org/pyDOE/) (optional)
* [Ray](https://docs.ray.io/en/master/) (multiprocessing, optional)

## Usage
A few illustrative examples are available as Jupyter Notebooks in the root directory. Below is a short summary of the core features.

### Distributions
The prior and likelihood can be defined using standard `scipy.stats` classes:
```python
import tinyDA as tda

from scipy.stats import multivariate_normal

mean_prior = np.zeros(n_dim)
cov_prior = np.eye(n_dim)
cov_likelihood = sigma**2*np.eye(data.shape[0])

my_prior = multivariate_normal(mean_prior, cov_prior)
my_loglike = tda.LogLike(data, cov_likelihood)
```
If using a Gaussian likelihood, we recommend using the `tinyDA` implementation, since it is unnormalised and plays along well with `tda.AdaptiveLogLike` used for the Adaptive Error Model. Home-brew distributions can easily be defined, and must have a `.rvs()` method for drawing random samples and a `logpdf(x)` method for computing the log-likelihood, as per the `SciPy` implementation.

### tinyDA.LinkFactory
At the heart of the TinyDA sampler sits what we call a `LinkFactory`, which is responsible for:
1. Calling the model with some parameters (a proposal) and collecting the model output.
2. Evaluating the prior density of the parameters, and the likelihood of the model output, given the parameters.
3. Constructing `tda.Link` instances that hold information for each sample.

![](https://github.com/mikkelbue/tinyDA/blob/main/misc/flowchart.png)

The `LinkFactory` must be defined by inheritance from either `tda.LinkFactory` or `tda.BlackBoxLinkFactory`. The former allows for computing the model output directly from the input parameters, using pure Python or whichever external library you want to call. The `evaluate_model()` method must thus be overwritten:

```python
class MyLinkFactory(tda.LinkFactory):
    def evaluate_model(self, parameters):
        output = parameters[0] + parameters[1]*x
        qoi = None
        return output, qoi

my_link_factory = MyLinkFactory(my_prior, my_loglike)
```

The latter allows for feeding some model object to the `LinkFactory` at initialisation, which is then assigned as a class attribute. This is useful for e.g. PDE solvers. Your model must return ordered data when called (e.g. via a `__call__(self, parameters)` method), and there is no need to overwrite `evaluate_model()`. This is what happend under the hood:
```python
class MyLinkFactory(tda.BlackBoxLinkFactory):
    def evaluate_model(self, parameters):
            
        # get the model output.
        model_output = self.model(parameters)
        
        # get the quantity of interest.
        if self.get_qoi:
            qoi = self.model.get_qoi()
        else:
            qoi = None
            
        # return everything.
        return model_output, qoi

my_link_factory = MyLinkFactory(my_model, my_datapoints, my_prior, my_loglike, get_qoi=True)
```
### Proposals
A proposal is simply initialised with its parameters:
```python
am_cov = np.eye(n_dim)
am_t0 = 1000
am_sd = 1
am_epsilon = 1e-6
my_proposal = tda.AdaptiveMetropolis(C0=am_cov, t0=am_t0, sd=am_sd, epsilon=am_epsilon)
```

### Sampling
The Delayed Acceptance sampler can then be initalised and run, simply with:
```python
my_chain = tda.DAChain(my_link_factory_coarse, my_link_factory_fine, my_proposal, subsampling_rate)
my_chain.sample(n_samples)
```
If you decide you need more samples, you can just call `tda.DAChain.sample()` again, since all samples and tuning parameters are cached:
```python
my_chain.sample(additional_n_samples)
```

### Postprocessing
The entire sampling history is then stored in `my_chain`, and you can extract an array of samples by doing:
```python
samples_fine = tda.get_parameters(my_chain)
samples_coarse = tda.get_parameters(my_chain, level='coarse')
```

Some diagnostics are available in the diagnostics module. Please refer to their respective docstrings for usage instructions.

# TODO
* ~~Parallel multi-chain sampling~~
* ~~Population-based proposals~~
* Multilevel Delayed Acceptance
* More user-friendly diagnostics

