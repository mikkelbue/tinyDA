.. tinyDA documentation master file, created by
   sphinx-quickstart on Thu Sep  2 12:44:08 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tinyDA
======

tinyDA is a Delayed Acceptance MCMC sampler with finite-length subchain sampling and adaptive error modelling. 

This is intended as a simple, lightweight implementation, with minimal dependencies, i.e. nothing beyond the SciPy stack. 

It is fully imperative and easy to use!

Features
--------

**Proposals**

- Random Walk Metropolis Hastings (RWMH)
- preconditioned Crank-Nicolson (pCN)
- Adaptive Metropolis (AM)
- Adaptive pCN
- DREAM(Z)
- Multiple-Try Metropolis (MTM)

**Adaptive Error Models**

- State independent
- State dependent

**Diagnostics**

- ArviZ compatibility

Documentation
-------------

.. toctree::
   :maxdepth: 1
   
   modules/sampler
   modules/posterior
   modules/proposals
   modules/distributions
   modules/diagnostics
   modules/chains
   modules/utils

Examples
--------

Please refer to the `Jupyter Notebooks`_ in the `GitHub repository`_.

.. _`Jupyter Notebooks`: https://github.com/mikkelbue/tinyDA/tree/main/examples
.. _`GitHub repository`: https://github.com/mikkelbue/tinyDA

Installation
------------

Install tinyDA by running:

    ``pip install tinyDA``

**Dependencies:**

- NumPy
- SciPy
- ArviZ
- tqdm
- Ray (multiprocessing, optional)

Contribute
----------

- `GitHub repository`_
- `Issue tracker`_

.. _`GitHub Repository`: https://github.com/mikkelbue/tinyDA
.. _`Issue Tracker`: https://github.com/mikkelbue/tinyDA/issues


License
-------

The project is licensed under the MIT_ license.

.. _MIT: https://github.com/mikkelbue/tinyDA/blob/main/LICENSE

References
----------

- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of State Calculations by Fast Computing Machines. The Journal of Chemical Physics, 21(6), 1087–1092. https://doi.org/10.1063/1.1699114
- Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. Biometrika, 13.
- Liu, J. S., Liang, F., & Wong, W. H. (2000). The Multiple-Try Method and Local Optimization in Metropolis Sampling. Journal of the American Statistical Association, 95(449), 121–134. https://doi.org/10.1080/01621459.2000.10473908
- Haario, H., Saksman, E., & Tamminen, J. (2001). An Adaptive Metropolis Algorithm. Bernoulli, 7(2), 223. https://doi.org/10.2307/3318737
- Cotter, S. L., Roberts, G. O., Stuart, A. M., & White, D. (2013). MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster. Statistical Science, 28(3), 424–446. https://doi.org/10.1214/13-STS421
- Hu, Z., Yao, Z., & Li, J. (2016). On an adaptive preconditioned Crank-Nicolson MCMC algorithm for infinite dimensional Bayesian inferences. http://arxiv.org/abs/1511.05838
- Vrugt, J. A. (2016). Markov chain Monte Carlo simulation using the DREAM software package: Theory, concepts, and MATLAB implementation. Environmental Modelling & Software, 75, 273–316. https://doi.org/10.1016/j.envsoft.2015.08.013
- Cui, T., Fox, C., & O’Sullivan, M. J. (2018). A posteriori stochastic correction of reduced models in delayed acceptance MCMC, with application to multiphase subsurface inverse problems. http://arxiv.org/abs/1809.03176
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2020). Rank-normalization, folding, and localization: An improved R for assessing convergence of MCMC. https://doi.org/10.1214/20-BA1221
- Lykkegaard, M. B., Dodwell, T. J., & Moxey, D. (2020). Accelerating Uncertainty Quantification of Groundwater Flow Modelling Using Deep Neural Networks. http://arxiv.org/abs/2007.00400
- Lykkegaard, M. B., Mingas, G., Scheichl, R., Fox, C., & Dodwell, T. J. (2020). Multilevel Delayed Acceptance MCMC with an Adaptive Error Model in PyMC3.
