Distributions
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

TinyDA is designed to make use of distributions from ``scipy.stats``. Hence, only tinyDA-specific distributions are documented here.

.. autosummary::
    :nosignatures:
    
    tinyDA.CompositePrior
    tinyDA.PoissonPointProcess
    tinyDA.GaussianLogLike
    tinyDA.AdaptiveGaussianLogLike

Composite Prior
---------------

.. autoclass:: tinyDA.CompositePrior
    :members:
    
    .. automethod:: __init__

Poisson Point Process
---------------------

.. autoclass:: tinyDA.PoissonPointProcess
    :members:

    .. automethod:: __init__

Gaussian Log-Likelihood
-----------------------

.. autoclass:: tinyDA.GaussianLogLike
    :members:
    
    .. automethod:: __init__

Adaptive Gaussian Log-Likelihood
--------------------------------

.. autoclass:: tinyDA.AdaptiveGaussianLogLike
    :members:
    
    .. automethod:: __init__
