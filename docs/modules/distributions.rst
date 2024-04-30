Distributions
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

TinyDA is designed to make use of distributions from ``scipy.stats``. Hence, only tinyDA-specific distributions are documented here.

.. autosummary::
    :nosignatures:
    
    tinyDA.JointPrior
    tinyDA.PoissonPointProcess
    tinyDA.GaussianLogLike
    tinyDA.DefaultGaussianLogLike
    tinyDA.AdaptiveGaussianLogLike

Joint Prior
---------------

.. autoclass:: tinyDA.JointPrior
    :members:
    
    .. automethod:: __init__

Poisson Point Process
---------------------

.. autoclass:: tinyDA.PoissonPointProcess
    :members:

    .. automethod:: __init__

Gaussian Log-Likelihood
-----------------------

.. autofunction:: tinyDA.GaussianLogLike

.. autoclass:: tinyDA.DefaultGaussianLogLike
    :members:
    
    .. automethod:: __init__

Adaptive Gaussian Log-Likelihood
--------------------------------

.. autoclass:: tinyDA.AdaptiveGaussianLogLike
    :members:
    
    .. automethod:: __init__
