Posterior and Links
===================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
.. autosummary::
    :nosignatures:
    
    tinyDA.Posterior
    tinyDA.Link
   
Posterior
---------

.. autoclass:: tinyDA.Posterior
    :members:
    
    .. automethod:: __init__

The Link class
--------------
**Note:** The tinyDA.Link class is a container class used to store all relevant 
sample information such as parameter values, model output and densities. It 
is used by tinyDA under the hood and the user is as such never required to 
directly utilize the class. The documentation is provided here only as a
reference.

.. autoclass:: tinyDA.Link
    :members:
    
    .. automethod:: __init__
