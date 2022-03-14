Diagnostics
===========

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
.. autosummary::
    :nosignatures:
    
    tinyDA.to_inference_data
    tinyDA.get_samples
    tinyDA.to_xarray
    tinyDA.plot_samples
    tinyDA.plot_sample_matrix
    tinyDA.compute_R_hat
    tinyDA.compute_ESS


Convert tinyDA.sample() output to ArviZ InferenceData
-----------------------------------------------------
**Note:** ArviZ ships with a wealth of MCMC postprocessing tools. Use this
function to convert output from tinyDA.sample() to an ArviZ InferenceData
object, which allows for directly using the ArviZ diagnostics suite.

.. autofunction:: tinyDA.to_inference_data

Helper functions
----------------------

.. autofunction:: tinyDA.get_samples

.. autofunction:: tinyDA.to_xarray

Legacy functions
----------------------

.. autofunction:: tinyDA.plot_samples

.. autofunction:: tinyDA.plot_sample_matrix

.. autofunction:: tinyDA.compute_R_hat

.. autofunction:: tinyDA.compute_ESS

