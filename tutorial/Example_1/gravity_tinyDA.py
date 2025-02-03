# limit multiprocessing.
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

# imports
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

import tinyDA as tda
from gravity import Gravity

# set a seed
np.random.seed(123)

# Set the model parameters.
depth = 0.1
n_quad = 60
n_data = 30

# set the "true" parameters
true_parameters = [{'position': np.array([0.2, 0.2]), 'radius': 0.15},
                   {'position': np.array([0.4, 0.75]), 'radius': 0.2},
                   {'position': np.array([0.8, 0.4]), 'radius': 0.1}]

# initialise a model
model_true = Gravity(depth, n_quad, n_data)
model_true.solve(true_parameters)

# add some noise to the data.
noise_stdev = 0.5
noise = np.random.normal(0, noise_stdev, n_data**2)
data = model_true.g + noise

# set up the prior
my_prior = tda.PoissonPointProcess(4, np.array([[0,1], [0,1]]), {'radius': stats.uniform(0.05, 0.25)})

# set up the likelihood
my_likelihood = tda.GaussianLogLike(data, noise_stdev**2*np.eye(data.size))

# Set the quadrature degree for each model level (coarsest first)
n_quadrature = [30, 60]
n_datapoints = [30, 30]

# Initialise the models, according the quadrature degree.
my_models = []
for i, (n_quad, n_data) in enumerate(zip(n_quadrature, n_datapoints)):
     my_models.append(Gravity(depth, n_quad, n_data))

# Plot the same random realisation for each level, and the corresponding signal,
# to validate that the levels are equivalents.
for i, m in enumerate(my_models):
    print('Level {}:'.format(i))
    m.solve(true_parameters)
    m.plot_model()
    plt.savefig('model_level_{}.png'.format(i), bbox_inches='tight')

# Plot the density and the signal.
fig, axes = plt.subplots(1, 2, figsize=(16,6))

axes[0].set_title('True Density')
t = axes[0].imshow(model_true.f.reshape(model_true.n_quad, model_true.n_quad), extent=(0,1,0,1), origin='lower', cmap='plasma')
fig.colorbar(t, ax=axes[0])
axes[0].grid(False)

axes[1].set_title('Noisy Signal')
c = axes[1].imshow(data.reshape(n_datapoints[0], n_datapoints[0]), extent=(0,1,0,1), origin='lower', cmap='plasma')
fig.colorbar(c, ax=axes[1])
axes[1].grid(False)
plt.savefig('density_and_data.png', bbox_inches='tight')

# initialise the tinyDA posteriors.
my_posteriors = [tda.Posterior(my_prior, my_likelihood, model) for model in my_models]

# Do a greedy search for a good initial model.
n_samples = 10000
state = my_prior.rvs()
logp_state = my_posteriors[-1].create_link(state).posterior

for i in tqdm(range(n_samples)):
    proposal = my_prior.rvs()
    logp_proposal = my_posteriors[-1].create_link(proposal).posterior
    if logp_proposal > logp_state:
        state = proposal
        logp_state = logp_proposal

# plot the initial point.
my_models[-1].solve(state)
my_models[-1].plot_model()
plt.savefig('initial_model.png', bbox_inches='tight')

# poisson point proposal
move_probabilities = {'create': 0.1, 'destroy': 0.1, 'move': 0.4, 'shuffle': 0.0, 'swap': 0.0, 'perturb': 0.4}
my_proposal = tda.PoissonPointProposal(move_probabilities)

# draw some samples.
chains = tda.sample(my_posteriors,
                    my_proposal,
                    iterations=60000,
                    n_chains=2,
                    subsampling_rate=5,
                    initial_parameters=[state, state])

# extract the samples.
solutions = [chains['chain_fine_0'][i].qoi.reshape(my_models[-1].TX.shape) for i in range(10000, 60000)] + \
            [chains['chain_fine_1'][i].qoi.reshape(my_models[-1].TX.shape) for i in range(10000, 60000)]
solutions = np.array(solutions)

# plot mean and the variance of the MCMC samples
fig, axes = plt.subplots(1, 2, figsize=(16,6))

axes[0].set_title('Mean')
m = axes[0].imshow(solutions.mean(axis=0), extent=(0,1,0,1), origin='lower', cmap='plasma')
fig.colorbar(m, ax=axes[0])
axes[0].grid(False)

axes[1].set_title('Variance')
v = axes[1].imshow(solutions.var(axis=0), extent=(0,1,0,1), origin='lower', cmap='plasma')
fig.colorbar(v, ax=axes[1])
axes[1].grid(False)

plt.savefig('mean_and_variance.png', bbox_inches='tight')
