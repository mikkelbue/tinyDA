#import pytest
from time import sleep
from scipy.stats import multivariate_normal
import numpy as np
from arviz import summary
import logging

from tinyDA.proposal import DREAMZ, DREAM, GaussianRandomWalk
from tinyDA.distributions import GaussianLogLike
from tinyDA.posterior import Posterior
from tinyDA.sampler import sample
from tinyDA.diagnostics import to_inference_data

PRIOR_MEAN = np.array([12, 9])
PRIOR_COV = np.array([[32, -16], [-16, 32]])

NOISE_SIGMA = 2e-4
OBSERVED = -1e-3

PRIOR = multivariate_normal(mean=PRIOR_MEAN, cov=PRIOR_COV)
LIKELIHOOD = GaussianLogLike(np.full((1, 1), OBSERVED), np.full((1, 1), NOISE_SIGMA))
def model(params):
    sleep(0.01) # add delay to model to allow for more thorough mixing in DREAM
    return np.array(-1 / 80  * (3 / np.exp(params[0]) + 1 / np.exp(params[1])))
MODEL = model
POSTERIOR = Posterior(PRIOR, LIKELIHOOD, MODEL)

BURN_IN = 200
ITERATIONS = 500 + BURN_IN
# DREAM(Z)'s period always 1?
#ADAPTION_PERIOD = 5
CHAINS = 20
M0 = 50 # initial archive size
DELTA = 3 # number of sample pairs to use to compute jump
NCR = 3

def test_DREAMZ():
    proposal = DREAMZ(
        M0=M0,
        delta=DELTA,
        nCR=NCR,
        #period=ADAPTION_PERIOD,
    )

    samples = sample(
        posteriors=POSTERIOR,
        proposal=proposal,
        iterations=ITERATIONS,
        n_chains=CHAINS,
        force_sequential=False
    )

    idata = to_inference_data(samples, burnin=BURN_IN)
    print(summary(idata))
    return idata

def test_DREAM():
    proposal = DREAM(
        M0=M0,
        delta=DELTA,
        nCR=NCR,
        #period=ADAPTION_PERIOD
    )

    samples = sample(
        posteriors=POSTERIOR,
        proposal=proposal,
        iterations=ITERATIONS,
        n_chains=CHAINS,
        force_sequential=False
    )

    idata = to_inference_data(samples, burnin=BURN_IN)
    print(summary(idata))
    return idata

def test_DREAMZ_sequential():
    proposal = DREAMZ(
        M0=M0,
        delta=DELTA,
        nCR=NCR,
        #period=ADAPTION_PERIOD
    )

    samples = sample(
        posteriors=POSTERIOR,
        proposal=proposal,
        iterations=(ITERATIONS - BURN_IN) * CHAINS // 2,
        # arviz needs 2 chains minimum?
        n_chains=2,
        force_sequential=False
    )

    idata = to_inference_data(samples, burnin=BURN_IN)
    print(summary(idata))
    return idata

def test_metropolis():
    proposal = GaussianRandomWalk(
        C = PRIOR_COV,
        scaling = 0.5
    )

    samples = sample(
        posteriors=POSTERIOR,
        proposal=proposal,
        iterations=(ITERATIONS - BURN_IN) * CHAINS // 2,
        # arviz needs 2 chains minimum?
        n_chains=2,
        force_sequential=False
    )

    idata = to_inference_data(samples, burnin=BURN_IN)
    print(summary(idata))
    return idata


if __name__ == "__main__":
    print(f"Running DREAM {ITERATIONS} samples x {CHAINS} chains")
    dream_idata = test_DREAM()
    print(f"Running DREAM(Z) {ITERATIONS} samples x {CHAINS} chains")
    dreamz_idata = test_DREAMZ()
    print(f"Running DREAM(Z) {ITERATIONS * CHAINS // 2} samples x 2 chains")
    dreamz_sequential_idata = test_DREAMZ_sequential()
    print(f"Running GRW {ITERATIONS * CHAINS // 2} samples x 2 chains")
    metropolis_idata = test_metropolis()