#import pytest
from scipy.stats import multivariate_normal
import numpy as np
from arviz import summary

from tinyDA.proposal import DREAMZ, DREAM
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
MODEL = lambda params: np.array(-1 / 80  * (3 / np.exp(params[0]) + 1 / np.exp(params[1])))
POSTERIOR = Posterior(PRIOR, LIKELIHOOD, MODEL)

ITERATIONS = 10000
BURN_IN = 2000
CHAINS = 4
ADAPTION_PERIOD = 500

M0 = 1000 # initial archive size
DELTA = 5 # number of sample pairs to use to compute jump
NCR = 2

def test_DREAMZ():
    proposal = DREAMZ(
        M0=M0,
        delta=DELTA,
        nCR=NCR,
        period=ADAPTION_PERIOD,
    )

    samples = sample(
        posteriors=POSTERIOR,
        proposal=proposal,
        iterations=ITERATIONS,
        n_chains=CHAINS,
        force_sequential=False
    )

    idata = to_inference_data(samples)
    print(summary(idata))

def test_DREAM():
    proposal = DREAM(
        M0=M0,
        delta=DELTA,
        nCR=NCR,
        period=ADAPTION_PERIOD
    )

    samples = sample(
        posteriors=POSTERIOR,
        proposal=proposal,
        iterations=ITERATIONS,
        n_chains=CHAINS,
        force_sequential=False
    )

    idata = to_inference_data(samples)
    print(summary(idata))


if __name__ == "__main__":
    test_DREAMZ()
    test_DREAM()