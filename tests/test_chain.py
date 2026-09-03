import pytest

import numpy as np
from scipy.stats import multivariate_normal

from tinyDA.chain import Chain, DAChain, MLDAChain
from tinyDA.posterior import Posterior
from tinyDA.proposal import GaussianRandomWalk
from tinyDA.proposal import MLDA
from tinyDA.link import Link
from tinyDA.distributions import DefaultGaussianLogLike
from tinyDA.distributions import AdaptiveGaussianLogLike

# --------------------------------------------------------------------------------------------
np.random.seed(21)


# --------------------------------------------------------------------------------------------
# Simple forward model: identity map
def forward_model(theta):
    return np.array(theta)


# Coarse forward model = identity but scaled (simplest possible coarse model)
def forward_model_coarse(theta):
    return 0.9 * np.array(theta)


# Model for third layer
def forward_model_layer3(theta):
    return 0.8 * np.array(theta)


# --------------------------------------------------------------------------------------------
# Prior: 2D Gaussian N(0, I)
prior_mu = np.zeros(2)
prior_cov = np.eye(2)
prior = multivariate_normal(mean=prior_mu, cov=prior_cov)

# Synthetic data for likelihood
true_params = np.array([0.2, -0.3])
data = forward_model(true_params) + 0.05 * np.random.randn(2)
likelihood = DefaultGaussianLogLike(data, covariance=0.05 * np.eye(2))
likelihood_coarse = DefaultGaussianLogLike(data, covariance=0.2 * np.eye(2))
likelihood_layer3 = DefaultGaussianLogLike(data, covariance=0.25 * np.eye(2))

likelihood_adaptive = AdaptiveGaussianLogLike(data, covariance=0.05 * np.eye(2))
likelihood_adaptive_coarse = AdaptiveGaussianLogLike(data, covariance=0.2 * np.eye(2))
likelihood_adaptive_layer3 = AdaptiveGaussianLogLike(data, covariance=0.25 * np.eye(2))

# Posterior object
posterior = Posterior(prior, likelihood, model=forward_model)
posterior_fine = posterior
posterior_coarse = Posterior(prior, likelihood_coarse, model=forward_model_coarse)
posterior_layer3 = Posterior(prior, likelihood_layer3, model=forward_model_layer3)
posteriors3 = [posterior_layer3, posterior_coarse, posterior_fine]

posterior_adaptive_fine = Posterior(prior, likelihood_adaptive, model=forward_model)
posterior_adaptive_coarse = Posterior(
    prior, likelihood_adaptive_coarse, model=forward_model_coarse
)
posterior_adaptive_layer3 = Posterior(
    prior, likelihood_adaptive_layer3, model=forward_model_layer3
)
posteriors_adaptive_3layers = [
    posterior_adaptive_layer3,
    posterior_adaptive_coarse,
    posterior_adaptive_fine,
]

proposal = GaussianRandomWalk(C=np.eye(2) * 0.1, adaptive=False)
proposal_da = GaussianRandomWalk(C=np.eye(2) * 0.1, adaptive=False)
proposal_mlda_base = GaussianRandomWalk(C=np.eye(2) * 0.1, adaptive=False)

# --------------------------------------------------------------------------------------------
# sanity checks


# test mean is correct
def assert_mean_close(samples, target, atol=0.1):
    mean = samples.mean(axis=0)
    assert np.allclose(mean, target, atol=atol), f"Mean {mean} not close to {target}"


# test variance is not zero
def assert_variance_nonzero(samples, min_std=1e-3):
    std = samples.std(axis=0)
    assert np.all(std > min_std), f"Std too small: {std}"


# test chain moves
def assert_chain_moves(samples):
    diffs = np.diff(samples, axis=0)
    assert np.any(np.abs(diffs) > 0), "Chain did not move"


# test acceptance is reasonable
def assert_reasonable_acceptance(chain, min_rate=0.05, max_rate=0.8):
    rate = np.mean(chain.accepted)
    assert min_rate < rate < max_rate, f"Bad acceptance rate: {rate}"


# test acceptance is reasonable for coarse and fine chain
def assert_reasonable_acceptance_da(
    da_chain, min_coarse=0.05, max_coarse=0.9, min_fine=0.01, max_fine=0.8
):

    coarse_rate = np.mean(da_chain.accepted_coarse)
    fine_rate = np.mean(da_chain.accepted_fine)

    assert (
        min_coarse < coarse_rate < max_coarse
    ), f"Bad coarse acceptance: {coarse_rate}"

    assert min_fine < fine_rate < max_fine, f"Bad fine acceptance: {fine_rate}"


# test chain is not too corralated
def assert_not_too_correlated(samples, max_lag1_corr=0.95):
    x = samples[:, 0]  # check one dimension
    corr = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert abs(corr) < max_lag1_corr, f"Too correlated: {corr}"


# compare early and late samples
def assert_stationary(samples):
    n = len(samples)
    first = samples[: n // 2].mean(axis=0)
    second = samples[n // 2 :].mean(axis=0)

    assert np.allclose(
        first, second, atol=0.2
    ), f"Chain not stationary: {first} vs {second}"


# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "posterior",
    [
        (posterior),
    ],
)
def test_posterior_create_link(posterior):

    theta0 = np.array([0.1, 0.1])
    link = posterior.create_link(theta0)

    assert isinstance(link.parameters, np.ndarray)
    assert isinstance(link.model_output, np.ndarray)
    assert np.isfinite(link.posterior)


def assert_list_of_type(lst, typ, expected_length):
    assert isinstance(lst, list)
    assert all(isinstance(x, typ) for x in lst)

    if expected_length >= 0:
        assert len(lst) == expected_length

    if typ is Link:
        assert all(isinstance(x.parameters, np.ndarray) for x in lst)


# --------------------------------------------------------------------------------------------
# assertions constructor


def assert_constructor(chain, level, initial_parameters):

    # MH, DA and MLDA
    assert isinstance(chain.initial_parameters, np.ndarray)
    assert all(isinstance(y, float) for y in chain.initial_parameters)

    # MH and DA
    if level == "MH" or level == "DA":
        assert isinstance(chain.proposal, GaussianRandomWalk)

    # MH and MLDA
    if level == "MH" or level == "MLDA":
        assert isinstance(chain.posterior, Posterior)

        assert_list_of_type(chain.chain, Link, 1)
        assert_list_of_type(chain.accepted, bool, 1)

    # DA and MLDA
    if level == "DA" or level == "MLDA":
        assert isinstance(chain.subchain_length, int)

        if initial_parameters is not None:
            assert len(chain.initial_parameters) == len(initial_parameters)
            np.testing.assert_array_equal(chain.initial_parameters, initial_parameters)

        assert isinstance(chain.adaptive_error_model, (str, type(None)))
        assert isinstance(chain.store_coarse_chain, bool)

        if chain.adaptive_error_model:
            assert hasattr(chain, "bias")

    # DA
    if level == "DA":
        assert isinstance(chain.posterior_coarse, Posterior)
        assert isinstance(chain.posterior_fine, Posterior)

        assert isinstance(chain.randomize_subchain_length, bool)

        assert_list_of_type(chain.chain_coarse, Link, 1)
        assert_list_of_type(chain.accepted_coarse, bool, 1)
        assert_list_of_type(chain.is_coarse, bool, 1)
        assert_list_of_type(chain.promoted_coarse, Link, 0)
        assert_list_of_type(chain.subchain_lengths, int, 0)
        assert_list_of_type(chain.chain_fine, Link, 1)
        assert_list_of_type(chain.accepted_fine, bool, 1)

    # MLDA
    if level == "MLDA":
        assert isinstance(chain.level, int)
        assert isinstance(chain.proposal, MLDA)


# --------------------------------------------------------------------------------------------
# assertions sample


def assert_sample_chain(chain, level, iterations):

    # MH and DA
    if level == "MH" or level == "DA":
        assert isinstance(chain.proposal, GaussianRandomWalk)

    # MH and MLDA
    if level == "MH" or level == "MLDA":
        assert_list_of_type(chain.chain, Link, iterations + 1)
        assert_list_of_type(chain.accepted, bool, iterations + 1)

    # DA
    if level == "DA":
        assert chain.subchain_length == 2

        assert_list_of_type(chain.chain_coarse, Link, (iterations * 3) + 1)
        assert_list_of_type(chain.accepted_coarse, bool, (iterations * 3) + 1)
        assert_list_of_type(chain.is_coarse, bool, (iterations * 3) + 1)
        assert_list_of_type(chain.promoted_coarse, Link, -1)
        assert_list_of_type(chain.subchain_lengths, int, -1)
        assert_list_of_type(chain.chain_fine, Link, iterations + 1)
        assert_list_of_type(chain.accepted_fine, bool, iterations + 1)

    # MLDA
    if level == "MLDA":
        assert isinstance(chain.proposal, MLDA)


# --------------------------------------------------------------------------------------------
# test simple chain (MH)


@pytest.mark.parametrize(
    "posterior, proposal, initial_parameters",
    [
        (posterior, proposal, None),
        (posterior, proposal, np.array([0.4, 1.02])),
    ],
)
def test_chain_constructor(posterior, proposal, initial_parameters):

    chain = Chain(posterior, proposal, initial_parameters=initial_parameters)

    assert_constructor(chain, "MH", initial_parameters=initial_parameters)


# --------------------------------------------------------------------------------------------
# test sample for MH chain


@pytest.mark.parametrize(
    "iterations, progressbar",
    [
        (0, False),
        (50, False),
        (100, False),
    ],
)
def test_sample_for_MHchain(iterations, progressbar):

    chain = Chain(posterior, proposal, initial_parameters=None)
    chain.sample(iterations, progressbar=progressbar)

    assert_sample_chain(chain, "MH", iterations)


# --------------------------------------------------------------------------------------------
# Class DAChain


@pytest.mark.parametrize(
    "posterior_coarse, posterior_fine, proposal, subchain_length, randomize_subchain_length, initial_parameters, adaptive_error_model, store_coarse_chain",
    [
        (
            posterior_coarse,
            posterior_fine,
            proposal_da,
            2,
            False,
            np.array([0.6, 0.782]),
            None,
            False,
        ),
        (posterior_coarse, posterior_fine, proposal_da, 2, True, None, None, True),
        (
            posterior_adaptive_coarse,
            posterior_adaptive_fine,
            proposal_da,
            2,
            False,
            None,
            "state-dependent",
            False,
        ),
        (
            posterior_adaptive_coarse,
            posterior_adaptive_fine,
            proposal_da,
            2,
            False,
            np.array([0.6, 0.782]),
            "state-independent",
            False,
        ),
    ],
)
def test_DA_chain_constructor(
    posterior_coarse,
    posterior_fine,
    proposal,
    subchain_length,
    randomize_subchain_length,
    initial_parameters,
    adaptive_error_model,
    store_coarse_chain,
):

    da_chain = DAChain(
        posterior_coarse,
        posterior_fine,
        proposal_da,
        subchain_length=subchain_length,
        randomize_subchain_length=randomize_subchain_length,
        initial_parameters=initial_parameters,
        adaptive_error_model=adaptive_error_model,
        store_coarse_chain=store_coarse_chain,
    )

    assert_constructor(da_chain, "DA", initial_parameters=initial_parameters)


# --------------------------------------------------------------------------------------------
# test sample for DA chain


@pytest.mark.parametrize(
    "iterations, progressbar",
    [
        (0, False),
        (50, False),
        (100, False),
    ],
)
def test_sample_for_DAchain(iterations, progressbar):

    da_chain = DAChain(
        posterior_coarse,
        posterior_fine,
        proposal_da,
        subchain_length=2,
        randomize_subchain_length=True,
        initial_parameters=None,
        adaptive_error_model=None,
        store_coarse_chain=True,
    )

    da_chain.sample(iterations, progressbar=progressbar)

    assert_sample_chain(da_chain, "DA", iterations)


# --------------------------------------------------------------------------------------------
# Class MLDAChain


@pytest.mark.parametrize(
    "posteriors, proposal, subchain_lengths, initial_parameters, adaptive_error_model",
    [
        #([posterior_coarse, posterior_fine], proposal_mlda_base, [2], None, None),
        (posteriors3, proposal_mlda_base, [2, 2], None, None),
        (
            posteriors3,
            proposal_mlda_base,
            [2, 2],
            np.array([0.6, 0.782]),
            None,
        ),
        (
            posteriors_adaptive_3layers,
            proposal_mlda_base,
            [2, 2],
            None,
            "state-independent",
        ),
    ],
)
def test_MLDA_chain_constructor(
    posteriors, proposal, subchain_lengths, initial_parameters, adaptive_error_model
):

    mlda_chain = MLDAChain(
        posteriors=posteriors_adaptive_3layers,
        proposal=proposal,
        subchain_lengths=subchain_lengths,
        initial_parameters=initial_parameters,
        adaptive_error_model=adaptive_error_model,
    )

    assert_constructor(mlda_chain, "MLDA", initial_parameters=initial_parameters)


# --------------------------------------------------------------------------------------------
# test sample for MLDA chain


@pytest.mark.parametrize(
    "iterations, progressbar",
    [
        (0, False),
        (50, False),
        (100, False),
    ],
)
def test_sample_for_MLDAchain(iterations, progressbar):

    mlda_chain = MLDAChain(
        posteriors=posteriors3,
        proposal=proposal_mlda_base,
        subchain_lengths=[2, 2],
        initial_parameters=np.array([0.6, 0.782]),
        adaptive_error_model=None,
    )

    mlda_chain.sample(iterations, progressbar=progressbar)

    assert_sample_chain(mlda_chain, "MLDA", iterations)


# --------------------------------------------------------------------------------------------
# run sanity checks


def run_sanity_checks(chain, specific_chain, da):

    chain.sample(iterations=1000, progressbar=False)
    samples = np.array([link.parameters for link in specific_chain])
    samples = samples[100:]  # burn-in

    assert_chain_moves(samples)
    assert_variance_nonzero(samples)
    assert_mean_close(samples, target=true_params, atol=0.15)
    assert_not_too_correlated(samples)
    assert_stationary(samples)

    if da:
        assert_reasonable_acceptance_da(chain)
    else:
        assert_reasonable_acceptance(chain)


# --------------------------------------------------------------------------------------------
# sanity checks mh


def test_mh_sample_stats():

    chain = Chain(posterior, proposal, initial_parameters=None)

    run_sanity_checks(chain, chain.chain, False)


# --------------------------------------------------------------------------------------------
# sanity checks da
def test_da_sample_stats():

    da_chain = DAChain(
        posterior_coarse,
        posterior_fine,
        proposal_da,
        subchain_length=2,
        randomize_subchain_length=True,
        initial_parameters=None,
        adaptive_error_model=None,
        store_coarse_chain=True,
    )

    run_sanity_checks(da_chain, da_chain.chain_fine, True)


# --------------------------------------------------------------------------------------------
# sanity checks mlda
def test_mlda_sample_stats():

    mlda_chain = MLDAChain(
        posteriors=posteriors3,
        proposal=proposal_mlda_base,
        subchain_lengths=[2, 2],
        initial_parameters=None,
        adaptive_error_model=None,
    )

    run_sanity_checks(mlda_chain, mlda_chain.chain, False)
