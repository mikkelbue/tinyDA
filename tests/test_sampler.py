import pytest

import numpy as np
from scipy.stats import multivariate_normal

from tinyDA.sampler import sample
from tinyDA.posterior import Posterior
from tinyDA.distributions import DefaultGaussianLogLike
from tinyDA.proposal import GaussianRandomWalk
from tinyDA.link import Link

# --------------------------------------------------------------------------------------------
np.random.seed(21)


# --------------------------------------------------------------------------------------------
# Simple forward models
def forward_fine(theta):
    return np.array(theta)


def forward_medium(theta):
    return 0.95 * np.array(theta)


def forward_coarse(theta):
    return 0.9 * np.array(theta)


# --------------------------------------------------------------------------------------------
@pytest.fixture
def prior():
    return multivariate_normal(mean=np.zeros(2), cov=np.eye(2))


@pytest.fixture
def data():
    theta_true = np.array([0.1, -0.2])
    return forward_fine(theta_true) + 0.05 * np.random.randn(2)


@pytest.fixture
def posteriors(prior, data):
    lik_fine = DefaultGaussianLogLike(data, covariance=0.05 * np.eye(2))
    lik_medium = DefaultGaussianLogLike(data, covariance=0.1 * np.eye(2))
    lik_coarse = DefaultGaussianLogLike(data, covariance=0.2 * np.eye(2))

    return {
        "fine": Posterior(prior, lik_fine, model=forward_fine),
        "medium": Posterior(prior, lik_medium, model=forward_medium),
        "coarse": Posterior(prior, lik_coarse, model=forward_coarse),
    }


# --------------------------------------------------------------------------------------------
# check chain content
def check_chain_content(out, n_chains, iterations, chain_name, is_coarse):

    for i in range(n_chains):
        key = f"{chain_name}{i}"
        assert key in out
        chain = out[key]
        if is_coarse:
            assert len(chain) == iterations * 2
        else:
            assert len(chain) == iterations + 1
        assert isinstance(chain, list)
        assert all(isinstance(link, Link) for link in chain)


# --------------------------------------------------------------------------------------------
# MH
def validate_sample_dict(out, expected_sampler, n_chains, iterations):
    # Check metadata (info)
    assert out["sampler"] == expected_sampler
    assert out["n_chains"] == n_chains
    assert out["iterations"] == iterations + 1

    # Check chain contents (chains)
    check_chain_content(out, n_chains, iterations, "chain_", False)


# --------------------------------------------------------------------------------------------
# DA
def validate_sample_dict_da(
    out, expected_sampler, n_chains, iterations, subchain_length
):
    # Check metadata (info)
    assert out["sampler"] == expected_sampler
    assert out["n_chains"] == n_chains
    assert out["iterations"] == iterations + 1
    assert out["subchain_length"] == subchain_length

    # Check chain contents (chains)
    # Validate fine samples
    check_chain_content(out, n_chains, iterations, "chain_fine_", False)
    # Validate coarse samples
    check_chain_content(out, n_chains, iterations, "chain_coarse_", True)


# --------------------------------------------------------------------------------------------
# MLDA
def validate_sample_dict_mlda(
    out, expected_sampler, n_chains, iterations, subchain_lengths
):
    # Check metadata (info)
    assert out["sampler"] == expected_sampler
    assert out["n_chains"] == n_chains
    assert out["iterations"] == iterations + 1
    # assert out["levels"] == levels
    assert out["subchain_lengths"] == subchain_lengths

    # Check chain contents (chains)
    # Validate fine samples
    for j in range(n_chains):
        for i in range(3):
            key = f"chain_l{i}_{j}"
            assert key in out
            chain = out[key]
            if i == 0:
                assert len(chain) == iterations * 4
            if i == 1:
                assert len(chain) == iterations * 2
            if i == 2:
                assert len(chain) == iterations + 1
            assert isinstance(chain, list)
            assert all(isinstance(link, Link) for link in chain)


# --------------------------------------------------------------------------------------------
# sanity checks


def extract_parameters(chain_dict, level="fine", chain_id=0, burnin=0):
    sampler = chain_dict["sampler"]

    if sampler == "MH":
        key = f"chain_{chain_id}"

    elif sampler == "DA":
        if level == "fine":
            key = f"chain_fine_{chain_id}"
        elif level == "coarse":
            key = f"chain_coarse_{chain_id}"
        elif level == "promoted":
            key = f"chain_promoted_coarse_{chain_id}"
        else:
            raise ValueError(f"Unknown level {level}")

    elif sampler == "MLDA":
        if isinstance(level, int):
            key = f"chain_l{level}_{chain_id}"
        elif level == "promoted":
            raise ValueError("Specify level index for MLDA promoted chains")
        else:
            raise ValueError(f"Invalid level {level}")

    else:
        raise ValueError(f"Unknown sampler {sampler}")

    chain = chain_dict[key][burnin:]
    return np.array([link.parameters for link in chain])


def assert_mean_close(samples, target, atol=0.1):
    mean = samples.mean(axis=0)
    assert np.allclose(mean, target, atol=atol), f"Mean {mean} not close to {target}"


def assert_variance_nonzero(samples, min_std=1e-3):
    std = samples.std(axis=0)
    assert np.all(std > min_std), f"Std too small: {std}"


def assert_chain_moves(samples):
    diffs = np.diff(samples, axis=0)
    assert np.any(np.abs(diffs) > 0), "Chain did not move"


# test chain is not too correlated
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
# test sample() with n_levels == 1 => sampler==MH


@pytest.mark.parametrize(
    "proposal, iterations, n_chains",
    [
        (GaussianRandomWalk(C=np.eye(2) * 0.1, adaptive=False), 100, 3),
        (GaussianRandomWalk(C=np.eye(2) * 0.1, adaptive=False), 0, 3),
    ],
)
def test_mh_sampling(posteriors, proposal, iterations, n_chains):

    result_mh = sample(
        posteriors=posteriors["fine"],
        proposal=proposal,
        iterations=iterations,
        n_chains=n_chains,
        force_sequential=True,  # ensure Ray is never used
    )

    validate_sample_dict(result_mh, "MH", n_chains=n_chains, iterations=iterations)


# --------------------------------------------------------------------------------------------
# test sample() with n_levels == 2 => sampler==DA


@pytest.mark.parametrize(
    "proposal_da, iterations, n_chains",
    [
        (GaussianRandomWalk(C=np.eye(2) * 0.1, adaptive=False), 100, 3),
        (GaussianRandomWalk(C=np.eye(2) * 0.1, adaptive=False), 0, 3),
    ],
)
def test_da_sampling(posteriors, proposal_da, iterations, n_chains):

    result_da = sample(
        posteriors=[posteriors["coarse"], posteriors["fine"]],
        proposal=proposal_da,
        iterations=iterations,
        n_chains=n_chains,
        subchain_length=2,
        randomize_subchain_length=False,
        force_sequential=True,
        store_coarse_chain=True,
    )

    validate_sample_dict_da(
        result_da, "DA", n_chains=n_chains, iterations=iterations, subchain_length=2
    )


# --------------------------------------------------------------------------------------------
# test sample() with n_levels == 3 => sampler==MLDA


@pytest.mark.parametrize(
    "proposal_mlda, iterations, n_chains",
    [
        (GaussianRandomWalk(C=np.eye(2) * 0.1, adaptive=False), 100, 3),
        (GaussianRandomWalk(C=np.eye(2) * 0.1, adaptive=False), 0, 3),
    ],
)
def test_mlda_sampling(posteriors, proposal_mlda, iterations, n_chains):
    result_mlda = sample(
        posteriors=[posteriors["coarse"], posteriors["medium"], posteriors["fine"]],
        proposal=proposal_mlda,
        iterations=iterations,
        n_chains=n_chains,
        subchain_length=[2, 2],  # two corrections levels
        randomize_subchain_length=False,
        store_coarse_chain=True,
        force_sequential=True,
    )

    validate_sample_dict_mlda(
        result_mlda,
        "MLDA",
        n_chains=n_chains,
        iterations=iterations,
        subchain_lengths=[2, 2],
    )


# --------------------------------------------------------------------------------------------
# run sanity checks
def run_sanity_checks(samples, target, atol):

    assert_mean_close(samples, target, atol)
    assert_variance_nonzero(samples)
    assert_chain_moves(samples)

    assert_not_too_correlated(samples, max_lag1_corr=0.95)
    assert_stationary(samples)


# --------------------------------------------------------------------------------------------
# sanity checks for MH
def test_mh_sampling_statistics(posteriors):
    result = sample(
        posteriors=posteriors["fine"],
        proposal=GaussianRandomWalk(C=np.eye(2) * 0.1, adaptive=False),
        iterations=1000,
        n_chains=1,
        force_sequential=True,
    )

    samples = extract_parameters(result, burnin=100)
    target = np.array([0.1, -0.2])

    run_sanity_checks(samples, target, 0.15)


# --------------------------------------------------------------------------------------------
# sanity checks for DA
def test_da_sampling_statistics(posteriors):
    result = sample(
        posteriors=[posteriors["coarse"], posteriors["fine"]],
        proposal=GaussianRandomWalk(C=np.eye(2) * 0.1, adaptive=False),
        iterations=500,
        n_chains=1,
        subchain_length=2,
        randomize_subchain_length=False,
        store_coarse_chain=True,
        force_sequential=True,
    )

    samples = extract_parameters(result, burnin=100)
    target = np.array([0.1, -0.2])

    run_sanity_checks(samples, target, 0.15)


# --------------------------------------------------------------------------------------------
# sanity checks for MLDA
def test_mlda_sampling_statistics(posteriors):
    result = sample(
        posteriors=[posteriors["coarse"], posteriors["medium"], posteriors["fine"]],
        proposal=GaussianRandomWalk(C=np.eye(2) * 0.1, adaptive=False),
        iterations=500,
        n_chains=1,
        subchain_length=[2, 2],
        randomize_subchain_length=False,
        store_coarse_chain=True,
        force_sequential=True,
    )

    samples = extract_parameters(result, level=2, burnin=100)
    target = np.array([0.1, -0.2])

    run_sanity_checks(samples, target, 0.2)
