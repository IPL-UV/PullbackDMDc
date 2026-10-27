"""Checks of synthetic_generator.py against the theory it is built from."""

import numpy as np

from ablations import M
from support import cheap_ensemble
from synthetic_generator import make_setup


def test_data_is_the_sum_of_its_uncentered_parts():
    """data = forced + process noise response + obs_noise, with nothing centered: the forced response keeps its mean."""
    ens = cheap_ensemble(3)
    measured_only = cheap_ensemble(3, variances=(1, 0, 0, 0, 0.1))  # no process noise: data - obs_noise = forced
    assert np.abs(measured_only["data"] - measured_only["obs_noise"] - ens["forced"]).max() < 1e-12
    clean = cheap_ensemble(variances=(1, 0, 0, 0, 0))  # no noise: data = forced exactly
    assert np.allclose(clean["data"][0], ens["forced"]) and np.abs(ens["forced"].mean(0)).max() > 1e-2
    assert np.allclose(ens["data_mean"], ens["data"].mean(axis=1))

def test_variances_match_the_components():
    """Modal variances of the internal variability and the measurement-noise variance, on many members.

    The slow mode is left out: over the 165-yr record its variance about the record mean is biased low
    (about 0.8 of the stationary value at tau1 = 20 yr), which is the record's, not the generator's.
    """
    ens = cheap_ensemble(400, variances=(1, 1, 1, 0.9, 0.1))
    process = ens["data"] - ens["forced"] - ens["obs_noise"]  # the response to the process noise
    modal = (process @ ens["W_inv"].T).var(axis=1).mean(axis=0)
    assert np.allclose(modal[1:] / ens["modal_var"][1:], 1, atol=0.03), modal / ens["modal_var"]
    assert abs(ens["obs_noise"].var() / ens["sigma_eps"] ** 2 - 1) < 0.01
    assert abs(ens["snr"] - 1 / 3) < 1e-12


def test_unit_norm_quasi_equilibrium():
    ens = cheap_ensemble()
    gain = np.linalg.solve(np.eye(M) - ens["A"], ens["B"])
    assert abs(abs(ens["y0"]) * np.linalg.norm(gain) - 1) < 1e-9


def test_overlap():
    """alpha rotates only the complement modes toward w1: q_k(alpha) = alpha w1 + sqrt(1 - alpha^2) q_k."""
    W0, _, B0 = make_setup(10)[:3]
    assert np.array_equal(B0, np.ones(M))  # the uniform forcing pattern b0, unscaled
    # alpha = 0 is the default W: the noise modes are the orthonormal complement of the structured patterns
    assert np.allclose(W0[:, :3].T @ W0[:, 3:], 0) and np.allclose(W0[:, 3:].T @ W0[:, 3:], np.eye(M - 3))
    assert np.allclose(np.linalg.norm(W0, axis=0), 1)
    phi = np.linspace(-np.pi / 2, np.pi / 2, M)
    assert np.allclose(W0[:, 1], np.sinc(2 * phi) / np.linalg.norm(np.sinc(2 * phi)))
    for alpha in (0.3, 0.9, 0.99):
        W, W_inv, B = make_setup(10, alpha=alpha)[:3]
        assert np.array_equal(W[:, :3], W0[:, :3]) and np.array_equal(B, B0)  # slow and fast patterns, b0 fixed
        assert np.allclose(np.linalg.norm(W, axis=0), 1)
        assert np.allclose(W[:, 0] @ W[:, 3:], alpha)
        assert np.allclose(W[:, 3:], alpha * W0[:, :1] + np.sqrt(1 - alpha ** 2) * W0[:, 3:])
        assert np.allclose(W_inv @ W, np.eye(M))

def test_noise_is_seeded_by_rng():
    a, b, c = (cheap_ensemble(2, rng=seed)["data"] for seed in (1, 1, 2))
    assert np.array_equal(a, b) and not np.allclose(a, c)
