"""Checks of synthetic_generator.py against the theory it is built from."""

import numpy as np

from ablations import M
from support import cheap_ensemble
from synthetic_generator import make_setup


def test_data_is_the_sum_of_its_uncentered_parts():
    """data = forced + internal + obs_noise, with nothing centered: the forced response keeps its record mean."""
    ens = cheap_ensemble(3)
    assert np.abs(ens["data"] - (ens["forced"] + ens["internal"] + ens["obs_noise"])).max() < 1e-12
    clean = cheap_ensemble(split=(1, 0, 0, 0, 0))  # no noise: data = forced exactly
    assert np.allclose(clean["data"][0], ens["forced"]) and np.abs(ens["forced"].mean(0)).max() > 1e-2
    assert np.allclose(ens["data_mean"], ens["data"].mean(axis=1))

def test_variances_match_the_split():
    """Modal variances of the internal variability and the measurement-noise variance, on many members.

    The slow mode is left out: over the 165-yr record its variance about the record mean is biased low
    (about 0.8 of the stationary value at tau1 = 20 yr), which is the record's, not the generator's.
    """
    ens = cheap_ensemble(400, split=(1, 1, 1, 0.9, 0.1))
    modal = (ens["internal"] @ ens["W_inv"].T).var(axis=1).mean(axis=0)
    assert np.allclose(modal[1:] / ens["modal_var"][1:], 1, atol=0.03), modal / ens["modal_var"]
    assert abs(ens["obs_noise"].var() / ens["obs_std"] ** 2 - 1) < 0.01
    assert abs(ens["snr"] - 1 / 3) < 1e-12


def test_unit_norm_quasi_equilibrium():
    ens = cheap_ensemble()
    gain = np.linalg.solve(np.eye(M) - ens["A"], ens["B"])
    assert abs(abs(ens["y0"]) * np.linalg.norm(gain) - 1) < 1e-9


def test_forcing_rotation():
    """b_rot rotates the flat B toward w1 by that fraction of the angle between them; W does not move."""
    flat = np.ones(M) / np.sqrt(M)
    W0, _, B0 = make_setup(10)[:3]
    theta0 = np.arccos(flat @ W0[:, 0])
    assert np.allclose(B0, flat)
    for b_rot in (0.25, 0.5, 1.0):
        W, W_inv, B = make_setup(10, b_rot=b_rot)[:3]
        assert np.array_equal(W, W0)
        assert abs(np.linalg.norm(B) - 1) < 1e-12
        assert abs(np.arccos(np.clip(B @ W[:, 0], -1, 1)) - (1 - b_rot) * theta0) < 1e-7
    assert np.allclose(W_inv @ B, np.eye(M)[0])  # b_rot = 1: the forcing drives the slow mode alone
    # the fast pair keeps its defined shapes and the noise modes stay the orthogonal complement
    phi = np.linspace(-np.pi / 2, np.pi / 2, M)
    assert np.allclose(W0[:, 1], np.sinc(2 * phi) / np.linalg.norm(np.sinc(2 * phi)))
    assert np.allclose(W0[:, :3].T @ W0[:, 3:], 0)

def test_noise_is_seeded_by_rng():
    a, b, c = (cheap_ensemble(2, rng=seed)["data"] for seed in (1, 1, 2))
    assert np.array_equal(a, b) and not np.allclose(a, c)
