"""Synthetic stochastic LTI system generation for the flag-comparison experiments."""

import numpy as np


def simulate_lti(A, B, U, x0, noise_std, rng=None):
    """Simulate x_{t+1} = A x_t + B u_t + noise for t = 0..T-1.

    Parameters
    ----------
    A : (n, n) array
    B : (n, m) array
    U : (m, T) array of control inputs u_0..u_{T-1}
    x0 : (n,) array, initial condition
    noise_std : float, std of iid Gaussian process noise added at each step
    rng : np.random.Generator, only needed when noise_std is nonzero

    Returns
    -------
    X : (n, T + 1) array, states x_0..x_T
    """
    n = A.shape[0]
    T = U.shape[1]
    X = np.empty((n, T + 1))
    X[:, 0] = x0
    for t in range(T):
        X[:, t + 1] = A @ X[:, t] + B @ U[:, t]
        if noise_std:
            X[:, t + 1] += noise_std * rng.standard_normal(n)
    return X


def make_shared_controls(K, m, T, rng):
    """K control sequences of shape (m, T), to be reused identically across systems."""
    return [rng.standard_normal((m, T)) for _ in range(K)]


def simulate_dataset(A, B, controls, noise_std, rng):
    """Simulate one trajectory per control sequence, with independent ICs/noise.

    Parameters
    ----------
    A : (n, n) array
    B : (n, m) array
    controls : list of (m, T) arrays, shared across systems being compared
    noise_std : float, same across systems being compared
    rng : np.random.Generator

    Returns
    -------
    list of (n, T + 1) arrays, one trajectory per control sequence
    """
    n = A.shape[0]
    trajectories = []
    for U in controls:
        x0 = rng.standard_normal(n)
        trajectories.append(simulate_lti(A, B, U, x0, noise_std, rng))
    return trajectories


def forced_response(A, B, U):
    """Control-driven part of the trajectory: the same recursion from rest, with no noise.

    This is the component that is a function of the control input alone -- everything else
    (initial-condition transient, process noise) is internal variability.
    """
    return simulate_lti(A, B, U, np.zeros(A.shape[0]), 0.0)


def simulate_ensemble(A, B, U, n_members, noise_std, rng):
    """Ensemble sharing one control, each member with its own initial condition and noise.

    Returns
    -------
    forced : (n, T + 1) array, the control-driven response shared by every member
    internal : list of (n, T + 1) arrays, each member's initial-condition transient plus its
        noise response

    The full signal for member i is forced + internal[i]. It is left unassembled so that large
    ensembles cost n_members + 1 recursions and arrays rather than twice that.
    """
    zero_control = np.zeros_like(U)
    internal = [
        simulate_lti(A, B, zero_control, rng.standard_normal(A.shape[0]), noise_std, rng)
        for _ in range(n_members)
    ]
    return forced_response(A, B, U), internal


def add_observation_noise(trajectories, obs_noise_std, rng):
    """Corrupt each trajectory's recorded snapshots with iid Gaussian measurement noise.

    Unlike the process noise already baked into simulate_dataset/simulate_lti (injected at every
    simulation step, so it propagates through the dynamics), observation noise is added directly
    to the recorded (n, T + 1) state arrays after simulation and does not feed back into future
    states -- it models sensor/measurement error rather than stochastic forcing.
    """
    return [X + obs_noise_std * rng.standard_normal(X.shape) for X in trajectories]
