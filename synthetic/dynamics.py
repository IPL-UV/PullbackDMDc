"""Synthetic stochastic LTI system generation for the flag-comparison experiments."""

import numpy as np


def simulate_lti(A, B, U, x0, noise_std, rng):
    """Simulate x_{t+1} = A x_t + B u_t + noise for t = 0..T-1.

    Parameters
    ----------
    A : (n, n) array
    B : (n, m) array
    U : (m, T) array of control inputs u_0..u_{T-1}
    x0 : (n,) array, initial condition
    noise_std : float, std of iid Gaussian process noise added at each step
    rng : np.random.Generator

    Returns
    -------
    X : (n, T + 1) array, states x_0..x_T
    """
    n = A.shape[0]
    T = U.shape[1]
    X = np.empty((n, T + 1))
    X[:, 0] = x0
    for t in range(T):
        noise = noise_std * rng.standard_normal(n)
        X[:, t + 1] = A @ X[:, t] + B @ U[:, t] + noise
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


def add_observation_noise(trajectories, obs_noise_std, rng):
    """Corrupt each trajectory's recorded snapshots with iid Gaussian measurement noise.

    Unlike the process noise already baked into simulate_dataset/simulate_lti (injected at every
    simulation step, so it propagates through the dynamics), observation noise is added directly
    to the recorded (n, T + 1) state arrays after simulation and does not feed back into future
    states -- it models sensor/measurement error rather than stochastic forcing.
    """
    return [X + obs_noise_std * rng.standard_normal(X.shape) for X in trajectories]
