import numpy as np

def lat_grid(n=20):
    return np.linspace(-np.pi / 2, np.pi / 2, n)          # radians, poles included

# Zonal-mean CO2 radiative forcing (W m^-2) at |latitude| = 0, 5, ..., 90 deg, digitized from the red CO2 curve
# of figures/reference/co2_zonal_forcing.png (clear-sky tropopause forcing, 1750-2005, hemispherically
# symmetric atmosphere); the north and south branches agree to 0.003 W m^-2 and are averaged.
CO2_ZONAL_LATS = np.arange(0, 91, 5)
CO2_ZONAL_FORCING = np.array([2.498, 2.498, 2.484, 2.470, 2.437, 2.395, 2.339, 2.274, 2.186, 2.078,
                              2.012, 1.950, 1.880, 1.801, 1.720, 1.646, 1.590, 1.556, 1.544])

def co2_forcing_pattern(phi):
    return np.interp(np.abs(np.rad2deg(phi)), CO2_ZONAL_LATS, CO2_ZONAL_FORCING)

def raw_patterns(n=20):
    phi = lat_grid(n)
    slow  = np.cos(phi - np.pi / 4)
    fast1 = np.sinc(2 * phi)
    fast2 = (0.5 - np.cos(10 * phi)) / (0.5 + np.abs(10 * phi))
    b     = co2_forcing_pattern(phi)
    return phi, slow, fast1, fast2, b

def make_W(n=20, seed=22):
    """W (unit-norm columns), W_inv, unit-norm B, grid (rad).
    Modes 4..n: Haar-random orthonormal basis of the complement of the 3
    structured modes (seed=None draws a fresh one)."""
    phi, slow, f1, f2, b = raw_patterns(n)
    unit = lambda v: v / np.linalg.norm(v)
    S = np.column_stack([unit(slow), unit(f1), unit(f2)])
    if np.linalg.matrix_rank(S) < 3:
        raise ValueError("structured patterns are linearly dependent on this grid")
    Q, _ = np.linalg.qr(S, mode="complete")
    Qp = Q[:, 3:]
    rng = np.random.default_rng(seed)
    R, T = np.linalg.qr(rng.standard_normal((n - 3, n - 3)))
    Qp = Qp @ (R * np.sign(np.diag(T)))
    W = np.column_stack([S, Qp])
    W_inv = np.vstack([np.linalg.pinv(S), Qp.T])
    return W, W_inv, unit(b), phi

