"""Shared helpers for the synthetic experiment scripts."""

import numpy as np


def _orthonormalize(vectors):
    """Gram-Schmidt orthonormalize a sequence of vectors, in order, returning an array of
    orthonormal rows (each still recognizable as a refinement of the input in that slot)."""
    basis = []
    for v in vectors:
        w = np.asarray(v, dtype=float).copy()
        for b in basis:
            w -= (w @ b) * b
        basis.append(w / np.linalg.norm(w))
    return np.array(basis)


def canonical_modes(n):
    """The 3 structured mode shapes (as rows) for canonical_A's slow/pair block, plus the
    (n - 3) remaining orthonormal "noise" directions that complete the basis. All rows are
    mean-centered and the full set is orthonormal.

    - real_slow mode: large at small and large indices, small in the middle (U-shape).
    - pair mode 1: a bump in the middle (large value there).
    - pair mode 2: a dip in the middle (small/negative value there), distinct in shape from
      pair mode 1 so the two are linearly independent.
    """
    idx = np.arange(n, dtype=float)
    mid = (n - 1) / 2
    width = max(n * 0.15, 1.0)

    slow_shape = (idx - mid) ** 2
    z = (idx - mid) / width
    peak_shape = np.exp(-0.5 * z**2)
    dip_shape = -(1 - z**2) * np.exp(-0.5 * z**2)

    structured = [v - v.mean() for v in (slow_shape, peak_shape, dip_shape)]
    remaining = [np.eye(n)[i] for i in range(3, n)]
    return _orthonormalize(structured + remaining)


def canonical_A(real_slow, pair_modulus, pair_angle, real_fast):
    """Real form built from canonical_modes: 1 real slow mode, 1 complex-conjugate pair,
    and one or more additional real "noise" modes, each with the mode shapes described in
    canonical_modes. `real_fast` may be a scalar (single mode, 4x4 result) or a sequence of
    eigenvalues (one mode per entry, (3 + len(real_fast))x(...) result)."""
    c, s = pair_modulus * np.cos(pair_angle), pair_modulus * np.sin(pair_angle)
    fast_eigs = np.atleast_1d(real_fast).astype(float)
    n = 3 + len(fast_eigs)

    Lambda = np.zeros((n, n))
    Lambda[0, 0] = real_slow
    Lambda[1, 1], Lambda[1, 2] = c, s
    Lambda[2, 1], Lambda[2, 2] = -s, c
    Lambda[3:, 3:] = np.diag(fast_eigs)

    V = canonical_modes(n).T  # modes as columns, orthonormal
    return V @ Lambda @ V.T

def canonical_B(n, mean=0.1, std=0.02, seed=21):
    """Deterministic n x 1 control matrix: a noisy positive signal centered at `mean`,
    drawn once from a fixed seed so it is reproducible across runs."""
    b = np.random.default_rng(seed).normal(mean, std, size=n)
    b = np.clip(b, mean / 10, None)
    return b.reshape(n, 1)


# 17 fast/noise eigenvalues for the 20-dim canonical form: drawn once from a fixed seed so
# that CANONICAL_A_20_PARAMS (and any A built from it) is reproducible across runs.
NOISE_EIG_RANGE = (0.0, 0.1)
NOISE_EIGS_17 = np.random.default_rng(20).uniform(*NOISE_EIG_RANGE, size=17)

CANONICAL_A_20_PARAMS = dict(
    real_slow=0.95, pair_modulus=0.80, pair_angle=np.deg2rad(25), real_fast=NOISE_EIGS_17
)

CANONICAL_B_20 = canonical_B(20)

def build_modes_only_systems(n_systems, n_states, rng):
    """n_systems sharing ONE fixed eigenvalue set (MODES_ONLY_LAMBDA_PARAMS), each rotated by its
    own independently-drawn random orthogonal matrix -- isolates mode/eigenvector differences from
    eigenvalue differences (the same per-system-rotation construction as synthetic_n_systems.py's
    build_systems, but with a single shared Lambda instead of per-system eigenvalue params).
    """
    Lambda = canonical_A(**MODES_ONLY_LAMBDA_PARAMS)
    As = []
    for _ in range(n_systems):
        Q = random_orthogonal(n_states, rng)
        As.append(Q @ Lambda @ Q.T)
    B = rng.standard_normal((n_states, 1))
    return As, B


