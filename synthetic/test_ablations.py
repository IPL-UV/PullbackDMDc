"""Tests for the ablation data generators. Run with `python test_ablations.py` (names are pytest-compatible)."""

from dataclasses import replace

import numpy as np

from ablations import (
    B_HAT,
    BASE_OVERLAP,
    HISTORY,
    MODE_OVERLAPS,
    N,
    PARTIAL_SNR_COMPONENTS,
    PARTIAL_SNR_FACTORS,
    PERIOD_P_YR,
    REFERENCE,
    REFERENCE_BUDGET,
    SLOW_TIMESCALES_YR,
    TOTAL_SNRS,
    UNIT_FORCING_REFERENCE,
    W_INV_REF,
    W_REF,
    decay_time,
    eigenvalue,
    forced_response,
    forced_variance,
    make_dataset,
    modal_coordinates,
    pair_plane_overlap,
    partial_snr_sweep,
    slow_timescale_sweep,
    spatial_overlap_sweep,
    total_snr_sweep,
)

SMALL = dict(n_realizations=2)


def all_sweeps(**kwargs):
    return (
        total_snr_sweep(**kwargs)
        + [ds for c in PARTIAL_SNR_COMPONENTS for ds in partial_snr_sweep(c, **kwargs)]
        + slow_timescale_sweep(hold="snr", **kwargs) + slow_timescale_sweep(hold="modal_variance", **kwargs)
        + spatial_overlap_sweep(**kwargs)
    )


def test_patterns():
    assert np.allclose(np.linalg.norm(W_REF, axis=0), 1)
    assert np.abs(W_INV_REF @ W_REF - np.eye(len(W_REF))).max() < 1e-12
    assert np.abs(W_REF[:, :3].T @ W_REF[:, 3:]).max() < 1e-12
    gram = np.column_stack([W_REF[:, :3], B_HAT])
    gram = gram.T @ gram
    expected = {(0, 1): 0.396, (0, 2): 0.329, (1, 2): 0.278, (3, 0): 0.650, (3, 1): 0.489, (3, 2): 0.478}
    for (i, j), value in expected.items():
        assert abs(gram[i, j] - value) < 5e-4, (i, j, gram[i, j])


def test_eigenvalues():
    for ds in all_sweeps(**SMALL):
        s = ds.system
        got = np.sort_complex(np.linalg.eigvals(s.A))
        assert np.abs(got - np.sort_complex(s.eigvals)).max() < 1e-10, (ds.study, ds.param_value)
        assert np.abs(s.eigvals).max() < 1
    assert abs(decay_time(REFERENCE.rho) - 12 * 2) < 1e-9
    assert abs(2 * np.pi / REFERENCE.theta - 12 * PERIOD_P_YR) < 1e-9
    # first zero of the pair autocorrelation rho^k cos(theta k) at one year
    assert abs(np.cos(REFERENCE.theta * 12)) < 1e-12


def test_slow_timescale():
    for hold in ("snr", "modal_variance"):
        for ds, tau in zip(slow_timescale_sweep(hold=hold, **SMALL), SLOW_TIMESCALES_YR):
            assert abs(decay_time(ds.system.lam1) / 12 - tau) < 1e-9
            assert ds.system.rho == REFERENCE.rho and ds.system.theta == REFERENCE.theta
            if hold == "snr":
                assert abs(ds.snr - 1 / 3) < 1e-12
            else:
                assert all(getattr(ds.system, k) == v for k, v in REFERENCE_BUDGET.items())


def test_structure():
    for ds in all_sweeps(**SMALL):
        s = ds.system
        assert ds.forced.shape == (N, 20) and ds.internal.shape == (SMALL["n_realizations"], N, 20)
        assert ds.y.shape == (s.spinup + N,)
        assert np.abs(ds.data - (ds.forced + ds.internal)).max() < 1e-12
        y = ds.y[s.spinup:]
        assert np.abs(ds.forced[1:] - (ds.forced[:-1] @ s.A.T + np.outer(y[1:], s.b))).max() < 1e-10
        assert abs(y.mean()) < 1e-12
        long_forcings, short_forcings, transition_time = ds.forcings()
        assert transition_time == HISTORY and long_forcings.shape == (HISTORY + N, 1)
        assert np.all(short_forcings[:, 0] == y) and np.all(long_forcings[:, 0] == ds.y[s.spinup - HISTORY:])


def test_spinup():
    for ds in all_sweeps(**SMALL):
        s = ds.system
        assert s.spinup >= HISTORY
        assert np.exp(-s.spinup / decay_time(s.eigvals).max()) < 1e-15, (ds.study, ds.param_value)


def test_forced_variance_scale():
    V_f = forced_variance(forced_response(UNIT_FORCING_REFERENCE)[1])
    fast = replace(UNIT_FORCING_REFERENCE, lam1=eigenvalue(12 * 0.84))
    V_f_fast = forced_variance(forced_response(fast)[1])
    print(f"    unit-forcing V_f: {V_f:.4g} at tau_1 = 20 yr, {V_f_fast:.4g} at 0.84 yr (AR6 CO2, 1915-2014)")
    assert abs(V_f / 1.284e3 - 1) < 1e-3 and abs(V_f_fast / 7.234 - 1) < 1e-3
    assert abs(forced_variance(forced_response(REFERENCE)[1]) - 20) < 1e-9


def test_modal_variances():
    ds = make_dataset(REFERENCE, lambda V_f: dict(REFERENCE_BUDGET), n_realizations=500)
    z = modal_coordinates(ds, ds.internal)
    empirical = (z**2).mean(axis=(0, 1))
    assert np.abs(empirical / ds.system.modal_variances - 1).max() < 0.1, empirical / ds.system.modal_variances
    noise = modal_coordinates(ds, ds.internal[:, 1:] - ds.internal[:, :-1] @ ds.system.A.T)
    assert np.abs(noise.var(axis=(0, 1)) / ds.system.noise_variances - 1).max() < 0.02
    cross = np.corrcoef(noise.reshape(-1, noise.shape[-1]).T) - np.eye(noise.shape[-1])
    assert np.abs(cross).max() < 0.01


def test_snr_levels():
    for ds, snr in zip(total_snr_sweep(**SMALL), TOTAL_SNRS):
        assert abs(ds.snr / snr - 1) < 1e-12
    for component, key in PARTIAL_SNR_COMPONENTS.items():
        for ds, factor in zip(partial_snr_sweep(component, **SMALL), PARTIAL_SNR_FACTORS):
            for k, v in REFERENCE_BUDGET.items():
                assert abs(getattr(ds.system, k) - (v * factor if k == key else v)) < 1e-12


def test_spatial_overlap():
    sweep = spatial_overlap_sweep(**SMALL)
    for ds, overlap in zip(sweep, MODE_OVERLAPS):
        W = ds.system.W
        assert abs(pair_plane_overlap(W) - overlap) < 1e-12, overlap
        assert np.allclose(np.linalg.norm(W, axis=0), 1)
        assert np.abs(ds.system.W_inv @ W - np.eye(len(W))).max() < 1e-10
        assert np.all(W[:, 1:] == W_REF[:, 1:])
        assert abs(ds.snr - 1 / 3) < 1e-12
    base = sweep[MODE_OVERLAPS.index(BASE_OVERLAP)].system
    assert np.abs(base.W - W_REF).max() < 1e-12 and np.abs(base.W_inv - W_INV_REF).max() < 1e-10


def test_common_random_numbers():
    sweep = total_snr_sweep(**SMALL)
    ref = sweep[0].internal / np.sqrt(sweep[0].system.s1_sq)
    for ds in sweep[1:]:
        assert np.abs(ds.internal / np.sqrt(ds.system.s1_sq) - ref).max() < 1e-10, ds.param_value


def main():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"  passed  {name}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
