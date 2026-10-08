"""Tests for ablation data and parameter sweeps."""

from dataclasses import replace

import numpy as np

from ablation_data import (
    M,
    BASE_B_OVERLAP,
    BASE_OVERLAP,
    W_INV_REF,
    W_REF,
    MIN_SPINUP,
    SPINUP_DECAY_TIMES,
    B_OVERLAPS,
    MODE_OVERLAPS,
    N,
    PARTIAL_SNR_COMPONENTS,
    PARTIAL_SNR_FACTORS,
    PERIOD_P_YR,
    REFERENCE,
    REFERENCE_BUDGET,
    SLOW_TIMESCALES_YR,
    TOTAL_SNRS,
    balanced_signs,
    build_reference,
    decay_time,
    eigenvalue_yr,
    forcing_overlap,
    forcing_overlap_sweep,
    forced_response,
    forced_variance,
    equal_budget,
    internal_variance,
    make_dataset,
    modal_coordinates,
    noise_overlap,
    noise_overlap_sweep,
    pair_plane_overlap,
    partial_snr_sweep,
    slow_timescale_sweep,
    rotate_slow_to_b,
    spatial_overlap_sweep,
    tilt_noise_to_slow,
    total_snr_sweep,
)
from support import SMALL, assert_unit_b_scale


# one level per study for test_sweeps_are_invariant_to_the_b_normalization; the slow-timescale levels
# are away from the reference tau_1 = 20 yr, where the two normalizations would agree by construction
INVARIANCE_LEVELS = (
    ("total_snr", 1 / 3),
    ("partial_snr_slow", 4),
    ("partial_snr_pair", 4),
    ("partial_snr_complement", 4),
    ("slow_timescale_snr", 100),
    ("spatial_overlap", 0.75),
    ("forcing_overlap", 0.9),
    ("noise_overlap", 0.75),
    ("joint_snr_timescale", (1, 100)),
    ("slow_timescale_modal_variance", 100),
)


def all_sweeps(**kwargs):
    return (
        total_snr_sweep(**kwargs)
        + [ds for c in PARTIAL_SNR_COMPONENTS for ds in partial_snr_sweep(c, **kwargs)]
        + slow_timescale_sweep(hold="snr", **kwargs) + slow_timescale_sweep(hold="modal_variance", **kwargs)
        + spatial_overlap_sweep(**kwargs) + forcing_overlap_sweep(**kwargs) + noise_overlap_sweep(**kwargs)
    )


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


def test_sweeps_are_invariant_to_the_b_normalization():
    """Rescaling B rescales the data, so every score is unchanged -- except in one study.

    Every budget is proportional to that level's own V_f (config_budget), so scaling B scales the
    forced response, the noise and the data by one common factor with the same random draws; the
    methods are linear and every score is a ratio. slow_timescale_modal_variance is the exception:
    it pins the budget to the reference while V_f follows tau_1, so the ratio of forced to internal
    amplitude -- and with it the SNR -- depends on how B is normalized. That is intended (the study
    asks what happens when internal variability keeps its size), and this test pins it down so the
    dependence cannot spread to the other studies unnoticed.
    """
    import ablation_data as data

    scaled = {study: data.sweep(study, [level], **SMALL) for study, level in INVARIANCE_LEVELS}
    original_B = data.System.B
    try:
        data.System.B = property(lambda self: self.b)  # the raw unit pattern, before System.b_scale
        data.build_reference.cache_clear()
        raw = {study: data.sweep(study, [level], **SMALL) for study, level in INVARIANCE_LEVELS}
    finally:
        data.System.B = original_B
        data.build_reference.cache_clear()

    for study, _ in INVARIANCE_LEVELS:
        (a,), (b,) = scaled[study], raw[study]
        factor = np.linalg.norm(b.data) / np.linalg.norm(a.data)
        error = np.abs(b.data - factor * a.data).max() / np.abs(b.data).max()
        if study == "slow_timescale_modal_variance":
            assert error > 0.1 and abs(b.snr / a.snr - 1) > 0.5, study
        else:
            assert error < 1e-12 and abs(b.snr / a.snr - 1) < 1e-12, (study, error)


def test_structure():
    for ds in all_sweeps(**SMALL):
        s = ds.system
        assert ds.forced.shape == (N, M) and ds.internal.shape == (SMALL["n_realizations"], N, 20)
        assert ds.y.shape == (s.spinup + N,)
        assert np.abs(ds.data - (ds.forced + ds.internal)).max() < 1e-12
        y = ds.y[s.spinup:]
        assert np.abs(ds.forced[1:] - (ds.forced[:-1] @ s.A.T + np.outer(y[1:], s.B))).max() < 1e-10
        assert abs(y.mean()) < 1e-12
        long_forcings, short_forcings, transition_time = ds.forcings()
        assert transition_time == s.spinup and long_forcings.shape == (s.spinup + N, 1)
        assert np.all(short_forcings[:, 0] == y) and np.all(long_forcings[:, 0] == ds.y)
        # derived from the longest-lived mode, floored at MIN_SPINUP where that is longer
        assert s.spinup == max(int(np.ceil(SPINUP_DECAY_TIMES * decay_time(s.eigvals).max())), MIN_SPINUP)


def test_spinup():
    """The spin-up equilibrates the slowest mode: the truth starts from quasi-equilibrium, not from zero."""
    for ds in all_sweeps(**SMALL):
        s = ds.system
        assert np.exp(-s.spinup / decay_time(s.eigvals).max()) < 1e-15, (ds.study, ds.param_value)


def test_b_scaling():
    """B is b scaled so the initial quasi-equilibrium state has unit norm, at every level of every sweep."""
    for ds in all_sweeps(**SMALL) + [make_dataset(REFERENCE, equal_budget, **SMALL)]:
        s = ds.system
        assert_unit_b_scale(s)
        assert abs(np.linalg.norm(s.b) - 1) < 1e-12  # b itself stays the raw unit pattern
    V_f = forced_variance(forced_response(REFERENCE)[1])
    fast = replace(REFERENCE, lam1=eigenvalue_yr(0.84))
    V_f_fast = forced_variance(forced_response(fast)[1])
    # golden values: V_f of the default exp + dip forcing over 1850-2014, under the B scaling
    assert abs(V_f / 0.2654566 - 1) < 1e-3 and abs(V_f_fast / 0.5777657 - 1) < 1e-3


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


def test_forcing_overlap():
    """w_1 is rotated toward b-hat and nothing else moves."""
    b = REFERENCE.b
    sweep = forcing_overlap_sweep(**SMALL)
    for ds, overlap in zip(sweep, B_OVERLAPS):
        W = ds.system.W
        assert abs(forcing_overlap(W, b) - overlap) < 1e-12, overlap
        assert abs(np.linalg.norm(W[:, 0]) - 1) < 1e-12, "a rotation preserves the norm"
        assert np.all(W[:, 1:] == W_REF[:, 1:]), "only the slow mode moves"
        assert np.array_equal(ds.system.b, b), "the forcing pattern itself does not move"
        # w_1 leaves span(S), so [pinv(S); Q_perp.T] is no longer the inverse and np.linalg.inv is used
        assert np.abs(ds.system.W_inv @ W - np.eye(len(W))).max() < 1e-10
        assert abs(ds.snr - 1 / 3) < 1e-12
    base = sweep[B_OVERLAPS.index(BASE_B_OVERLAP)].system
    assert np.abs(base.W - W_REF).max() < 1e-12 and np.abs(base.W_inv - W_INV_REF).max() < 1e-10


def test_forcing_overlap_matches_the_closed_form():
    """The rotation equals c b-hat + sqrt(1 - c^2) u_perp, which is what the README documents."""
    b, w1 = REFERENCE.b, W_REF[:, 0] / np.linalg.norm(W_REF[:, 0])
    perpendicular = w1 - (w1 @ b) * b
    u_perp = perpendicular / np.linalg.norm(perpendicular)
    for c in (0.0, 0.25, 0.5, BASE_B_OVERLAP, 0.8, 0.9, 1.0):
        W, _ = rotate_slow_to_b(W_REF, b, c)
        assert np.abs(W[:, 0] - (c * b + np.sqrt((1 - c) * (1 + c)) * u_perp)).max() < 1e-12, c
    for bad in (-1.5, 1.0000001, 2.0):
        try:
            rotate_slow_to_b(W_REF, b, bad)
        except ValueError:
            continue
        raise AssertionError(f"overlap {bad} outside [-1, 1] should raise, not return nan")


def test_noise_overlap():
    """The noisy modes tilt toward +-w_1 and nothing else moves: not w_1, not the forcing, not the noise total.

    The forced response is held by the sign balancing to within 1e-3 (README.md), and the total internal
    variance exactly, since every column stays unit norm; what grows is the noise on w_1, by c^2 times the
    complement's whole budget.
    """
    reference = total_snr_sweep(snrs=[1 / 3], **SMALL)[0]
    sweep = noise_overlap_sweep(**SMALL)
    gamma1 = (W_INV_REF @ REFERENCE.b)[0]
    for ds, overlap in zip(sweep, build_reference().noise_overlaps):
        s = ds.system
        W, w1 = s.W, s.W[:, 0]
        assert abs(noise_overlap(W) - overlap) < 1e-12, overlap
        assert np.allclose(np.abs(w1 @ W[:, 3:]), overlap, atol=1e-12), "every noisy mode at the same angle"
        assert np.allclose(np.linalg.norm(W, axis=0), 1), "a rotation preserves the norm"
        assert np.all(W[:, :3] == W_REF[:, :3]), "only the noisy modes move"
        assert np.abs(s.W_inv @ W - np.eye(M)).max() < 1e-10
        assert abs((s.W_inv @ s.b)[0] - gamma1) < 1e-4, "the slow mode's share of the forcing is held"
        assert np.abs(ds.forced - reference.forced).max() < 1e-3 * np.abs(reference.forced).max(), overlap
        assert abs(ds.snr - 1 / 3) < 1e-12
        # unit columns and independent modal amplitudes: the total is the budget, whatever the tilt
        total = internal_variance(s.s1_sq, s.sp_sq, s.sc_sq)
        assert abs(np.sum(np.linalg.norm(W, axis=0) ** 2 * s.modal_variances) - total) < 1e-12 * total
        # the noise on the slow fingerprint: the reference's, plus c^2 of the complement's whole budget
        on_w1 = np.sum((w1 @ W) ** 2 * s.modal_variances)
        pair = ((w1 @ W[:, 1]) ** 2 + (w1 @ W[:, 2]) ** 2) * s.sp_sq
        expected = s.s1_sq + pair + overlap**2 * (M - 3) * s.sc_sq
        assert abs(on_w1 - expected) < 1e-12 * expected, overlap
    base = sweep[0].system
    assert np.all(base.W == W_REF) and np.abs(base.W_inv - W_INV_REF).max() < 1e-10
    for bad in (-0.1, 1.0, 1.5):
        try:
            tilt_noise_to_slow(W_REF, np.ones(M - 3), bad)
        except ValueError:
            continue
        raise AssertionError(f"noise overlap {bad} outside [0, 1) should raise")


def test_balanced_signs_cancel_the_forcing_share():
    """The exhaustive search finds the best cancellation, with the first sign fixed at +1."""
    gamma = np.array([0.5, 0.3, 0.2, 0.15, 0.15])  # 0.5 - 0.3 - 0.2 + 0.15 - 0.15 cancels exactly
    signs = balanced_signs(gamma)
    assert signs[0] == 1 and set(np.abs(signs)) == {1}
    assert abs(signs @ gamma) < 1e-12
    reference_signs = np.asarray(build_reference().noise_signs)
    assert abs(reference_signs @ (W_INV_REF @ REFERENCE.b)[3:]) < 1e-4


def test_common_random_numbers():
    sweep = total_snr_sweep(**SMALL)
    ref = sweep[0].internal / np.sqrt(sweep[0].system.s1_sq)
    for ds in sweep[1:]:
        assert np.abs(ds.internal / np.sqrt(ds.system.s1_sq) - ref).max() < 1e-10, ds.param_value
