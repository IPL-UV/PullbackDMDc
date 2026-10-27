import numpy as np

from ablations import STUDIES, find_level, forced_variance, make_level, scaled_variances, study_levels
from support import CHEAP


def test_every_study_level_is_enumerated():
    counts = {study: sum(s == study for s, _ in study_levels(CHEAP)) for study in STUDIES}
    assert counts == {"total_snr": 9, "partial_snr_slow": 5, "partial_snr_pair": 5, "partial_snr_complement": 5,
                      "slow_timescale_snr": 8, "slow_timescale_modal_variance": 8, "obs_noise": 7, "overlap": 6,
                      "joint_snr_timescale": 72}, counts
    assert find_level(CHEAP, "joint_snr_timescale", (1, 20)) == (1, 20)


def test_scaled_variances():
    assert scaled_variances((1, 1, 2, 3, 4), 2) == (1, 2, 4, 6, 8)
    assert scaled_variances((1, 1, 2, 3, 4), 2, index=2) == (1, 1, 4, 3, 4)


def test_snr_studies_hit_their_targets():
    for snr in (0.1, 3):
        assert abs(make_level(CHEAP, "total_snr", snr)["snr"] / snr - 1) < 1e-12
    for tau in (5, 50):
        assert abs(make_level(CHEAP, "slow_timescale_snr", tau)["snr"] / CHEAP.snr - 1) < 1e-12


def test_modal_variance_study_pins_the_absolute_variances():
    reference = make_level(CHEAP)
    for tau in (5, 50):
        ens = make_level(CHEAP, "slow_timescale_modal_variance", tau)
        assert np.allclose(ens["modal_var"], reference["modal_var"], rtol=1e-10)
        assert np.isclose(ens["sigma_eps"], reference["sigma_eps"], rtol=1e-10)
        assert np.isclose(ens["Vf"], forced_variance(CHEAP, tau))


def test_obs_noise_levels_share_their_draws():
    """Same spin-up, same seed: the ratio V_eps / V_f scales the same draws, and nothing else moves."""
    a, b = (make_level(CHEAP, "obs_noise", v) for v in (0.01, 3))
    process = [ens["data"] - ens["forced"] - ens["obs_noise"] for ens in (a, b)]
    assert np.allclose(*process, atol=1e-12) and np.array_equal(a["forced"], b["forced"])
    assert np.allclose(a["obs_noise"] / a["sigma_eps"], b["obs_noise"] / b["sigma_eps"])
    for ens, v in ((a, 0.01), (b, 3)):
        assert abs(ens["snr"] * (2.9 + v) - 1) < 1e-12
