import numpy as np

from ablations import STUDIES, find_level, forced_variance, make_level, scaled_split, study_levels
from support import CHEAP


def test_every_study_level_is_enumerated():
    counts = {study: sum(s == study for s, _ in study_levels(CHEAP)) for study in STUDIES}
    assert counts == {"total_snr": 7, "partial_snr_slow": 5, "partial_snr_pair": 5, "partial_snr_complement": 5,
                      "slow_timescale_snr": 7, "slow_timescale_modal_variance": 7, "obs_noise": 6, "forcing_rotation": 5,
                      "joint_snr_timescale": 49}, counts
    assert find_level(CHEAP, "joint_snr_timescale", (1, 20)) == (1, 20)


def test_scaled_split():
    assert scaled_split((1, 1, 2, 3, 4), 2) == (1, 2, 4, 6, 8)
    assert scaled_split((1, 1, 2, 3, 4), 2, index=2) == (1, 1, 4, 3, 4)


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
        assert np.isclose(ens["obs_std"], reference["obs_std"], rtol=1e-10)
        assert np.isclose(ens["Vf"], forced_variance(CHEAP, tau))


def test_obs_noise_levels_share_their_draws():
    """Same spin-up, same seed: the measurement-noise share scales the same draws, and nothing else moves."""
    a, b = (make_level(CHEAP, "obs_noise", v) for v in (0.1, 2))
    assert np.allclose(a["internal"], b["internal"], atol=1e-12) and np.array_equal(a["forced"], b["forced"])
    assert np.allclose(a["obs_noise"] / a["obs_std"], b["obs_noise"] / b["obs_std"])
    for ens, v in ((a, 0.1), (b, 2)):
        assert abs(ens["snr"] * (2.9 + v) - 1) < 1e-12
