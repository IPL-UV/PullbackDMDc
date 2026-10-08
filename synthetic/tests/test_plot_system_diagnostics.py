import pathlib
import tempfile

import numpy as np

from ablation_data import (DEFAULT_STUDIES, M, STUDIES, SWEEP_STUDIES, build_reference, find_level,
                           rotate_slow_to_b)
from config import DEFAULT, with_overrides
from plot_system_diagnostics import (
    ABLATION_PANELS,
    forced_panel,
    level_datasets,
    noise_pattern_panel,
    pattern_panel,
    plot_ensemble_super_spaghetti,
    plot_forced_response_ablations,
    plot_forced_response_shape,
    realization_panel,
    reference_dataset,
)


def test_tau1_diagnostics_and_response_plots():
    # the short levels only: tau_1 = 100 yr costs a 48 000-month spin-up and adds nothing here
    taus = (1, 2, 5)
    cfg = with_overrides(DEFAULT, [f"slow_timescales_yr={taus}", "total_snrs=(0.1,1,10)"])
    datasets = level_datasets(cfg, "slow_timescale_snr", taus)
    assert tuple(datasets) == taus
    for tau, ds in datasets.items():
        (sweep,) = find_level(cfg, "slow_timescale_snr", tau)(n_realizations=2)
        assert np.array_equal(ds.forced, sweep.forced) and np.array_equal(ds.y, sweep.y)
    V_f = [ds.V_f for ds in datasets.values()]
    assert V_f[0] > V_f[1] > V_f[2], V_f
    # total_snr rescales the noise budget only, which is why its column draws one black forced response
    snr_data = level_datasets(cfg, "total_snr", cfg.total_snrs)
    forced = [ds.forced for ds in snr_data.values()]
    assert all(np.allclose(f, forced[0]) for f in forced)
    # the forcing_overlap column's punchline: the rotation is norm-preserving, so at c = 1 it puts w_1
    # exactly on b-hat
    reference = build_reference(cfg)
    W, _ = rotate_slow_to_b(reference.system.W, reference.system.b, 1.0)
    assert np.allclose(W[:, 0], reference.system.b)
    # and the levels that column colours by are resolved from the reference, not read off the config
    assert cfg.b_overlaps is None and reference.b_overlaps[0] == 0.0
    (fast,) = find_level(cfg, "slow_timescale_snr", 1)(n_realizations=1)
    with tempfile.TemporaryDirectory() as tmp:
        plot_forced_response_ablations(pathlib.Path(tmp) / "ablations.png", cfg=cfg)
        plot_forced_response_shape(fast, pathlib.Path(tmp) / "shape.png")
        assert all((pathlib.Path(tmp) / name).stat().st_size > 0 for name in ("ablations.png", "shape.png"))
    s = fast.system
    gain = np.linalg.solve(np.eye(M) - s.A, s.B)
    quasi = np.outer(fast.y[s.spinup:], gain)
    assert np.sqrt(((fast.forced - quasi) ** 2).mean() / (fast.forced ** 2).mean()) < 0.1
    assert np.abs(fast.forced.mean(axis=0)).max() < 0.05 * np.abs(fast.forced).max()


def test_the_ablation_columns_are_the_sweep_figure_s_columns():
    """The figure is a row of the sweep figure's columns: same studies, same order, one panel kind each."""
    # every column plot_sweeps can draw has a panel here, so --studies means the same thing in both
    assert set(ABLATION_PANELS) == set(SWEEP_STUDIES)
    # the kind follows what the study bends: the budget alone, lambda_1, or the shape of w_1
    assert ABLATION_PANELS["total_snr"] is realization_panel
    assert ABLATION_PANELS["partial_snr_slow"] is realization_panel
    assert ABLATION_PANELS["slow_timescale_modal_variance"] is forced_panel
    assert ABLATION_PANELS["forcing_overlap"] is pattern_panel
    assert ABLATION_PANELS["noise_overlap"] is noise_pattern_panel
    # the noise_overlap column's punchline: the mode it draws closes on +w_1, not -w_1, at the top level
    reference = build_reference(DEFAULT)
    w1 = reference.system.W[:, 0]
    assert reference.noise_signs[0] == 1
    assert abs(STUDIES["noise_overlap"].system(reference, 0.95).W[:, 3] @ w1 - 0.95) < 1e-12

    taus = (1, 2, 5)  # as above: tau_1 = 100 yr costs a 48 000-month spin-up and adds nothing here
    cfg = with_overrides(DEFAULT, [f"slow_timescales_yr={taus}", "total_snrs=(0.1,1,10)"])
    with tempfile.TemporaryDirectory() as tmp:
        default = plot_forced_response_ablations(pathlib.Path(tmp) / "default.png", cfg=cfg)
        assert default == list(DEFAULT_STUDIES)
        # --studies sets the columns and their order, exactly as it does for plot_sweeps; this selection
        # reaches all three panel kinds, the pattern one through both of the studies that use it
        chosen = ("forcing_overlap", "total_snr", "slow_timescale_snr", "spatial_overlap")
        assert plot_forced_response_ablations(pathlib.Path(tmp) / "chosen.png", cfg=cfg,
                                              studies=chosen) == list(chosen)
        assert all(path.stat().st_size > 0 for path in pathlib.Path(tmp).glob("*.png"))


def test_super_spaghetti_plot():
    ds = reference_dataset(with_overrides(DEFAULT, ["n_realizations=3"]))
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "ensemble.png"
        plot_ensemble_super_spaghetti(ds, path, label="test")
        assert path.stat().st_size > 0
