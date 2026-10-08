import pathlib
import tempfile

import numpy as np
import pandas as pd

from ablation_data import M
from config import DEFAULT, with_overrides
from plot_ablation_run_results import (DEFAULT_STUDIES, ROW_SPECS, Run, default_level, median_residual,
                                       plot_forcing_change, plot_phase, plot_phase_change,
                                       plot_slow_mode_shapes, plot_sweeps)
from plot_system_diagnostics import plot_forced_response_ablations

# the four default columns plus the ones the figures only show via --studies: spatial_overlap covers the
# LINEAR_X branch, and partial_snr_pair covers the pair_corr variant of row (c)
PARAM_NAMES = {
    "total_snr": "SNR",
    "partial_snr_slow": "slow variance factor",
    "partial_snr_pair": "pair variance factor",
    "partial_snr_complement": "complement variance factor",
    "slow_timescale_modal_variance": r"$\tau_1$ (yr)",
    "slow_timescale_snr": r"$\tau_1$ (yr)",
    "spatial_overlap": "mode overlap",
    "forcing_overlap": r"$\cos\angle(w_1,\hat b)$",
    "noise_overlap": r"$|\cos\angle(q_k,w_1)|$",
}
MODE_METHODS = ("PullbackDMDc", "LIM", "truth")


def make_run(name, cfg, offset=0.0):
    rows = []
    for study, param_name in PARAM_NAMES.items():
        for value in (0.1, 1.0):
            for method, shift in (("PullbackDMDc", 0.0), ("LIM", 0.03)):
                rows.append(dict(
                    study=study,
                    param_name=param_name,
                    param_value=value,
                    method=method,
                    # the true tau_1 is the swept parameter in the timescale studies and the run's
                    # config everywhere else, so row (b)'s truth line is exercised both ways
                    tau1_yr=value if study.startswith("slow_timescale") else cfg.tau1_yr,
                    forced_corr=0.75 + 0.2 * value + shift + offset,
                    forced_rel_rmse=0.8 - 0.3 * value - shift,
                    slow_eig_err=0.01 + shift,
                    slow_tau_yr=cfg.tau1_yr * (1.3 + shift),
                    slow_tau_rel_err=0.35 + shift,
                    slow_tau_log_ratio=0.3 + shift,
                    slow_unstable=0.0 if method == "PullbackDMDc" else 0.5,
                    slow_corr=0.95 - shift,
                    slow_angle=5 + shift,
                    pair_corr=0.9 - shift,
                    pair_angle=2 + shift,
                ))
    for snr in (0.1, 1.0):
        for tau in (2.0, 20.0):
            for method, shift in (("PullbackDMDc", 0.0), ("LIM", 0.03)):
                rows.append(dict(
                    study="joint_snr_timescale",
                    param_name="SNR",
                    param_value=snr,
                    tau1_yr=tau,
                    method=method,
                    forced_corr=0.78 + 0.15 * snr + 0.02 * (tau == 20) + shift + offset,
                    forced_rel_rmse=0.75 - 0.2 * snr - 0.1 * (tau == 20) - shift,
                ))
    return Run(name, cfg, pd.DataFrame(rows), make_modes())


def make_modes():
    """A stand-in slow_modes.npz frame: a pattern per (study, level, method), plus the per-level truth.

    The truth is varied in the overlap studies and held fixed elsewhere, so the figure's "is the truth
    moving?" branch is exercised both ways.
    """
    lat = np.linspace(-1, 1, M)
    rows = []
    for study, param_name in PARAM_NAMES.items():
        for value in (0.1, 1.0):
            for method in MODE_METHODS:
                moves = study in {"spatial_overlap", "forcing_overlap"} and method == "truth"
                pattern = np.cos(lat * (1 + value)) if moves else np.cos(lat)
                rows.append(dict(study=study, param_name=param_name, param_value=value, tau1_yr=20.0,
                                 realization=0, method=method,
                                 pattern=pattern / np.linalg.norm(pattern)))
    return pd.DataFrame(rows)


def test_ablation_run_result_figures():
    reference = make_run("baseline", DEFAULT)
    variant = make_run("slow50", with_overrides(DEFAULT, ["tau1_yr=50"]), offset=0.02)
    with tempfile.TemporaryDirectory() as tmp:
        paths = {
            "sweeps": pathlib.Path(tmp) / "sweeps.png",
            "phase": pathlib.Path(tmp) / "phase.png",
            "change": pathlib.Path(tmp) / "change.png",
        }
        plot_sweeps([reference, variant], paths["sweeps"])
        plot_phase(reference, paths["phase"])
        plot_phase_change(reference, variant, paths["change"])
        assert all(path.stat().st_size > 0 for path in paths.values())


def test_sweep_columns_follow_the_studies_asked_for():
    """The figure is transposed: one row per metric, one column per study, in the order given."""
    run = make_run("baseline", DEFAULT)
    with tempfile.TemporaryDirectory() as tmp:
        default = plot_sweeps([run], pathlib.Path(tmp) / "default.png")
        assert default == list(DEFAULT_STUDIES)

        # --studies picks any other selection, left to right, and sets the column order
        chosen = ("spatial_overlap", "total_snr")
        assert plot_sweeps([run], pathlib.Path(tmp) / "chosen.png", studies=chosen) == list(chosen)

        # a study no run measured is dropped rather than drawn empty, defaults included
        thin = Run("thin", DEFAULT, run.results[run.results.study != "partial_snr_slow"])
        assert "partial_snr_slow" in DEFAULT_STUDIES
        assert "partial_snr_slow" not in plot_sweeps([thin], pathlib.Path(tmp) / "thin.png")
        assert plot_sweeps([thin], pathlib.Path(tmp) / "none.png", studies=("partial_snr_slow",)) == []

        # partial_snr_slow is a default column, but the pair and complement variants are not; --studies
        # still draws them, and partial_snr_pair is the only way row (c)'s pair_corr variant is reached
        assert "partial_snr_pair" not in DEFAULT_STUDIES
        assert plot_sweeps([run], pathlib.Path(tmp) / "pair.png",
                           studies=("partial_snr_pair",)) == ["partial_snr_pair"]

        assert all(path.stat().st_size > 0 for path in pathlib.Path(tmp).glob("*.png"))


def test_rows_are_the_three_metrics():
    """Row labels carry the panel letters, and row (c) scores the pair plane only in PAIR_STUDIES."""
    assert [spec.label[:3] for spec in ROW_SPECS] == ["(a)", "(b)", "(c)"]
    assert [spec.yscale for spec in ROW_SPECS] == ["linear", "log", "linear"]
    shape = ROW_SPECS[-1]
    assert shape.panel("partial_snr_pair", 1)[0] == "pair_corr"
    assert shape.panel("total_snr", 1)[0] == "slow_corr"
    # every panel in the row is annotated, so no column is a silent exception to the shared row label
    assert all(shape.panel(study, 1)[1] is not None for study in DEFAULT_STUDIES)
    # row (b) plots the estimate itself, against the true tau_1 drawn from the results table
    assert ROW_SPECS[1].panel("total_snr", 1) == ("slow_tau_yr", None)
    assert ROW_SPECS[1].truth == "tau1_yr" and ROW_SPECS[1].note
    assert all(spec.truth is None for spec in ROW_SPECS if spec is not ROW_SPECS[1])


def test_a_legacy_run_skips_the_metrics_it_never_scored():
    """results/b_unscaled/ predates the operator columns: its rows are left empty, not raised on."""
    run = make_run("legacy", DEFAULT)
    legacy = Run("legacy", DEFAULT, run.results.drop(columns=["slow_tau_yr", "slow_corr", "pair_corr"]))
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "legacy.png"
        assert plot_sweeps([legacy], path) == list(DEFAULT_STUDIES)
        assert path.stat().st_size > 0


def test_the_study_column_figures_are_laid_out_on_one_grid():
    """results_sweeps, results_slow_mode_shapes and forced_response_ablations stack, so a column of one
    sits over the same column of the others: same studies, same order, same width.

    Each is saved with bbox="tight", which crops to its own contents, so the test is on the saved images:
    the same study selection has to come out the same number of pixels wide in all three. The three differ
    by a few pixels of tick-label margin -- hence the tolerance -- but a figure that stopped reserving
    LEGEND_INCHES, or dropped to a different COLUMN_W, would be out by a sixth of its width.
    """
    # a cheap selection: spatial_overlap needs no dataset simulated at all, and total_snr's levels are
    # the fast ones. No slow_timescale column, whose tau_1 = 100 yr level costs a 48 000-month spin-up.
    studies = ("total_snr", "spatial_overlap")
    cfg = with_overrides(DEFAULT, ["total_snrs=(0.1,1,10)"])
    run = make_run("baseline", cfg)
    with tempfile.TemporaryDirectory() as tmp:
        paths = {name: pathlib.Path(tmp) / f"{name}.png" for name in ("sweeps", "shapes", "ablations")}
        plot_sweeps([run], paths["sweeps"], studies=studies)
        plot_slow_mode_shapes(run, paths["shapes"], studies=studies)
        plot_forced_response_ablations(paths["ablations"], cfg=cfg, studies=studies)
        # the PNG's IHDR width, bytes 16:20, so the test needs no image library
        widths = {name: int.from_bytes(path.read_bytes()[16:20], "big") for name, path in paths.items()}
        assert max(widths.values()) - min(widths.values()) < 0.03 * min(widths.values()), widths


def test_the_residual_is_the_departure_from_that_level_s_truth():
    """A fit on the truth leaves nothing, and an offset fit leaves exactly the offset."""
    truth = np.cos(np.linspace(-1, 1, M))
    fits = pd.DataFrame(dict(pattern=[truth, truth]))
    assert np.allclose(median_residual(fits, truth), 0)
    shifted = pd.DataFrame(dict(pattern=[truth + 0.1, truth + 0.3]))
    assert np.allclose(median_residual(shifted, truth), 0.2)


def test_slow_mode_shape_figure():
    """Both branches of the truth: fixed in total_snr, moving per level in the overlap studies."""
    run = make_run("baseline", DEFAULT)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "shapes.png"
        drawn = plot_slow_mode_shapes(run, path, studies=("total_snr", "spatial_overlap", "forcing_overlap"))
        assert drawn == ["total_snr", "spatial_overlap", "forcing_overlap"]
        assert path.stat().st_size > 0


def test_slow_mode_shape_figure_skips_a_run_without_patterns():
    """b_unscaled predates slow_modes.npz: no figure, rather than a crash."""
    legacy = Run("b_unscaled", DEFAULT, make_run("x", DEFAULT).results, None)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "shapes.png"
        assert plot_slow_mode_shapes(legacy, path) == []
        assert not path.exists()


def test_forcing_change_figure():
    reference = make_run("baseline", DEFAULT)
    variant = make_run("dip1", with_overrides(DEFAULT, ["gauss_dip_amp=1"]), offset=0.02)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "forcing_change.png"
        plot_forcing_change(reference, variant, path)
        assert path.stat().st_size > 0


def test_the_default_system_is_marked_in_every_sweep_column():
    """Each sweep bends one parameter away from the default, so each column has one level to mark.

    The overlap studies are left out: their default comes from build_reference, which builds the system.
    """
    assert default_level("total_snr", DEFAULT) == 1 / sum(DEFAULT.variances)
    assert all(default_level(f"partial_snr_{c}", DEFAULT) == 1.0 for c in ("slow", "pair", "complement"))
    assert all(default_level(f"slow_timescale_{h}", DEFAULT) == DEFAULT.tau1_yr
               for h in ("snr", "modal_variance"))
    # the line follows the config it is given, not the package default
    slow50 = with_overrides(DEFAULT, ["tau1_yr=50"])
    assert default_level("slow_timescale_snr", slow50) == 50

    # the phase study sweeps two parameters at once, so no single level is the default
    assert default_level("joint_snr_timescale", DEFAULT) is None
    # every column the sweep figure draws by default is marked
    assert all(default_level(study, DEFAULT) is not None for study in DEFAULT_STUDIES)
