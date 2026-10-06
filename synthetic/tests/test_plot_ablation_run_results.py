import pathlib
import tempfile

import pandas as pd

from config import DEFAULT, with_overrides
from plot_ablation_run_results import (DEFAULT_STUDIES, ROW_SPECS, Run, plot_phase, plot_phase_change,
                                       plot_sweeps)

# the five default columns plus two the figure only shows via --studies: spatial_overlap covers the
# LINEAR_X branch, and partial_snr_pair (a default) covers the pair_angle variant of row (d)
PARAM_NAMES = {
    "total_snr": "SNR",
    "partial_snr_slow": "slow variance factor",
    "partial_snr_pair": "pair variance factor",
    "partial_snr_complement": "complement variance factor",
    "slow_timescale_modal_variance": r"$\tau_1$ (yr)",
    "slow_timescale_snr": r"$\tau_1$ (yr)",
    "spatial_overlap": "mode overlap",
}


def make_run(name, cfg, offset=0.0):
    rows = []
    for study, param_name in PARAM_NAMES.items():
        for value in (0.1, 1.0):
            for method, shift in (("PullbackDMDc", 0.0), ("LIM", 0.03), ("oracle", -0.02)):
                rows.append(dict(
                    study=study,
                    param_name=param_name,
                    param_value=value,
                    method=method,
                    forced_corr=0.75 + 0.2 * value + shift + offset,
                    forced_rel_rmse=0.8 - 0.3 * value - shift,
                    slow_eig_err=0.01 + shift,
                    slow_tau_rel_err=0.35 + shift,
                    slow_tau_log_ratio=0.3 + shift,
                    slow_unstable=0.0 if method == "PullbackDMDc" else 0.5,
                    slow_angle=5 + shift,
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
    return Run(name, cfg, pd.DataFrame(rows))


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

        # a study no run measured is dropped rather than drawn empty
        thin = Run("thin", DEFAULT, run.results[run.results.study != "partial_snr_pair"])
        assert "partial_snr_pair" not in plot_sweeps([thin], pathlib.Path(tmp) / "thin.png")
        assert plot_sweeps([thin], pathlib.Path(tmp) / "none.png", studies=("partial_snr_pair",)) == []

        assert all(path.stat().st_size > 0 for path in pathlib.Path(tmp).glob("*.png"))


def test_rows_are_the_four_metrics():
    """Row labels carry the panel letters, and row (d) scores the pair plane only in PAIR_STUDIES."""
    assert [spec.label[:3] for spec in ROW_SPECS] == ["(a)", "(b)", "(c)", "(d)"]
    assert [spec.yscale for spec in ROW_SPECS] == ["linear", "linear", "log", "linear"]
    shape = ROW_SPECS[-1]
    assert shape.panel("partial_snr_pair", 1)[0] == "pair_angle"
    assert shape.panel("total_snr", 1)[0] == "slow_angle"
    # every panel in the row is annotated, so no column is a silent exception to the shared row label
    assert all(shape.panel(study, 1)[1] is not None for study in DEFAULT_STUDIES)
    assert ROW_SPECS[2].panel("total_snr", 1) == ("slow_tau_log_ratio", None)
