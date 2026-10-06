import pathlib
import tempfile

import pandas as pd

from config import DEFAULT, with_overrides
from plot_ablation_run_results import Run, plot_phase, plot_phase_change, plot_sweeps


def make_run(name, cfg, offset=0.0):
    rows = []
    for value in (0.1, 1.0):
        for method, shift in (("PullbackDMDc", 0.0), ("LIM", 0.03), ("oracle", -0.02)):
            rows.append(dict(
                study="total_snr",
                param_name="SNR",
                param_value=value,
                method=method,
                forced_corr=0.75 + 0.2 * value + shift + offset,
                forced_rel_rmse=0.8 - 0.3 * value - shift,
                slow_eig_err=0.01 + shift,
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
