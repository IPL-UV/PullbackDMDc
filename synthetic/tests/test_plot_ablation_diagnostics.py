import pathlib
import tempfile

from ablation_data import REFERENCE, equal_budget, make_dataset
from plot_ablation_diagnostics import plot_calibration, plot_sweep


def test_ablation_diagnostic_figures():
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=2)
    with tempfile.TemporaryDirectory() as tmp:
        calibration = pathlib.Path(tmp) / "calibration.png"
        sweep = pathlib.Path(tmp) / "sweep.png"
        plot_calibration(ds, calibration)
        plot_sweep([ds], sweep, "smoke test")
        assert calibration.stat().st_size > 0
        assert sweep.stat().st_size > 0
