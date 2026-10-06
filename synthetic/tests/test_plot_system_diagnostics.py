import pathlib
import tempfile

import numpy as np

from ablation_data import M, find_level
from config import DEFAULT, with_overrides
from plot_system_diagnostics import (
    TAU1_COMPARED,
    plot_ensemble_super_spaghetti,
    plot_forced_response_shape,
    plot_forced_response_tau1,
    reference_dataset,
    tau1_datasets,
)


def test_tau1_diagnostics_and_response_plots():
    datasets = tau1_datasets(DEFAULT)
    assert tuple(datasets) == TAU1_COMPARED
    for tau, ds in datasets.items():
        (sweep,) = find_level(DEFAULT, "slow_timescale_snr", tau)(n_realizations=2)
        assert np.array_equal(ds.forced, sweep.forced) and np.array_equal(ds.y, sweep.y)
    V_f = [ds.V_f for ds in datasets.values()]
    assert V_f[0] > V_f[1] > V_f[2], V_f
    (fast,) = find_level(DEFAULT, "slow_timescale_snr", 1)(n_realizations=1)
    with tempfile.TemporaryDirectory() as tmp:
        plot_forced_response_tau1(pathlib.Path(tmp) / "tau1.png")
        plot_forced_response_shape(fast, pathlib.Path(tmp) / "shape.png")
        assert all((pathlib.Path(tmp) / name).stat().st_size > 0 for name in ("tau1.png", "shape.png"))
    s = fast.system
    gain = np.linalg.solve(np.eye(M) - s.A, s.B)
    quasi = np.outer(fast.y[s.spinup:], gain)
    assert np.sqrt(((fast.forced - quasi) ** 2).mean() / (fast.forced ** 2).mean()) < 0.1
    assert np.abs(fast.forced.mean(axis=0)).max() < 0.05 * np.abs(fast.forced).max()


def test_super_spaghetti_plot():
    ds = reference_dataset(with_overrides(DEFAULT, ["n_realizations=3"]))
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "ensemble.png"
        plot_ensemble_super_spaghetti(ds, path, label="test")
        assert path.stat().st_size > 0
