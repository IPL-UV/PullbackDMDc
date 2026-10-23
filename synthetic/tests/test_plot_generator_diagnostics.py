import pathlib
import tempfile

from plot_generator_diagnostics import plot_pullback_results, plot_setup
from support import cheap_ensemble
from synthetic_generator import fit_pullback_ensemble


def test_generator_figures():
    """setup.pdf and pullback_results.pdf render from a cheap ensemble, with the default lag of 1."""
    ens = cheap_ensemble(2)
    models, forced_est = fit_pullback_ensemble(ens["data"], ens["y"], ens["y_full"], lag=1)
    with tempfile.TemporaryDirectory() as tmp:
        fig_dir = pathlib.Path(tmp)
        plot_setup(spin_up=ens["spin_up"], fig_dir=fig_dir)
        plot_pullback_results(ens, models, forced_est, fig_dir=fig_dir)
        assert all((fig_dir / name).stat().st_size > 0 for name in ("setup.pdf", "pullback_results.pdf"))
