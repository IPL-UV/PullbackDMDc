import pathlib
import tempfile

from ablation_data import CO2_GAUSS_FIT
from config import DEFAULT, with_overrides
from plot_forcing_comparison import plot_forcing_comparison


def test_forcing_comparison_figures_and_fit_ordering():
    gauss_fit = with_overrides(DEFAULT, [
        "forcing_source=analytic_gauss",
        f"gauss_efold_yr={100 / CO2_GAUSS_FIT['k']}",
        f"gauss_bump_amp={CO2_GAUSS_FIT['A_pos']}",
        f"gauss_bump_year={CO2_GAUSS_FIT['mu_pos']}",
        f"gauss_bump_width_yr={CO2_GAUSS_FIT['sigma_pos']}",
        f"gauss_dip_amp={CO2_GAUSS_FIT['A_neg']}",
        f"gauss_dip_year={CO2_GAUSS_FIT['mu_neg']}",
        f"gauss_dip_width_yr={CO2_GAUSS_FIT['sigma_neg']}",
    ])
    no_dip = with_overrides(gauss_fit, ["gauss_dip_amp=0"])
    with tempfile.TemporaryDirectory() as tmp:
        tables = [
            plot_forcing_comparison(cfg, pathlib.Path(tmp) / f"{name}.png")
            for cfg, name in ((DEFAULT, "default"), (gauss_fit, "fit"), (no_dip, "no_dip"))
        ]
        assert all((pathlib.Path(tmp) / f"{name}.png").stat().st_size > 0
                   for name in ("default", "fit", "no_dip"))
    fit, without_dip = tables[1], tables[2]
    assert fit["analytic_gauss"][0] < fit["analytic"][0] and fit["analytic_gauss"][2] < fit["analytic"][2]
    assert without_dip["analytic_gauss"][0] > fit["analytic_gauss"][0]
    assert without_dip["analytic"] == fit["analytic"]
