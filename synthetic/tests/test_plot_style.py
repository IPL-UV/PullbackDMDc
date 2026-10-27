import matplotlib as mpl

import plot_style  # noqa: F401  (importing it is what applies the style)


def test_shared_plot_style():
    assert mpl.rcParams["font.family"] == ["sans-serif"]
    assert mpl.rcParams["mathtext.fontset"] == "custom"
    assert mpl.rcParams["font.size"] == 8
    assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["ps.fonttype"] == 42
    # the sizes the figure scripts no longer pass at every call site
    assert mpl.rcParams["axes.titlesize"] == 9
    assert mpl.rcParams["legend.fontsize"] == 7.5
    assert mpl.rcParams["savefig.dpi"] == 150


def test_level_ticklabels_blank_only_a_crowded_interior_label():
    """0.9 beside 0.95 on a linear bar loses its label, not its tick; the ends and well-spaced levels keep theirs."""
    from matplotlib.colors import LogNorm, Normalize

    from plot_style import level_ticklabels

    assert level_ticklabels([0, 0.25, 0.5, 0.75, 0.9, 0.95], Normalize(0, 0.95)) == [
        "0", "0.25", "0.5", "0.75", "", "0.95"]
    snrs = [1 / 30, 0.1, 1 / 3, 1, 3, 10, 30]
    assert level_ticklabels(snrs, LogNorm(min(snrs), max(snrs))) == ["0.0333", "0.1", "0.333", "1", "3", "10", "30"]
