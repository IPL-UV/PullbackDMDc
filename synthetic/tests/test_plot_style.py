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
