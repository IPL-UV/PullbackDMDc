"""Shared matplotlib style and figure helpers for every figure in synthetic/; import it before creating figures.

Sizes and colours live here, not at the call sites: a figure script sets a fontsize or a colour only where it
means something specific to that panel.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PAGE_W = 13  # inches, the width of a full-width figure
# The study-column figures -- results_sweeps and results_slow_mode_shapes -- are both
# laid out on this grid: one column per ablation study at COLUMN_W, one row per method or metric at ROW_H, and
# LEGEND_INCHES kept clear at the right for the legends plot_sweeps hangs off its last panel. Both
# reserve the strip, whether or not it has a legend to put there: they are meant to be stacked and read
# column by column, and a figure that skips the reservation lays its columns out at a different pitch.
COLUMN_W = 3.1
ROW_H = 2.5
LEGEND_INCHES = 1.45
PANEL_TITLE = 8  # the 20 latitude panels sit tighter than a normal axes title
ZERO_GREY = "0.5"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",   # same family as the text
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "savefig.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42,                            # embed TrueType, editable text
})


def save(fig, path, tight=True, bbox=None):
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches=bbox)
    plt.close(fig)
    print(f"saved {path}")


def legend_rect(fig):
    """The fraction of `fig` left for the axes once LEGEND_INCHES is reserved; a tight_layout rect right."""
    return 1 - LEGEND_INCHES / fig.get_figwidth()


def reserve_legend_strip(fig):
    """Keep LEGEND_INCHES clear at the right of a constrained-layout figure."""
    fig.get_layout_engine().set(rect=(0, 0, legend_rect(fig), 1))


def keep_legend_strip(fig):
    """Keep the reserved strip in a bbox="tight" save, legend or not.

    The crop follows the visible artists, so an empty strip -- or one whose legend is shorter than it -- is cut
    off and the figure comes out narrower than its column-mate. A transparent patch over the strip pins the
    crop to the figure's right edge in both, so their columns stay at the same pitch.
    """
    fig.add_artist(mpl.patches.Rectangle((legend_rect(fig), 0), 1 - legend_rect(fig), 1, transform=fig.transFigure,
                                         facecolor="none", edgecolor="none", zorder=-1))


# the narrowest gap, as a fraction of a level colorbar's length, at which two neighbouring tick labels still
# clear each other at PANEL_TITLE on a COLUMN_W-wide bar ("0.95" is about a twelfth of it)
LEVEL_LABEL_GAP = 0.09


def level_ticklabels(levels, norm):
    """'%.3g' labels for a colorbar ticked at a sweep's own levels, blank where one would run into another.

    Both ends of the sweep are always labelled; an interior level keeps its label only if it clears the
    labels already kept, walking down from the top, so a crowded level -- 0.9 beside 0.95 on a linear bar --
    keeps its tick but loses its label. `norm` is the bar's own, so the gap is measured where the labels sit.
    """
    positions = np.asarray(norm(np.asarray(levels, dtype=float)), dtype=float)
    order = np.argsort(-positions, kind="stable")
    kept = [positions[order[0]], positions[order[-1]]]
    labels = [""] * len(levels)
    for i in (order[0], order[-1]):
        labels[i] = f"{levels[i]:.3g}"
    for i in order[1:-1]:
        if min(abs(positions[i] - k) for k in kept) >= LEVEL_LABEL_GAP:
            labels[i] = f"{levels[i]:.3g}"
            kept.append(positions[i])
    return labels


def zero_line(ax, **kwargs):
    """The y = 0 reference line every timeseries panel draws."""
    return ax.axhline(0, color=ZERO_GREY, linewidth=0.6, **kwargs)
