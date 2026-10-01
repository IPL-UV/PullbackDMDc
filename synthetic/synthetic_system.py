import pathlib

import numpy as np

def lat_grid(n=20):
    return np.linspace(-np.pi / 2, np.pi / 2, n)          # radians, poles included

def raw_patterns(n=20):
    phi = lat_grid(n)
    slow  = np.cos(phi - np.pi / 4)
    fast1 = np.sinc(2 * phi)
    fast2 = (0.5 - np.cos(10 * phi)) / (0.5 + np.abs(10 * phi))
    b     = 1.2 - np.cos(phi)
    return phi, slow, fast1, fast2, b

def make_W(n=20, seed=22):
    """W (unit-norm columns), W_inv, unit-norm B, grid (rad).
    Modes 4..n: Haar-random orthonormal basis of the complement of the 3
    structured modes (seed=None draws a fresh one)."""
    phi, slow, f1, f2, b = raw_patterns(n)
    unit = lambda v: v / np.linalg.norm(v)
    S = np.column_stack([unit(slow), unit(f1), unit(f2)])
    if np.linalg.matrix_rank(S) < 3:
        raise ValueError("structured patterns are linearly dependent on this grid")
    Q, _ = np.linalg.qr(S, mode="complete")
    Qp = Q[:, 3:]
    rng = np.random.default_rng(seed)
    R, T = np.linalg.qr(rng.standard_normal((n - 3, n - 3)))
    Qp = Qp @ (R * np.sign(np.diag(T)))
    W = np.column_stack([S, Qp])
    W_inv = np.vstack([np.linalg.pinv(S), Qp.T])
    return W, W_inv, unit(b), phi


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
# ----------------------------------------------------------------------------- style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",   # same family as the text
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42,                            # embed TrueType, editable text
})

# ----------------------------------------------------------------------------- figures
def fig_patterns(P, lat, path):
    spec = [  # label, colour (Okabe-Ito), marker, linestyle
        (r"slow mode ($\mathbf{w}_1$)",   "#0072B2", "o", "-"),
        (r"fast mode 1 ($\mathbf{w}_2$)", "#D55E00", "s", "-"),
        (r"fast mode 2 ($\mathbf{w}_3$)", "#009E73", "^", "-"),
        (r"forcing pattern ($B$)",        "#222222", "D", (0, (3, 1.6))),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.axhline(0, color="0.55", lw=0.6, zorder=1)
    ax.axvline(0, color="0.80", lw=0.5, ls=":", zorder=1)
    for k, (lab, c, m, ls) in enumerate(spec):
        hollow = (k == 3)
        ax.plot(lat, P[:, k], color=c, lw=1.1, ls=ls, marker=m, ms=3.4 if m != "D" else 3.0,
                mfc="white" if hollow else c, mec=c, mew=0.8, label=lab, zorder=3 + k)
    ax.set_xlim(-93, 93)
    ax.set_ylim(-0.30, 0.65)
    ax.xaxis.set_major_locator(FixedLocator(np.arange(-90, 91, 30)))
    ax.set_xticklabels(["90°S", "60°S", "30°S", "0°", "30°N", "60°N", "90°N"])
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Loading (unit norm)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              handlelength=2.4, columnspacing=1.2, handletextpad=0.5, borderaxespad=0.3)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def fig_correlation(P, path):
    G = P.T @ P                                                       # uncentered: no mean subtraction
    labels = [r"slow ($\mathbf{w}_1$)", r"fast 1 ($\mathbf{w}_2$)", r"fast 2 ($\mathbf{w}_3$)", r"$B$"]
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    im = ax.imshow(G, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    n = G.shape[0]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True); ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if G[i, j] > 0.6 else "#0B2A4A",
                    fontweight="bold" if i == j else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Uncentered correlation")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2.5, width=0.6, labelsize=7.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    figures_dir = pathlib.Path(__file__).resolve().parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    W, W_inv, B, phi = make_W(20)
    P = np.column_stack([W[:, 0], W[:, 1], W[:, 2], B])               # slow, fast 1, fast 2, B (unit norm)
    fig_patterns(P, np.rad2deg(phi), figures_dir / "fig_patterns.pdf")
    fig_correlation(P, figures_dir / "fig_correlation.pdf")
    print(np.round(P.T @ P, 3))
