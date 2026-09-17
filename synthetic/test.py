"""Sanity-check script: plots the structured modes and decay timescales of canonical_A
for the 20-dim case (CANONICAL_A_20_PARAMS)."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from _common import CANONICAL_A_20_PARAMS, CANONICAL_B_20, canonical_A, canonical_modes
from dynamics import make_shared_controls, simulate_dataset

FIGURES_DIR = pathlib.Path(__file__).resolve().parent/ "figures"

ENSEMBLE_STATE_INDICES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]


def plot_ensemble_spaghetti(out_path, n=20, n_members=10, T=200, noise_std=0.5):
    A = canonical_A(**CANONICAL_A_20_PARAMS)
    B = CANONICAL_B_20
    rng = np.random.default_rng(0)

    controls = make_shared_controls(n_members, 1, T, rng)
    trajectories = simulate_dataset(A, B, controls, noise_std, rng)

    t = np.arange(T + 1)
    fig, axes = plt.subplots(
        len(ENSEMBLE_STATE_INDICES), 1, figsize=(8, 1.2 * len(ENSEMBLE_STATE_INDICES)), sharex=True
    )
    for ax, i in zip(axes, ENSEMBLE_STATE_INDICES):
        for X in trajectories:
            ax.plot(t, X[i], linewidth=0.8, alpha=0.7)
        ax.set_ylabel(f"x[{i}]")
    axes[-1].set_xlabel("time step")
    fig.suptitle(f"{n_members}-member ensemble spaghetti plot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_ensemble_spaghetti(FIGURES_DIR / "canonical_A_ensemble_spaghetti.png")

    n = 3 + len(np.atleast_1d(CANONICAL_A_20_PARAMS["real_fast"]))
    modes = canonical_modes(n)
    A = canonical_A(**CANONICAL_A_20_PARAMS)
    eigvals, eigvecs = np.linalg.eig(A)
    decay_times = -1 / np.log(np.abs(eigvals))
    order = np.argsort(decay_times)[::-1]

    slow_i = np.argmax(eigvals.real)
    slow_vec = eigvecs[:, slow_i].real
    slow_vec /= np.linalg.norm(slow_vec)

    complex_mask = eigvals.imag > 1e-8
    pair_i = np.where(complex_mask)[0][np.argmax(np.abs(eigvals[complex_mask]))]
    pair_vec = eigvecs[:, pair_i]
    pair_real = pair_vec.real / np.linalg.norm(pair_vec.real)
    pair_imag = pair_vec.imag / np.linalg.norm(pair_vec.imag)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    idx = np.arange(n)
    axes[0].plot(idx, modes[0], marker="o", label="real slow mode")
    axes[0].plot(idx, modes[1], marker="o", label="pair mode 1 (peak)")
    axes[0].plot(idx, modes[2], marker="o", label="pair mode 2 (dip)")
    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].set_xlabel("state index")
    axes[0].set_ylabel("mode component")
    axes[0].set_title("Structured mode shapes (canonical_modes)")
    axes[0].legend()

    axes[1].plot(idx, slow_vec, marker="o", label="slow eigenvector (real)")
    axes[1].plot(idx, pair_real, marker="o", label="pair eigenvector (Re)")
    axes[1].plot(idx, pair_imag, marker="o", label="pair eigenvector (Im)")
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_xlabel("state index")
    axes[1].set_ylabel("eigenvector component")
    axes[1].set_title("Eigenvectors of A (np.linalg.eig)")
    axes[1].legend()

    axes[2].stem(np.arange(n), decay_times[order])
    axes[2].set_xlabel("mode (sorted, slowest to fastest)")
    axes[2].set_ylabel("decay time  -1/ln|eig|")
    axes[2].set_title("Decay timescales")
    axes[2].set_yscale("log")

    fig.tight_layout()
    out_path = FIGURES_DIR / "canonical_A_check.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
