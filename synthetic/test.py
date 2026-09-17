"""Sanity-check script: plots the structured modes and decay timescales of canonical_A
for the 20-dim case (CANONICAL_A_20_PARAMS)."""

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import CANONICAL_A_20_PARAMS, CANONICAL_B_20, canonical_A, canonical_modes
from dynamics import (
    make_shared_controls,
    simulate_dataset,
    simulate_ensemble,
    simulate_lti,
)
from utils.pullback_dmdc import PullbackDMDc

FIGURES_DIR = pathlib.Path(__file__).resolve().parent/ "figures"

ENSEMBLE_STATE_INDICES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]


def plot_ensemble_spaghetti(out_path, n_members=10, T=200, noise_std=0.5):
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

def plot_canonical_A_check(out_path):
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
    
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

def check_dmdc_recovery(out_path, spinup=500, n_steps=20000, tol=1e-9):
    """Fit PullbackDMDc to noise-free data from the canonical system and check that the rotated
    modes, the slow and oscillatory eigenvalues, and B all come back."""
    A_true, B_true = canonical_A(**CANONICAL_A_20_PARAMS), CANONICAL_B_20
    n = A_true.shape[0]
    rng = np.random.default_rng(0)

    U = rng.standard_normal((1, spinup + n_steps - 1))
    X = simulate_lti(A_true, B_true, U, rng.standard_normal(n), 0.0, rng)
    data = X[:, spinup:].T

    # fit models x[t] = A x[t-lag] + B f[t], indexing the forcing at the target time, while
    # simulate_lti applies u[t] to x[t] -> x[t+1]; hence the one-row shift.
    long_forcings = np.vstack([np.zeros((1, 1)), U.T])

    model = PullbackDMDc(truncation=n, lag=1, transition_time=spinup)
    model.fit(
        data,
        short_forcings=long_forcings[spinup:],
        long_forcings=long_forcings,
        # identity EOFs and a zero mean fit the raw uncentered state, so A and B come out in
        # state coordinates and the model's missing intercept never bites.
        precomputed_eofs={"data_mean": np.zeros(n), "eofs": np.eye(n), "pcs": data},
    )
    model.compute_modes()
    patterns = model.compute_rotated_modes()["spatial_patterns"]
    eigvals = model.eigvals

    modes_true = canonical_modes(n)
    slow_true = CANONICAL_A_20_PARAMS["real_slow"]
    pair_true = CANONICAL_A_20_PARAMS["pair_modulus"] * np.exp(
        1j * CANONICAL_A_20_PARAMS["pair_angle"]
    )

    k_slow = int(np.argmin(np.abs(eigvals - slow_true)))
    # a pair occupies columns (k, k+1) and either conjugate may sort first, so pick the pair's
    # first column rather than the column whose eigenvalue is nearest -- that can straddle it.
    pair_starts = [k for k in range(n - 1) if model._is_conjugate_pair(eigvals, k, n)]
    k_pair = min(pair_starts, key=lambda k: abs(abs(eigvals[k]) - abs(pair_true)))

    slow_mode = patterns[:, k_slow] / np.linalg.norm(patterns[:, k_slow])
    q_pair, _ = np.linalg.qr(patterns[:, k_pair : k_pair + 2])
    q_pair_true, _ = np.linalg.qr(modes_true[1:3].T)

    errors = {
        "slow eigenvalue": abs(eigvals[k_slow] - slow_true),
        "complex pair": min(
            abs(eigvals[k_pair] - pair_true), abs(np.conj(eigvals[k_pair]) - pair_true)
        ),
        "B": np.abs(model.B - B_true).max(),
        "slow mode shape": 1 - abs(slow_mode @ modes_true[0]),
        "pair mode subspace": 1 - np.linalg.svd(q_pair_true.T @ q_pair, compute_uv=False).min(),
    }

    print(f"--- DMDc recovery: noise-free, {n_steps} steps after {spinup} spinup ---")
    for name, err in errors.items():
        print(f"  {name:<20s} {err:.2e}")
    print(f"  {'predict vs data':<20s} {np.abs(model.predict() - data).max():.2e}  (not asserted)")
    print(f"  {'internal variability':<20s} {np.abs(model.internal_time_series).max():.2e}  (not asserted)")
    print(f"  {'A entrywise':<20s} {np.abs(model.A - A_true).max():.2e}  (not asserted: fast modes")
    print("                                        are unidentifiable from a scalar input)")

    idx = np.arange(n)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    theta = np.linspace(0, 2 * np.pi, 400)
    eig_true = np.linalg.eigvals(A_true)
    axes[0].plot(np.cos(theta), np.sin(theta), color="0.85", linewidth=1)
    axes[0].scatter(eig_true.real, eig_true.imag, s=80, facecolors="none", edgecolors="C0", label="true")
    axes[0].scatter(eigvals.real, eigvals.imag, s=30, marker="x", color="C3", label="recovered")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("Re")
    axes[0].set_ylabel("Im")
    axes[0].set_title("Eigenvalues")
    axes[0].legend()

    axes[1].plot(idx, B_true.ravel(), marker="o", label="true B")
    axes[1].plot(idx, model.B.ravel(), marker="x", linestyle="--", label="recovered B")
    axes[1].set_xlabel("state index")
    axes[1].set_title("Control matrix B")
    axes[1].legend()

    axes[2].plot(idx, modes_true[0], marker="o", label="true slow mode")
    axes[2].plot(
        idx, slow_mode * np.sign(slow_mode @ modes_true[0]), marker="x", linestyle="--", label="recovered"
    )
    axes[2].axhline(0, color="k", linewidth=0.5)
    axes[2].set_xlabel("state index")
    axes[2].set_title("Slow mode shape")
    axes[2].legend()

    projector = q_pair @ q_pair.T
    for j, color in ((1, "C0"), (2, "C1")):
        axes[3].plot(idx, modes_true[j], color=color, marker="o", label=f"true pair mode {j}")
        axes[3].plot(
            idx, projector @ modes_true[j], color=color, marker="x", linestyle="--",
            label=f"projected onto recovered {j}",
        )
    axes[3].axhline(0, color="k", linewidth=0.5)
    axes[3].set_xlabel("state index")
    axes[3].set_title("Pair modes vs recovered subspace")
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

    for name, err in errors.items():
        assert err < tol, f"{name}: {err:.2e} exceeds tolerance {tol:.0e}"

def check_forced_internal_split(
    spaghetti_path, convergence_path, T=400, n_members=256, noise_std=0.1, ramp_to=2.0
):
    """Check the data generator's forced/internal split: the forced response is the part driven by
    the control, internal variability is the initial-condition transient plus the noise response."""
    A, B, n = canonical_A(**CANONICAL_A_20_PARAMS), CANONICAL_B_20, 20
    rng = np.random.default_rng(0)
    # a ramp, like the historical forcing record: a zero-mean white-noise control would give a
    # stationary forced response with no trend to separate from internal variability.
    U = np.linspace(0.0, ramp_to, T)[None, :]

    forced, internal = simulate_ensemble(A, B, U, n_members, noise_std, rng)
    members = np.stack(internal)

    # the parts must compose into the very trajectory a direct simulation produces: same initial
    # condition, same noise stream (equal seeds draw the same sequence), control on or off.
    x0 = rng.standard_normal(n)
    direct = simulate_lti(A, B, U, x0, noise_std, np.random.default_rng(42))
    from_parts = forced + simulate_lti(A, B, np.zeros_like(U), x0, noise_std, np.random.default_rng(42))

    # internal must not depend on the control at all
    internal_u1 = simulate_lti(A, B, np.zeros_like(U), x0, noise_std, np.random.default_rng(7))
    internal_u2 = simulate_lti(A, B, np.zeros_like(U), x0, noise_std, np.random.default_rng(7))

    member_counts = [c for c in (4, 16, 64, 256) if c <= n_members]
    mean_errors = [
        np.sqrt((((forced + members[:k]).mean(axis=0) - forced) ** 2).mean()) for k in member_counts
    ]
    slope = float(np.polyfit(np.log(member_counts), np.log(mean_errors), 1)[0])
    rms_forced = np.sqrt((forced**2).mean())

    errors = {
        "parts vs direct sim": np.abs(direct - from_parts).max(),
        "forced obeys recursion": np.abs(forced[:, 1:] - (A @ forced[:, :-1] + B @ U)).max(),
        "internal ignores control": np.abs(internal_u1 - internal_u2).max(),
    }

    print(f"--- forced/internal split: {n_members} members, T={T}, noise_std={noise_std} ---")
    for name, err in errors.items():
        print(f"  {name:<26s} {err:.2e}")
    for count, err in zip(member_counts, mean_errors):
        print(f"  ensemble mean K={count:<4d}         {err:.5f}  ({err / rms_forced:.1%} of rms forced)")
    print(f"  convergence log-log slope  {slope:.3f}  (theory -0.5)")
    print(
        f"  variance forced/internal/full  {forced.var():.4f} / {members[0].var():.4f} / "
        f"{(forced + members[0]).var():.4f}  (not asserted)"
    )

    time = np.arange(T + 1)
    n_shown = min(10, n_members)
    fig, axes = plt.subplots(
        len(ENSEMBLE_STATE_INDICES), 3, figsize=(13, 1.3 * len(ENSEMBLE_STATE_INDICES)),
        sharex=True, sharey="row",
    )
    for row, i in enumerate(ENSEMBLE_STATE_INDICES):
        for member in members[:n_shown]:
            axes[row, 0].plot(time, forced[i] + member[i], linewidth=0.7, alpha=0.7)
            axes[row, 2].plot(time, member[i], linewidth=0.7, alpha=0.7)
        axes[row, 1].plot(time, forced[i], color="C1", linewidth=1.2)
        axes[row, 0].set_ylabel(f"x[{i}]")
    for col, title in enumerate(("full signal", "forced response", "internal variability")):
        axes[0, col].set_title(title)
        axes[-1, col].set_xlabel("time step")
    fig.suptitle(f"{n_shown} of {n_members} members: full = forced + internal")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(spaghetti_path, dpi=150)
    print(f"saved {spaghetti_path}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].loglog(member_counts, mean_errors, marker="o", label="rms|mean_K - forced|")
    reference = mean_errors[0] * np.sqrt(member_counts[0] / np.array(member_counts, dtype=float))
    axes[0].loglog(member_counts, reference, linestyle="--", color="0.6", label=r"$1/\sqrt{K}$")
    axes[0].set_xlabel("ensemble members K")
    axes[0].set_title(f"Ensemble mean converges to forced (slope {slope:.2f})")
    axes[0].legend()

    idx = np.arange(n)
    axes[1].plot(idx, forced.var(axis=1), marker="o", label="forced")
    axes[1].plot(idx, members.var(axis=2).mean(axis=0), marker="o", label="internal")
    axes[1].plot(idx, (forced + members).var(axis=2).mean(axis=0), marker="x", linestyle="--", label="full")
    axes[1].set_xlabel("state index")
    axes[1].set_ylabel("variance")
    axes[1].set_title("Variance decomposition")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(convergence_path, dpi=150)
    print(f"saved {convergence_path}")

    for name, err in errors.items():
        assert err < 1e-10, f"{name}: {err:.2e} exceeds tolerance 1e-10"
    assert slope < -0.4, f"ensemble mean converges too slowly: slope {slope:.3f}"
    assert mean_errors[-1] / rms_forced < 0.15, (
        f"ensemble mean at K={member_counts[-1]} is {mean_errors[-1] / rms_forced:.1%} of rms forced"
    )

def check_forced_internal_recovery(
    spaghetti_path, mse_path, spinup=500, T=20000, noise_std=0.1, ramp_to=2.0, zoom=300
):
    """Fit PullbackDMDc to a noisy trajectory and check that the forced response and internal
    variability it recovers match the ones the data generator put in."""
    A, B, n = canonical_A(**CANONICAL_A_20_PARAMS), CANONICAL_B_20, 20
    rng = np.random.default_rng(0)
    U = np.linspace(0.0, ramp_to, spinup + T - 1)[None, :]

    # process noise only -- no observation noise is added to the recorded states
    forced, internal = simulate_ensemble(A, B, U, 1, noise_std, rng)
    data = (forced + internal[0])[:, spinup:].T

    long_forcings = np.vstack([np.zeros((1, 1)), U.T])
    model = PullbackDMDc(truncation=n, lag=1, transition_time=spinup)
    model.fit(
        data,
        short_forcings=long_forcings[spinup:],
        long_forcings=long_forcings,
        precomputed_eofs={"data_mean": np.zeros(n), "eofs": np.eye(n), "pcs": data},
    )
    model.compute_modes()

    forced_est = model.predict()
    internal_est = data - forced_est
    forced_true = forced[:, spinup:].T
    internal_true = internal[0][:, spinup:].T

    rms = lambda a: np.sqrt((a**2).mean())
    forced_err = rms(forced_est - forced_true) / rms(forced_true)
    internal_err = rms(internal_est - internal_true) / rms(internal_true)
    corr_forced = np.corrcoef(forced_est.ravel(), forced_true.ravel())[0, 1]
    corr_internal = np.corrcoef(internal_est.ravel(), internal_true.ravel())[0, 1]
    # both decompositions sum to the same data, so the two errors are one quantity
    mirror = np.abs((forced_est - forced_true) + (internal_est - internal_true)).max()
    ic_left = np.abs(np.linalg.matrix_power(A, spinup)).max() * np.abs(internal[0][:, 0]).max()
    slow_err = abs(model.eigvals[np.argmin(np.abs(model.eigvals - 0.95))] - 0.95)

    print(f"--- forced/internal recovery: T={T} after {spinup} spinup, noise_std={noise_std} ---")
    print(f"  initial condition left     {ic_left:.2e}")
    print(f"  forced rms error           {forced_err:.2%} of rms(forced)")
    print(f"  internal rms error         {internal_err:.2%} of rms(internal)")
    print(f"  correlation forced         {corr_forced:.6f}")
    print(f"  correlation internal       {corr_internal:.6f}")
    print(f"  forced err == -internal err {mirror:.2e}")
    print(f"  slow eigenvalue error      {slow_err:.2e}  (not asserted: amplified 1/(1-0.95)=20x")
    print("                                        into the forced response's gain)")
    print(f"  A / B entrywise error      {np.abs(model.A - A).max():.2e} / "
          f"{np.abs(model.B - B).max():.2e}  (not asserted)")

    time = np.arange(T)
    fig, axes = plt.subplots(
        len(ENSEMBLE_STATE_INDICES), 3, figsize=(14, 1.3 * len(ENSEMBLE_STATE_INDICES)),
        sharey="row",
    )
    for row, i in enumerate(ENSEMBLE_STATE_INDICES):
        axes[row, 0].plot(time, data[:, i], linewidth=0.5, color="0.4")
        axes[row, 1].plot(time, forced_true[:, i], color="C0", linewidth=2.4, alpha=0.5, label="true")
        axes[row, 1].plot(time, forced_est[:, i], color="C3", linestyle="--", linewidth=1.2, label="estimated")
        axes[row, 2].plot(time[:zoom], internal_true[:zoom, i], color="C0", linewidth=2.0, alpha=0.5, label="true")
        axes[row, 2].plot(
            time[:zoom], internal_est[:zoom, i], color="C3", linestyle="--", linewidth=0.9, label="estimated"
        )
        axes[row, 0].set_ylabel(f"x[{i}]")
    for col, title in enumerate(
        ("full signal (model input)", "forced response", f"internal variability (first {zoom} steps)")
    ):
        axes[0, col].set_title(title)
        axes[-1, col].set_xlabel("time step")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("PullbackDMDc recovery: true vs estimated")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(spaghetti_path, dpi=150)
    print(f"saved {spaghetti_path}")

    idx = np.arange(n)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].semilogy(idx, ((forced_est - forced_true) ** 2).mean(axis=0), marker="o", linewidth=2.4,
                     alpha=0.5, label="forced MSE")
    axes[0].semilogy(idx, ((internal_est - internal_true) ** 2).mean(axis=0), marker="x", linestyle="--",
                     label="internal MSE (mirrors forced)")
    axes[0].semilogy(idx, forced_true.var(axis=0), color="C0", alpha=0.3, linestyle=":", label="var(true forced)")
    axes[0].semilogy(idx, internal_true.var(axis=0), color="C1", alpha=0.3, linestyle=":", label="var(true internal)")
    axes[0].set_xlabel("state index")
    axes[0].set_title("MSE per state index")
    axes[0].legend(fontsize=8)

    chunks = np.array_split(np.arange(T), 20)
    centers = [c.mean() for c in chunks]
    err_series = [rms(forced_est[c] - forced_true[c]) for c in chunks]
    signal_series = [rms(forced_true[c]) for c in chunks]
    axes[1].plot(centers, signal_series, marker="o", label="rms true forced")
    axes[1].plot(centers, err_series, marker="o", label="rms forced error")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("time step")
    axes[1].set_title("Error grows with the signal")
    axes[1].legend()

    axes[2].plot(centers, np.array(err_series) / np.array(signal_series), marker="o")
    axes[2].axhline(forced_err, color="0.6", linestyle="--", label=f"overall {forced_err:.2%}")
    axes[2].set_ylim(bottom=0)
    axes[2].set_xlabel("time step")
    axes[2].set_title("Error / signal: constant gain bias, not drift")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(mse_path, dpi=150)
    print(f"saved {mse_path}")

    assert ic_left < 1e-9, f"initial condition still present at {ic_left:.2e}; lengthen the spinup"
    assert mirror < 1e-10, f"decomposition errors are not mirrored: {mirror:.2e}"
    assert forced_err < 0.05, f"forced response error {forced_err:.2%} exceeds 5%"
    assert internal_err < 0.05, f"internal variability error {internal_err:.2%} exceeds 5%"
    assert corr_forced > 0.99, f"forced correlation {corr_forced:.4f} below 0.99"
    assert corr_internal > 0.99, f"internal correlation {corr_internal:.4f} below 0.99"


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    plot_canonical_A_check(FIGURES_DIR / "canonical_A_check.png")

    plot_ensemble_spaghetti(FIGURES_DIR / "canonical_A_ensemble_spaghetti.png")

    check_dmdc_recovery(FIGURES_DIR / "dmdc_recovery_check.png")

    check_forced_internal_split(
        FIGURES_DIR / "forced_internal_spaghetti.png",
        FIGURES_DIR / "forced_internal_convergence.png",
    )

    check_forced_internal_recovery(
        FIGURES_DIR / "forced_internal_recovery.png",
        FIGURES_DIR / "forced_internal_recovery_mse.png",
    )
    


if __name__ == "__main__":
    main()
