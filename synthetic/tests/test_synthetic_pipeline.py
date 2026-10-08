"""System checks for the synthetic system: eigenstructure, forced/internal split and PullbackDMDc recovery."""

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ablation_data import (
    M,
    N,
    PHI,
    REFERENCE,
    SPINUP_CHUNK,
    decay_time,
    equal_budget,
    make_dataset,
    annual_years,
    record_years,
)
from plot_ablation_diagnostics import annual, lat_label
from methods import fit_pullback
from run_ablation_studies import centered_rms, plane_cosines, rms, slow_index
from plot_system_diagnostics import plot_modal_overview
from plot_style import save, zero_line
from support import noise_free_white_forcing, recovery_metrics

FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures" / "tests"

ENSEMBLE_STATE_INDICES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
LAT = np.rad2deg(PHI)


def unit_aligned(v, ref):
    v = v / np.linalg.norm(v)
    return v * np.sign(v @ ref)


def plot_system_check(out_path):
    s = REFERENCE
    W = s.W
    eigvals, eigvecs = np.linalg.eig(s.A)
    slow_vec = unit_aligned(eigvecs[:, slow_index(eigvals, s.lam1)].real, W[:, 0])
    pair_vec = eigvecs[:, np.argmin(np.abs(eigvals - s.eigvals[1]))]
    # the pair eigenvector is c (w2 + i w3) for some complex c
    c = np.vdot(W[:, 1] + 1j * W[:, 2], pair_vec) / np.vdot(W[:, 1] + 1j * W[:, 2], W[:, 1] + 1j * W[:, 2])
    pair_vec = pair_vec / c

    errors = {
        "slow eigenvector vs w1": 1 - abs(slow_vec @ W[:, 0]),
        "pair eigenvector Re vs w2": np.abs(pair_vec.real - W[:, 1]).max(),
        "pair eigenvector Im vs w3": np.abs(pair_vec.imag - W[:, 2]).max(),
        "pair plane vs span(w2, w3)": 1 - plane_cosines(np.column_stack([pair_vec.real, pair_vec.imag]), W[:, 1:3]).min(),
    }
    print("--- system check: eigenstructure of A = W Lambda_R W^-1 ---")
    for name, err in errors.items():
        print(f"  {name:<28s} {err:.2e}")

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.5))
    for j, name in enumerate((r"$w_1$ slow", r"$w_2$ pair", r"$w_3$ pair")):
        axes[0].plot(LAT, W[:, j], marker="o", label=name)
    axes[0].plot(LAT, s.b, marker="D", color="k", linestyle="--", label=r"$\hat b$ forcing")
    axes[0].set_title("Structured patterns")

    for j, (vec, name) in enumerate(((slow_vec, "slow eigvec"), (pair_vec.real, "pair eigvec Re"),
                                     (pair_vec.imag, "pair eigvec Im"))):
        axes[1].plot(LAT, W[:, j], color=f"C{j}", linewidth=2.4, alpha=0.4)
        axes[1].plot(LAT, vec, color=f"C{j}", marker="x", linestyle="none", label=name)
    axes[1].set_title("Eigenvectors of A (x) vs patterns (lines)")

    for ax in axes[:2]:
        zero_line(ax)
        ax.set_xlabel("latitude (deg)")
        ax.legend(fontsize=8)

    taus = decay_time(s.eigvals) / 12
    order = np.argsort(taus)[::-1]
    axes[2].stem(np.arange(M), taus[order])
    axes[2].set_yscale("log")
    axes[2].set_xlabel("mode (slowest to fastest)")
    axes[2].set_ylabel("decay time (yr)")
    axes[2].set_title(rf"Decay times: $\tau_1$={taus[0]:.3g} yr, $\tau_p$={taus[1]:.3g} yr, "
                      rf"period {2 * np.pi / s.theta / 12:.3g} yr")

    theta = np.linspace(0, 2 * np.pi, 400)
    axes[3].plot(np.cos(theta), np.sin(theta), color="0.85")
    axes[3].scatter(s.eigvals.real, s.eigvals.imag, s=60, facecolors="none", edgecolors="C0", label="specified")
    axes[3].scatter(eigvals.real, eigvals.imag, s=20, marker="x", color="C3", label="eig(A)")
    axes[3].set_aspect("equal")
    axes[3].set_title("Eigenvalues")
    axes[3].legend(fontsize=8)
    save(fig, out_path)

    for name, err in errors.items():
        assert err < 1e-10, f"{name}: {err:.2e}"


def plot_ensemble_spaghetti(out_path, n_realizations=10):
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=n_realizations)
    fig, axes = plt.subplots(len(ENSEMBLE_STATE_INDICES), 1, figsize=(8, 1.3 * len(ENSEMBLE_STATE_INDICES)),
                             sharex=True)
    years = annual_years(ds.system)
    for ax, i in zip(axes, ENSEMBLE_STATE_INDICES[::-1]):
        for member in annual(ds.data[:, :, i], axis=-1):
            ax.plot(years, member, linewidth=0.7, alpha=0.7)
        ax.plot(years, annual(ds.forced[:, i], axis=-1), color="k", linewidth=1.5)
        ax.set_ylabel(lat_label(i))
    axes[-1].set_xlabel("year")
    fig.suptitle(f"{n_realizations} realizations (annual means), forced in black, SNR {ds.snr:.3g}")
    save(fig, out_path)


def check_dmdc_recovery(out_path, n_steps=20000, tol=1e-8):
    s, y, data = noise_free_white_forcing(n_steps)

    model = fit_pullback(data, y[:, None], y[s.spinup:, None], s.spinup)
    patterns = model.compute_rotated_modes()["spatial_patterns"]
    eigvals = model.eigvals
    pair_true = s.eigvals[1]

    k_slow = slow_index(eigvals, s.lam1)
    pair_starts = [k for k in range(M - 1) if model._is_conjugate_pair(eigvals, k, M)]
    k_pair = min(pair_starts, key=lambda k: abs(abs(eigvals[k]) - s.rho))
    slow_mode = unit_aligned(patterns[:, k_slow], s.W[:, 0])
    pair_plane = patterns[:, k_pair:k_pair + 2]

    errors = {
        "slow eigenvalue": abs(eigvals[k_slow] - s.lam1),
        "pair eigenvalue": min(abs(eigvals[k_pair] - pair_true), abs(np.conj(eigvals[k_pair]) - pair_true)),
        "B vs B_true": np.abs(model.B.ravel() - s.B).max(),
        "slow mode shape": 1 - abs(slow_mode @ s.W[:, 0]),
        "pair mode plane": 1 - plane_cosines(pair_plane, s.W[:, 1:3]).min(),
    }
    print(f"--- DMDc recovery: noise-free, white forcing, {n_steps} steps after {s.spinup} spinup ---")
    for name, err in errors.items():
        print(f"  {name:<20s} {err:.2e}")
    print(f"  {'predict vs data':<20s} {np.abs(model.predict() - data).max():.2e}  (not asserted)")
    print(f"  {'A entrywise':<20s} {np.abs(model.A - s.A).max():.2e}  (not asserted: the 17 near-zero")
    print("                                        complement modes are not separately excited)")

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.5))
    theta = np.linspace(0, 2 * np.pi, 400)
    axes[0].plot(np.cos(theta), np.sin(theta), color="0.85")
    axes[0].scatter(s.eigvals.real, s.eigvals.imag, s=80, facecolors="none", edgecolors="C0", label="true")
    axes[0].scatter(eigvals.real, eigvals.imag, s=30, marker="x", color="C3", label="recovered")
    axes[0].set_aspect("equal")
    axes[0].set_title("Eigenvalues")

    axes[1].plot(LAT, s.B, marker="o", label=r"true $B$")
    axes[1].plot(LAT, model.B.ravel(), marker="x", linestyle="--", label="recovered B")
    axes[1].set_title("Forcing pattern")

    axes[2].plot(LAT, s.W[:, 0], marker="o", label=r"true $w_1$")
    axes[2].plot(LAT, slow_mode, marker="x", linestyle="--", label="recovered")
    axes[2].set_title("Slow mode shape")

    projector = np.linalg.qr(pair_plane)[0]
    projector = projector @ projector.T
    for j, color in ((1, "C0"), (2, "C1")):
        axes[3].plot(LAT, s.W[:, j], color=color, marker="o", label=f"true $w_{j + 1}$")
        axes[3].plot(LAT, projector @ s.W[:, j], color=color, marker="x", linestyle="--",
                     label="projected on recovered plane")
    axes[3].set_title("Pair patterns vs recovered plane")
    for ax in axes[1:]:
        zero_line(ax)
        ax.set_xlabel("latitude (deg)")
    for ax in axes:
        ax.legend(fontsize=8)
    save(fig, out_path)

    for name, err in errors.items():
        assert err < tol, f"{name}: {err:.2e} exceeds tolerance {tol:.0e}"


def direct_simulation(ds, seed, n_direct):
    s = ds.system
    R = ds.internal.shape[0]
    spinup_rng, record_rng = (np.random.default_rng(q) for q in np.random.SeedSequence(seed).spawn(2))
    eps = [spinup_rng.standard_normal((min(SPINUP_CHUNK, s.spinup - start), R, M))[:, :n_direct].copy()
           for start in range(0, s.spinup, SPINUP_CHUNK)]
    eps.append(record_rng.standard_normal((N, R, M))[:, :n_direct].copy())
    xi = (np.concatenate(eps) * np.sqrt(s.noise_variances)) @ s.W.T
    x = np.zeros((n_direct, M))
    X = np.empty((N, n_direct, M))
    for t in range(s.spinup + N):
        x = x @ s.A.T + ds.y[t] * s.B + xi[t]
        if t >= s.spinup:
            X[t - s.spinup] = x
    return X.transpose(1, 0, 2)


def check_forced_internal_split(spaghetti_path, convergence_path, modal_path, n_realizations=256, seed=0,
                                n_direct=4):
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=n_realizations, seed=seed)
    direct = direct_simulation(ds, seed, n_direct)
    direct_err = np.abs(direct - ds.data[:n_direct]).max() / np.abs(direct).max()

    counts = [k for k in (4, 16, 64, 256) if k <= n_realizations]
    mean_errors = [rms(ds.data[:k].mean(axis=0) - ds.forced) for k in counts]
    slope = float(np.polyfit(np.log(counts), np.log(mean_errors), 1)[0])
    rms_forced = centered_rms(ds.forced)  # the same V_f / M the SNR is built from (see centered_rms)

    print(f"--- forced/internal split: {n_realizations} realizations, SNR {ds.snr:.3g} ---")
    print(f"  parts vs direct x-space sim  {direct_err:.2e}  (relative)")
    for k, err in zip(counts, mean_errors):
        print(f"  ensemble mean K={k:<4d}         {err:.5f}  ({err / rms_forced:.1%} of rms forced)")
    print(f"  convergence log-log slope    {slope:.3f}  (theory -0.5)")

    n_shown = 10
    fig, axes = plt.subplots(len(ENSEMBLE_STATE_INDICES), 3, figsize=(13, 1.3 * len(ENSEMBLE_STATE_INDICES)),
                             sharex=True, sharey="row")
    years = annual_years(ds.system)
    for row, i in enumerate(ENSEMBLE_STATE_INDICES[::-1]):
        for full, internal in zip(annual(ds.data[:n_shown, :, i], axis=-1), annual(ds.internal[:n_shown, :, i], axis=-1)):
            axes[row, 0].plot(years, full, linewidth=0.7, alpha=0.7)
            axes[row, 2].plot(years, internal, linewidth=0.7, alpha=0.7)
        axes[row, 1].plot(years, annual(ds.forced[:, i], axis=-1), color="C1", linewidth=1.2)
        axes[row, 0].set_ylabel(lat_label(i))
    for col, title in enumerate(("full signal", "forced response", "internal variability")):
        axes[0, col].set_title(title)
        axes[-1, col].set_xlabel("year")
    fig.suptitle(f"{n_shown} of {n_realizations} realizations (annual means): full = forced + internal")
    save(fig, spaghetti_path)
    plot_modal_overview(ds, modal_path, n_shown)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].loglog(counts, mean_errors, marker="o", label="rms|mean_K - forced|")
    axes[0].loglog(counts, mean_errors[0] * np.sqrt(counts[0] / np.array(counts, dtype=float)),
                   linestyle="--", color="0.6", label=r"$1/\sqrt{K}$")
    axes[0].set_xlabel("realizations K")
    axes[0].set_title(f"Ensemble mean converges to forced (slope {slope:.2f})")
    axes[0].legend()
    axes[1].plot(LAT, ds.forced.var(axis=0), marker="o", label="forced")
    axes[1].plot(LAT, ds.internal.var(axis=1).mean(axis=0), marker="o", label="internal")
    axes[1].plot(LAT, ds.data.var(axis=1).mean(axis=0), marker="x", linestyle="--", label="full")
    axes[1].set_xlabel("latitude (deg)")
    axes[1].set_ylabel("variance over time")
    axes[1].set_title("Variance decomposition")
    axes[1].legend()
    save(fig, convergence_path)

    assert direct_err < 1e-9, f"parts vs direct simulation: {direct_err:.2e}"
    assert slope < -0.4, f"ensemble mean converges too slowly: slope {slope:.3f}"
    assert mean_errors[-1] / rms_forced < 0.15, (
        f"ensemble mean at K={counts[-1]} is {mean_errors[-1] / rms_forced:.1%} of rms forced")


def check_forced_internal_recovery(spaghetti_path, mse_path, n_realizations=100, zoom_years=25):
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=n_realizations)
    m = recovery_metrics(ds)
    print(f"--- forced/internal recovery: starting point, SNR {ds.snr:.3g}, {n_realizations} realizations ---")
    for key, name, fmt in (("forced_corr", "corr forced", ".4f"), ("forced_err", "forced rms error", ".2%"),
                           ("internal_err", "internal rms error", ".2%"), ("slow_eig", "slow eigenvalue", ".5f")):
        q25, q50, q75 = np.percentile(m[key], [25, 50, 75])
        print(f"  {name:<20s} median {q50:{fmt}}  (IQR {q25:{fmt}} - {q75:{fmt}})")
    print(f"  true slow eigenvalue {REFERENCE.lam1:.5f}  (skill reported, not asserted)")

    t = record_years(ds.system)
    zoom = 12 * zoom_years
    fig, axes = plt.subplots(len(ENSEMBLE_STATE_INDICES), 3, figsize=(14, 1.3 * len(ENSEMBLE_STATE_INDICES)),
                             sharey="row")
    forced_est = m["forced_est"][0]
    internal_est = ds.data[0] - forced_est
    for row, i in enumerate(ENSEMBLE_STATE_INDICES[::-1]):
        axes[row, 0].plot(t, ds.data[0, :, i], linewidth=0.5, color="0.4")
        axes[row, 1].plot(t, ds.forced[:, i], color="C0", linewidth=2.4, alpha=0.5, label="true")
        axes[row, 1].plot(t, forced_est[:, i], color="C3", linestyle="--", linewidth=1.2, label="estimated")
        axes[row, 2].plot(t[:zoom], ds.internal[0, :zoom, i], color="C0", linewidth=2.0, alpha=0.5, label="true")
        axes[row, 2].plot(t[:zoom], internal_est[:zoom, i], color="C3", linestyle="--", linewidth=0.9,
                          label="estimated")
        axes[row, 0].set_ylabel(lat_label(i))
    for col, title in enumerate(("full signal (model input)", "forced response",
                                 f"internal variability (first {zoom_years} yr)")):
        axes[0, col].set_title(title)
        axes[-1, col].set_xlabel("year")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(f"PullbackDMDc recovery, realization 0, SNR {ds.snr:.3g}: true vs estimated")
    save(fig, spaghetti_path)

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
    axes[0].semilogy(LAT, np.median(m["forced_mse"], axis=0), marker="o", linewidth=2.4, alpha=0.5,
                     label="forced MSE (median)")
    axes[0].semilogy(LAT, np.median(m["internal_mse"], axis=0), marker="x", linestyle="--",
                     label="internal MSE (mirrors forced)")
    axes[0].semilogy(LAT, ds.forced.var(axis=0), color="C0", alpha=0.3, linestyle=":", label="var(true forced)")
    axes[0].semilogy(LAT, ds.internal.var(axis=1).mean(axis=0), color="C1", alpha=0.3, linestyle=":",
                     label="var(true internal)")
    axes[0].set_xlabel("latitude (deg)")
    axes[0].set_title("MSE per latitude")
    axes[0].legend(fontsize=8)

    axes[1].hist(m["forced_err"], bins=20)
    axes[1].set_title("Forced relative rms error")

    axes[2].hist(m["forced_corr"], bins=20)
    axes[2].set_title("Forced correlation")

    axes[3].hist(m["slow_eig"], bins=20)
    axes[3].axvline(REFERENCE.lam1, color="C3", linestyle="--", label=r"true $\lambda_1$")
    axes[3].set_title("Slow eigenvalue estimate")
    axes[3].legend(fontsize=8)
    for ax in axes[1:]:
        ax.set_ylabel("realizations")
    fig.suptitle(f"PullbackDMDc recovery over {n_realizations} realizations at the starting point (SNR {ds.snr:.3g})")
    save(fig, mse_path)

    assert m["mirror"].max() < 1e-10, f"decomposition errors are not mirrored: {m['mirror'].max():.2e}"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_system_check(FIGURES_DIR / "system_check.png")
    plot_ensemble_spaghetti(FIGURES_DIR / "ensemble_spaghetti.png")
    check_dmdc_recovery(FIGURES_DIR / "dmdc_recovery_check.png")
    check_forced_internal_split(FIGURES_DIR / "forced_internal_spaghetti.png",
                                FIGURES_DIR / "forced_internal_convergence.png",
                                FIGURES_DIR / "modal_overview.png")
    check_forced_internal_recovery(FIGURES_DIR / "forced_internal_recovery.png",
                                   FIGURES_DIR / "forced_internal_recovery_mse.png")


if __name__ == "__main__":
    main()
