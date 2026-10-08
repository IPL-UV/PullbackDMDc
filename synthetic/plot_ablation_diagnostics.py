"""Diagnostic plots for the ablation datasets (ablation_data.py), written to figures/diagnostics/data/."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from plot_style import save, zero_line
from ablation_data import (
    BASE_OVERLAP,
    annual_years,
    N,
    PARTIAL_SNR_COMPONENTS,
    PHI,
    REFERENCE,
    SLOW_TIMESCALE_HOLDS,
    decay_time_yr,
    empirical_snr,
    equal_budget,
    make_dataset,
    modal_coordinates,
    pair_plane_overlap,
    record_years,
    partial_snr_sweep,
    slow_timescale_sweep,
    forcing_overlap_sweep,
    spatial_overlap_sweep,
    total_snr_sweep,
)

FIGURES_DIR = pathlib.Path(__file__).resolve().parent / "figures" / "diagnostics" / "data"

N_REALIZATIONS = 10
CALIBRATION_LATS = (18, 14, 10, 4)  # 81N, 43N, 5N, 52S
SWEEP_LATS = (14, 10)
MAX_LAG = 120


def lat_label(i):
    deg = np.rad2deg(PHI[i])
    return f"{abs(deg):.0f}°{'N' if deg >= 0 else 'S'}"


def annual(x, axis=-2):
    axis = axis % x.ndim
    shape = list(x.shape)
    shape[axis:axis + 1] = [shape[axis] // 12, 12]
    return x.reshape(shape).mean(axis=axis + 1)


def autocorrelation(z, max_lag):
    z = z - z.mean(axis=-1, keepdims=True)
    var = (z**2).mean()
    return np.array([(z[..., lag:] * z[..., : z.shape[-1] - lag]).mean() / var for lag in range(max_lag + 1)])


def standardized(v):
    return (v - v.mean()) / v.std()


def theory_acf(eig, lags):
    return np.abs(eig) ** lags * np.cos(np.angle(eig) * lags)


def plot_realizations(ax, ds, lat):
    years = annual_years(ds.system)
    for member in annual(ds.data[:, :, lat], axis=-1):
        ax.plot(years, member, linewidth=0.5, alpha=0.6)
    ax.plot(years, annual(ds.forced[:, lat], axis=-1), color="k", linewidth=1.5, label="forced")


def label_row(ax, ds):
    ax.set_ylabel(f"{ds.param_name}\n= {ds.param_value:.3g}\nSNR {ds.snr:.3g}")


def plot_calibration(ds, path):
    s = ds.system
    z_data = modal_coordinates(ds, ds.data)
    z_forced = modal_coordinates(ds, ds.forced)
    z_internal = modal_coordinates(ds, ds.internal)
    lags = np.arange(MAX_LAG + 1)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    record = record_years(s)
    t_hist = record[0] + np.arange(-s.history, N) / 12
    axes[0, 0].plot(t_hist, ds.y[s.spinup - s.history:], color="k")
    axes[0, 0].axvspan(t_hist[0], record[0], color="0.9", label="forcing history")
    axes[0, 0].set_title("forcing y(t)")
    axes[0, 0].set_xlabel("year")
    axes[0, 0].legend()

    years = annual_years(s)
    for member in annual(z_data[..., 0], axis=-1):
        axes[0, 1].plot(years, member, linewidth=0.5, alpha=0.6)
    axes[0, 1].plot(years, annual(z_forced[:, 0], axis=-1), color="k", linewidth=1.5, label="forced")
    axes[0, 1].set_title(r"slow mode $z_1$ (annual means)")
    axes[0, 1].legend()

    months = record[:360]
    axes[0, 2].plot(months, z_internal[0, :360, 1], label=r"$z_2$")
    axes[0, 2].plot(months, z_internal[0, :360, 2], label=r"$z_3$")
    axes[0, 2].set_title("pair internal, realization 0 (monthly, first 30 yr)")
    axes[0, 2].set_xlabel("year")
    axes[0, 2].legend()

    axes[0, 3].plot(lags, autocorrelation(z_internal[..., 1], MAX_LAG), color="C3", label="pair")
    axes[0, 3].plot(lags, theory_acf(s.eigvals[1], lags), color="C3", linestyle=":", label="pair theory")
    axes[0, 3].plot(lags, autocorrelation(z_internal[..., 0], MAX_LAG), color="C0", label="slow")
    axes[0, 3].plot(lags, theory_acf(s.lam1, lags), color="C0", linestyle=":", label="slow theory")
    for k in (12, 48):
        axes[0, 3].axvline(k, color="0.7", linewidth=0.5)
    zero_line(axes[0, 3])
    axes[0, 3].set_title("internal autocorrelation")
    axes[0, 3].set_xlabel("lag (months)")
    axes[0, 3].legend()

    for ax, lat in zip(axes[1], CALIBRATION_LATS):
        plot_realizations(ax, ds, lat)
        ax.set_title(f"x at {lat_label(lat)} (annual means)")
        ax.set_xlabel("year")
    axes[1, 0].legend()
    fig.suptitle(f"starting point: {len(ds.internal)} realizations, SNR {ds.snr:.3g}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, path, tight=False)


def plot_variance_per_mode(ax, ds):
    z = modal_coordinates(ds, ds.internal)
    modes = np.arange(z.shape[-1]) + 1
    ax.semilogy(modes, ds.system.modal_variances, color="0.5", linestyle="--", label="theory")
    ax.semilogy(modes, (z**2).mean(axis=(0, 1)), marker=".", linestyle="none", label="empirical")


def plot_sweep(datasets, path, title, extra=None, extra_title=""):
    n_cols = len(SWEEP_LATS) + 1 + (extra is not None)
    fig, axes = plt.subplots(len(datasets), n_cols, figsize=(4.5 * n_cols, 2.1 * len(datasets)), squeeze=False)
    for row, ds in zip(axes, datasets):
        for ax, lat in zip(row, SWEEP_LATS):
            plot_realizations(ax, ds, lat)
        label_row(row[0], ds)
        plot_variance_per_mode(row[len(SWEEP_LATS)], ds)
        if extra is not None:
            extra(row[-1], ds)
    for ax, lat in zip(axes[0], SWEEP_LATS):
        ax.set_title(f"x at {lat_label(lat)} (annual means)")
    axes[0, len(SWEEP_LATS)].set_title("internal variance per mode")
    if extra is not None:
        axes[0, -1].set_title(extra_title)
    axes[0, 0].legend()
    axes[0, len(SWEEP_LATS)].legend()
    axes[-1, 0].set_xlabel("year")
    axes[-1, len(SWEEP_LATS)].set_xlabel("mode")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, path, tight=False)


def acf_panel(ax, ds):
    z = modal_coordinates(ds, ds.internal)
    lags = np.arange(MAX_LAG + 1)
    s = ds.system
    ax.plot(lags, autocorrelation(z[..., 0], MAX_LAG), color="C0", label="slow")
    ax.plot(lags, theory_acf(s.lam1, lags), color="C0", linestyle=":")
    ax.plot(lags, autocorrelation(z[..., 1], MAX_LAG), color="C3", label="pair")
    ax.plot(lags, theory_acf(s.eigvals[1], lags), color="C3", linestyle=":")
    zero_line(ax)
    ax.text(0.98, 0.85, rf"$\tau_1$={decay_time_yr(s.lam1):.3g} yr, $\tau_p$={decay_time_yr(s.rho):.3g} yr",
            transform=ax.transAxes, ha="right", fontsize=8)


def mode_shape_panel(ax, ds):
    lat = np.rad2deg(PHI)
    W = ds.system.W
    for j, name in enumerate((r"$w_1$ (tilted)", r"$w_2$", r"$w_3$")):
        ax.plot(lat, W[:, j], marker=".", label=name)
    zero_line(ax)
    norms = [np.linalg.norm(np.linalg.matrix_power(ds.system.A, k), 2) for k in range(0, 241, 12)]
    ax.text(0.02, 0.05, f"cond(W)={np.linalg.cond(W):.2f}, max$_t\\|A^t\\|_2$={max(norms):.2f}",
            transform=ax.transAxes, fontsize=8)


def tau_lag1(ds):
    z = modal_coordinates(ds, ds.internal)[..., 0]
    lag1_corr = (z[:, 1:] * z[:, :-1]).sum() / (z[:, :-1] ** 2).sum()
    return decay_time_yr(lag1_corr)


def snr_spread(ax, x, datasets, **kwargs):
    emp = np.array([empirical_snr(ds) for ds in datasets])
    ax.errorbar(x, np.median(emp, axis=1),
                yerr=[np.median(emp, axis=1) - emp.min(axis=1), emp.max(axis=1) - np.median(emp, axis=1)],
                marker="o", capsize=2, **kwargs)


def plot_summary(total, partial, timescale, spatial, path):
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.5))

    snrs = [ds.param_value for ds in total]
    axes[0].loglog(snrs, snrs, color="0.6", linestyle="--", label="theoretical")
    snr_spread(axes[0], snrs, total, label="empirical (median, range)")
    axes[0].set_xlabel("theoretical SNR")
    axes[0].set_title("total SNR")
    axes[0].legend()

    for ci, (component, sweep) in enumerate(partial.items()):
        factors = [ds.param_value for ds in sweep]
        axes[1].loglog(factors, [ds.snr for ds in sweep], color=f"C{ci}", linestyle="--")
        snr_spread(axes[1], factors, sweep, color=f"C{ci}", label=component)
    axes[1].set_xlabel("modal variance factor")
    axes[1].set_ylabel("SNR (dashed: theoretical)")
    axes[1].set_title("partial SNR")
    axes[1].legend()

    taus = [ds.param_value for ds in timescale["snr"]]
    axes[2].loglog(taus, taus, color="0.6", linestyle="--", label="nominal")
    axes[2].loglog(taus, [tau_lag1(ds) for ds in timescale["snr"]], marker="o", label="lag-1 estimate")
    axes[2].set_xlabel(r"nominal $\tau_1$ (yr)")
    axes[2].set_title(r"slow timescale: internal $\tau_1$")
    axes[2].legend()

    for hold, marker in (("snr", "o"), ("modal_variance", "s")):
        sweep = timescale[hold]
        axes[3].loglog(taus, [ds.snr for ds in sweep], color="0.6", linestyle="--")
        snr_spread(axes[3], taus, sweep, label=f"hold {hold}")
    axes[3].loglog(taus, [ds.V_f / timescale["snr"][taus.index(20)].V_f for ds in timescale["snr"]],
                   color="k", linestyle=":", label=r"$V^{(f)}$ / reference")
    axes[3].set_xlabel(r"$\tau_1$ (yr)")
    axes[3].set_title("slow timescale: SNR (dashed: theoretical)")
    axes[3].legend()

    overlaps = [ds.param_value for ds in spatial]
    axes[4].plot(overlaps, overlaps, color="0.6", linestyle="--", label="nominal")
    axes[4].plot(overlaps, [pair_plane_overlap(ds.system.W) for ds in spatial], marker="o", label="from W")
    axes[4].plot(overlaps, [ds.V_f / spatial[0].V_f for ds in spatial], marker="s",
                 label=r"$V^{(f)}$ / overlap-0")
    axes[4].axvline(BASE_OVERLAP, color="0.8", linewidth=0.8, label="default patterns")
    axes[4].set_xlabel("nominal overlap")
    axes[4].set_title("spatial overlap")
    axes[4].legend()

    save(fig, path)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    kwargs = dict(n_realizations=N_REALIZATIONS)
    total = total_snr_sweep(**kwargs)
    partial = {c: partial_snr_sweep(c, **kwargs) for c in PARTIAL_SNR_COMPONENTS}
    timescale = {hold: slow_timescale_sweep(hold=hold, **kwargs) for hold in SLOW_TIMESCALE_HOLDS}
    spatial = spatial_overlap_sweep(**kwargs)
    forcing = forcing_overlap_sweep(**kwargs)

    plot_calibration(make_dataset(REFERENCE, equal_budget, **kwargs), FIGURES_DIR / "calibration.png")
    plot_sweep(total, FIGURES_DIR / "total_snr.png", "total SNR: all modal variances scaled together")
    for component, sweep in partial.items():
        plot_sweep(sweep, FIGURES_DIR / f"partial_snr_{component}.png",
                   f"partial SNR: {component} modal variance scaled")
    for hold, sweep in timescale.items():
        plot_sweep(sweep, FIGURES_DIR / f"slow_timescale_{hold}.png",
                   f"slow timescale, {hold} held fixed", extra=acf_panel,
                   extra_title="internal autocorrelation (dotted: theory)")
    plot_sweep(spatial, FIGURES_DIR / "spatial_overlap.png",
               "spatial overlap: slow mode tilts into the pair plane", extra=mode_shape_panel,
               extra_title="mode shapes vs latitude")
    plot_sweep(forcing, FIGURES_DIR / "forcing_overlap.png",
               r"forcing overlap: slow mode rotates toward $\hat b$", extra=mode_shape_panel,
               extra_title="mode shapes vs latitude")
    plot_summary(total, partial, timescale, spatial, FIGURES_DIR / "summary.png")


if __name__ == "__main__":
    main()
