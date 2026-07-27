import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from utils.params import artifact_root, esms, method_markers, predictions_dir


CLIM_VAR_LABELS = {
	"tas": "SAT",
	"psl": "SLP",
	"tas_ocean": "OSAT",
}

MODEL_DISPLAY_NAMES = {
	"MPI-ESM1-2-LR": "MPI-ESM",
}

METHOD_NAMES_ON_FIG = {
	"PullbackDMDc": "PullbackDMDc",
	"PullbackDMDc-2d": "PullbackDMDc(2d)",
	"PullbackDMDc-3d": "PullbackDMDc(3d)",
}

# Keep ESM colors consistent with other evaluation scripts.
ESM_COLORS = {
	"20CRv3": "black",
	"CESM2": "#7f7f7f",
	"MPI-ESM1-2-LR": "#8c564b",
	"CanESM5": "#2ca02c",
	"MIROC6": "#0b1f5b",
}

# Distinct linestyles make models separable in grayscale prints.
ESM_LINESTYLES = {
	"20CRv3": "-",
	"CESM2": "--",
	"MPI-ESM1-2-LR": "-.",
	"CanESM5": ":",
	"MIROC6": (0, (3, 1, 1, 1)),
}


def display_model_name(model_name):
	return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def display_method_name(method_name):
	return METHOD_NAMES_ON_FIG.get(method_name, method_name)


def compute_yearly_decay_time(lambdas, tau):
	return -tau / (12.0 * np.log(np.abs(lambdas)))


def build_suffixes(lag, start_year, end_year):
	if lag == 3:
		lag_suffix = ""
	elif lag == 12:
		lag_suffix = "_lag12"
	else:
		raise ValueError(f"Unsupported lag: {lag}")

	if (start_year, end_year) == (1850, 2014):
		year_suffix = ""
	elif (start_year, end_year) == (1950, 2014):
		year_suffix = "_tier1"
	else:
		raise ValueError(f"Unsupported year range: {(start_year, end_year)}")

	return lag_suffix, year_suffix


def method_for_variable(clim_var):
	if clim_var == "psl":
		return "PullbackDMDc-2d"
	return "PullbackDMDc-3d"


def load_decay_member_vectors(clim_var, model, method, base_pred_dir):
	"""Return list of decay-time vectors (one vector per ensemble member/file)."""
	model_dir = os.path.join(base_pred_dir, clim_var, model)
	if not os.path.isdir(model_dir):
		return []

	member_vectors = []
	for f_name in sorted(os.listdir(model_dir)):
		if not f_name.endswith(".pkl"):
			continue

		this_method = f_name.split("_")[-1].split(".")[0]
		if this_method != method:
			continue

		f_path = os.path.join(model_dir, f_name)
		with open(f_path, "rb") as f_handle:
			dmd = pickle.load(f_handle)

		if hasattr(dmd, "eigvals"):
			eigvals = np.asarray(dmd.eigvals)
		elif hasattr(dmd, "A"):
			eigvals = np.linalg.eigvals(dmd.A)
		else:
			continue

		if eigvals.size == 0 or not hasattr(dmd, "lag"):
			continue

		# Ensure largest-to-smallest mode ordering by |lambda|.
		order = np.argsort(np.abs(eigvals))[::-1]
		eigvals = eigvals[order]

		decay = np.real(compute_yearly_decay_time(eigvals, dmd.lag))
		# Keep only finite, positive decay times.
		decay = np.where(np.isfinite(decay) & (decay > 0.0), decay, np.nan)
		if np.all(np.isnan(decay)):
			continue

		member_vectors.append(decay)

	return member_vectors


def compute_common_min_modes(data_by_var):
	lengths = []
	for clim_var_data in data_by_var.values():
		for model_data in clim_var_data.values():
			for vec in model_data:
				lengths.append(len(vec))

	if not lengths:
		return 0
	return int(np.min(lengths))


def summarize_by_model(member_vectors, n_modes):
	member_vectors = np.asarray(member_vectors, dtype=float)
	if member_vectors.size == 0:
		return None

	if member_vectors.ndim == 1:
		member_vectors = member_vectors.reshape(1, -1)

	trimmed = [vec[:n_modes] for vec in member_vectors if len(vec) >= n_modes]
	if not trimmed:
		return None

	arr = np.vstack(trimmed)
	mean = np.nanmean(arr, axis=0)
	lower = np.nanpercentile(arr, 2.5, axis=0)
	upper = np.nanpercentile(arr, 97.5, axis=0)
	return mean, lower, upper



def acf_bands(acf_values, sample_size, alpha=0.05):
	# White-noise null: CI at lag k is +/- z_score / sqrt(N-k).
	z_score = 1.96
	acf_values = np.asarray(acf_values, dtype=float).reshape(-1)
	se = np.zeros_like(acf_values, dtype=float)
	max_lag = min(acf_values.size, sample_size)
	for lag_idx in range(max_lag):
		se[lag_idx] = 1.0 / np.sqrt(sample_size - lag_idx)
	band = z_score * se
	return -band, band


def compute_member_mode_ci_coverage(member_acfs, top_n, sample_size, years_window):
	member_coverage = []
	max_lag_count = int(years_window * 12) + 1  # include lag-0
	for member_acf in member_acfs:
		mode_coverage = []
		for mode_idx in range(top_n):
			acf_values = np.asarray(member_acf[mode_idx], dtype=float).reshape(-1)
			acf_values = acf_values[:max_lag_count]
			if acf_values.size <= 1:
				mode_coverage.append(np.nan)
				continue

			acf_lower, acf_upper = acf_bands(acf_values, sample_size=sample_size, alpha=0.05)
			# Exclude lag-0 (always 1 by construction) to focus on internal variability noise structure.
			inside = (acf_values[1:] >= acf_lower[1:]) & (acf_values[1:] <= acf_upper[1:])
			mode_coverage.append(100.0 * np.mean(inside))

		member_coverage.append(mode_coverage)
	return np.asarray(member_coverage, dtype=float)


def _resolve_acf_path(results_dirs, filename):
	for directory in results_dirs:
		candidate = os.path.join(directory, filename)
		if os.path.exists(candidate):
			return candidate
	raise FileNotFoundError(
		"Missing required ACF file. Checked: "
		+ ", ".join(os.path.join(d, filename) for d in results_dirs)
	)


def load_acf_ci_coverage_by_var(clim_vars, results_dirs, sample_size, years_window):
	acf_coverage_by_var = {cv: {} for cv in clim_vars}
	mode_lengths = []

	for clim_var in clim_vars:
		method = method_for_variable(clim_var)
		acf_filename = f"acfs_{clim_var}_{method}.pkl"
		acf_path = _resolve_acf_path(results_dirs, acf_filename)

		with open(acf_path, "rb") as f_handle:
			acf_data = pickle.load(f_handle)

		for model in esms:
			model_member_acfs = acf_data.get(model, [])
			if len(model_member_acfs) == 0:
				acf_coverage_by_var[clim_var][model] = np.empty((0, 0), dtype=float)
				continue

			member_mode_count = min(member_acf.shape[0] for member_acf in model_member_acfs)
			mode_lengths.append(member_mode_count)
			acf_coverage_by_var[clim_var][model] = compute_member_mode_ci_coverage(
				member_acfs=model_member_acfs,
				top_n=member_mode_count,
				sample_size=sample_size,
				years_window=years_window,
			)

	if not mode_lengths:
		raise ValueError("No valid ACF member data found in required files.")

	return acf_coverage_by_var, int(np.min(mode_lengths))


def plot_decay_with_acf_row(data_by_var, acf_coverage_by_var, clim_vars, n_modes):
	fig, axes = plt.subplots(
		2,
		len(clim_vars),
		figsize=(5.2 * len(clim_vars), 8.2),
		sharex="col",
		constrained_layout=True,
	)

	if len(clim_vars) == 1:
		axes = axes.reshape(2, 1)

	mode_axis = np.arange(1, n_modes + 1)

	for col_idx, clim_var in enumerate(clim_vars):
		top_ax = axes[0, col_idx]
		bottom_ax = axes[1, col_idx]
		method = method_for_variable(clim_var)
		method_marker = method_markers.get(method, "o")

		for model in esms:
			member_vectors = data_by_var[clim_var].get(model, [])
			summary = summarize_by_model(member_vectors, n_modes=n_modes)
			if summary is None:
				continue

			mean, lower, upper = summary
			label = display_model_name(model)
			color = ESM_COLORS.get(model, "tab:gray")
			linestyle = ESM_LINESTYLES.get(model, "-")

			top_ax.plot(
				mode_axis,
				mean,
				color=color,
				linestyle=linestyle,
				linewidth=1.8,
				label=label,
				zorder=3,
			)
			top_ax.fill_between(
				mode_axis,
				lower,
				upper,
				color=color,
				alpha=0.16,
				linewidth=0,
				zorder=2,
			)

		for model in esms:
			acf_member_data = acf_coverage_by_var[clim_var].get(model, np.empty((0, 0), dtype=float))
			summary = summarize_by_model(acf_member_data, n_modes=n_modes)
			if summary is None:
				continue

			mean, lower, upper = summary
			color = ESM_COLORS.get(model, "tab:gray")
			linestyle = ESM_LINESTYLES.get(model, "-")

			bottom_ax.plot(
				mode_axis,
				mean,
				color=color,
				linestyle=linestyle,
				marker=method_marker,
				markersize=4,
				linewidth=1.8,
				label=display_model_name(model),
				zorder=3,
			)
			bottom_ax.fill_between(
				mode_axis,
				lower,
				upper,
				color=color,
				alpha=0.16,
				linewidth=0,
				zorder=2,
			)

		top_ax.axvline(4.5, color="red", linestyle="--", linewidth=1.6, alpha=0.9, zorder=4)
		bottom_ax.axvline(4.5, color="red", linestyle="--", linewidth=1.6, alpha=0.9, zorder=4)
		top_ax.set_title(
			f"{CLIM_VAR_LABELS.get(clim_var, clim_var)} | {display_method_name(method)} ",
			fontsize=12,
		)
		# top_ax.set_yscale("log")
		top_ax.grid(True, which="major", alpha=0.3)
		top_ax.grid(True, which="minor", alpha=0.14)

		bottom_ax.set_ylim(0.0, 100.0)
		bottom_ax.set_yticks([0, 20, 40, 60, 80, 100])
		bottom_ax.grid(True, which="major", alpha=0.3)
		bottom_ax.grid(True, which="minor", alpha=0.14)
		bottom_ax.set_xlabel("Mode (sorted by |lambda|)", fontsize=10)

	axes[0, 0].set_ylabel("Decay Time (years)", fontsize=11)
	axes[1, 0].set_ylabel("ACF Points Within 95% CI (%)\n(first 40 years)", fontsize=11)

	handles, labels = axes[0, 0].get_legend_handles_labels()
	if handles:
		seen = set()
		unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
		fig.legend(
			[h for h, _ in unique],
			[l for _, l in unique],
			loc="lower center",
			ncol=len(unique),
			frameon=True,
			edgecolor="#9e9e9e",
			framealpha=1.0,
			bbox_to_anchor=(0.5, -0.03),
			fontsize=9,
		)

	return fig


def main():
	lag = 3
	start_year, end_year = 1850, 2014
	acf_stat_years = 40

	lag_suffix, year_suffix = build_suffixes(lag, start_year, end_year)
	clim_vars = ["tas", "psl", "tas_ocean"]
	sample_size = (end_year - start_year + 1) * 12 - lag
	if sample_size <= 1:
		raise ValueError(f"Invalid sample size for ACF confidence bands: {sample_size}")

	base_pred_dir = predictions_dir(year_suffix, lag_suffix)
	results_dir = os.path.join(".", f"evaluation_results/results{year_suffix}{lag_suffix}")
	artifact_results_dir = os.path.join(artifact_root, "evaluation_results", f"results{year_suffix}{lag_suffix}")
	os.makedirs(results_dir, exist_ok=True)

	data_by_var = {cv: {} for cv in clim_vars}
	for clim_var in clim_vars:
		method = method_for_variable(clim_var)
		for model in esms:
			data_by_var[clim_var][model] = load_decay_member_vectors(
				clim_var=clim_var,
				model=model,
				method=method,
				base_pred_dir=base_pred_dir,
			)

	common_n_modes = compute_common_min_modes(data_by_var)
	if common_n_modes <= 0:
		raise ValueError("No valid eigenvalue decay data found for the requested settings.")
	acf_coverage_by_var, acf_min_modes = load_acf_ci_coverage_by_var(
		clim_vars=clim_vars,
		results_dirs=[results_dir, artifact_results_dir],
		sample_size=sample_size,
		years_window=acf_stat_years,
	)
	if acf_min_modes < common_n_modes:
		raise ValueError(
			f"ACF data has fewer modes ({acf_min_modes}) than decay data ({common_n_modes})."
		)

	fig = plot_decay_with_acf_row(
		data_by_var=data_by_var,
		acf_coverage_by_var=acf_coverage_by_var,
		clim_vars=clim_vars,
		n_modes=common_n_modes,
	)

	out_fig = os.path.join(results_dir, f"decay_lines_all_modes_ci95_commonN{common_n_modes}.pdf")
	fig.savefig(out_fig, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved: {out_fig}", flush=True)


if __name__ == "__main__":
	main()
