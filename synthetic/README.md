# Synthetic experiments

A 20-mode linear system — one slow mode, one oscillating pair, 17 complement modes — driven by a forcing shaped
after the AR6 CO$_2$ ERF. Four estimators (`PullbackDMDc`, `LIM`, `LIM-opt`, `LR`) are fitted to 100 noise
realizations at every level of every ablation study, and scored on how well they recover the forced response and
the slow mode. [Data generation](#data-generation) at the end of this file is the full specification of the system.

Everything here runs from `synthetic/`. The folder is not a package: scripts are run from inside it.

## Quickstart

```bash
./run_synthetic_experiments.sh   # every ablation run and every figure (long; the fits dominate)
./run_synthetic_tests.sh         # unit tests, then the end-to-end system checks
./run_synthetic_tests.sh --list  # just list the tests
```

Both drivers activate the `dmdc_variants` conda env, `cd` into `synthetic/`, and export `OMP_NUM_THREADS=1`
(the per-step simulation loop is BLAS-thread bound — about 50x less CPU time on the cluster nodes) and
`MPLBACKEND=Agg`. Run them, rather than the scripts directly, unless you want one specific figure.

## Where the results are

Every output path, and the command that writes it:

| output | written by |
|---|---|
| `results/<run>/ablations.csv`, `config.json`, `provenance.json` | `run_ablation_studies.py --name <run>` |
| `figures/ablations/<run>/results_*.png` | `plot_ablation_run_results.py --runs <run>` |
| `figures/ablations/compare_<a>_vs_<b>/results_*.png` | `plot_ablation_run_results.py --runs <a> <b>` |
| `figures/ablations/<run>/` — the 5 system diagnostics | `run_ablation_studies.py` (its closing `plot_diagnostics` call) |
| `figures/diagnostics/data/` — 9 dataset diagnostics | `plot_ablation_diagnostics.py` |
| `figures/diagnostics/data/compare_forcings.png` | `plot_forcing_comparison.py` |
| `figures/diagnostics/system/<name>/` | `plot_system_diagnostics.py` |
| `figures/tests/` — 8 system checks | `tests/test_synthetic_pipeline.py` |
| `figures/reference/co2_zonal_forcing.png` | **nothing — this is an input**, the image `CO2_ZONAL_FORCING` in `system_patterns.py` was digitized from |

The headline figures are in `figures/ablations/`:

| figure | what it shows |
|---|---|
| `results_sweeps.png` | the four metrics (a)-(d) as rows, one ablation study per column, median and band over realizations, one line per method |
| `results_phase_snr_timescale_<run>.png` | the `joint_snr_timescale` study as a phase diagram: SNR against $\tau_1$ |
| `results_phase_change.png` | the same phase diagram, later run minus reference (written only when exactly two runs are plotted) |

The metric rows are fixed (`ROW_SPECS` in `plot_ablation_run_results.py`): (a) forced pattern correlation,
(b) forced relative RMSE, (c) slow-mode decay time $|\log(\hat\tau_1/\tau_1)|$, (d) mode shape, principal angle
in degrees. The columns are `DEFAULT_STUDIES`; `--studies` selects any other subset of `SWEEP_STUDIES`.

The 5 system diagnostics that accompany each run — `modal_overview`, `ensemble_super_spaghetti`, `forcing`,
`forced_response_tau1`, `forced_response_shape` — show the *data*, not the fits.

### Figure directory names

- `figures/ablations/`: one run goes to `<run>`, several to `compare_<a>_vs_<b>[_vs_...]`. With several runs
  the first is the reference, drawn as thin dotted medians under the others.
- `figures/diagnostics/system/<name>/`: `<name>` is `--name` if given, else `--from-run`, else `config.slug(cfg)`
  — `default` when the config is the default, otherwise a sanitized `config_diff` such as `n_realizations=10`.
  Using `--study/--level` without `--name` appends `__<study>=<level>`, e.g. `default__slow_timescale_snr=100`.
- `figures/diagnostics/data/compare_forcings.png` gains a `__<slug>` suffix under a non-default config, so the
  default figure is never overwritten.

### The runs that exist

| run | config | note |
|---|---|---|
| `baseline` | the default config | the reference for every comparison |
| `dip1` | `--set gauss_dip_amp=1` | a much deeper mid-century dip in the forcing |
| `b_unscaled` | pre-$B$-scaling calibration | **not regenerated**: current code cannot reproduce it, and its CSV uses the legacy schema (no `slow_tau_log_ratio`, `slow_unstable`, `slow_angle`), so rows (c) and (d) are skipped for it |

### `ablations.csv`

One row per (study, level, realization, method):

```
study, param_name, param_value, snr, tau1_yr, realization, method,
forced_corr, forced_rel_rmse, slow_eig_err, slow_tau_rel_err, slow_tau_log_ratio,
slow_unstable, slow_angle, pair_angle
```

`realization = -1` with `method = "oracle"` is the per-level oracle row: the true $A, B$ run over the same
forcing history, i.e. the floor set by truncating the history. `baseline` and `dip1` each hold 91 levels x
(1 oracle + 100 realizations x 4 methods) = 36 491 rows.

Each run also writes `provenance.json` — the structural values that are *not* in the config
(`record_months`, `state_dim`, `b_scaling`) — so runs made across a structural change cannot be compared by mistake.

## Scripts

| script | what it does |
|---|---|
| `run_ablation_studies.py` | the experiment runner: fits every method on every level of every study and scores them. `--name`, `--set KEY=VALUE ...`, `--from-run`, `--studies`, `--n-jobs` (default -1, parallel over levels) |
| `plot_ablation_run_results.py` | figures from existing `results/<run>/`. `--runs NAME [NAME ...]`, `--studies` |
| `plot_system_diagnostics.py` | diagnostics of the system and its data, for any config or sweep level. `--set`, `--from-run`, `--study`/`--level` (together), `--plots`, `--n-shown`, `--name` |
| `plot_ablation_diagnostics.py` | calibration and per-study dataset diagnostics. No options |
| `plot_forcing_comparison.py` | the true AR6 forcing against the two analytic models, all centered on the record; prints an RMSE table. `--set`, `--from-run` |
| `run_tests.py` | the unit tests, without pytest. `--only SUBSTRING`, `--list` |

## Modules

| module | contents |
|---|---|
| `config.py` | `Config`, the frozen dataclass holding every tunable value, plus `with_overrides`, `slug`, `load_config`, JSON round-trip |
| `system_patterns.py` | latitude grid, the three structured patterns, the zonal-mean CO$_2$ forcing pattern, and `make_W` |
| `ablation_data.py` | the core: `System`, the forcing models, simulation, the SNR budget, `SyntheticDataset`, and the `STUDIES` table of sweeps |
| `methods.py` | thin wrappers fitting the four estimators from `utils/` on one realization |
| `plot_style.py` | shared matplotlib style, `save`, `zero_line`, `record_span`, the forcing panels |

## Configuration

`config.py`'s `Config` is the single source of truth; `M = 20` and `N = 1980` months are structural and live in
`ablation_data.py`. Override anything on the command line:

```bash
python run_ablation_studies.py --name slow50 --set tau1_yr=50 slow_variance=0.5
python run_ablation_studies.py --name lag3 --from-run baseline --set lag=3 --studies total_snr
python plot_system_diagnostics.py --study slow_timescale_snr --level 100
```

`--from-run <name>` loads `results/<name>/config.json` as the starting point, so any `--set` is a delta on that run.

## Tests

`run_tests.py` imports every `tests/test_*.py` and calls its module-level `test_*` functions — plain functions with
asserts, no framework. It skips `tests/test_synthetic_pipeline.py`, which is not a test module but a script of
end-to-end checks that writes `figures/tests/`; `run_synthetic_tests.sh` runs it afterwards. The checks and their
tolerances are tabulated in the System checks table at the end of this file.

## Data generation

Follows the synthetic-example appendix of the paper, except for the forcing: the time series is an exponential plus
one Gaussian dip, shaped after the AR6 CO$_2$ ERF, and the pattern is the zonal-mean CO$_2$ forcing profile (both
replace the appendix's forcing). The code is `system_patterns.py` (patterns, modal matrix) and `ablation_data.py`
(dynamics, forcing, simulation, sweeps).

### Grid and patterns: `lat_grid`, `raw_patterns`
$$\phi_i=-\tfrac{\pi}{2}+\tfrac{i\pi}{M-1},\quad i=0,\dots,M-1,\qquad M=20\ (\Delta\phi\approx 9.47^\circ,\ \text{poles included})$$
$$a_1=\cos(\phi-\tfrac{\pi}{4}),\quad a_2=\operatorname{sinc}(2\phi)=\tfrac{\sin(2\pi\phi)}{2\pi\phi},\quad a_3=\frac{0.5-\cos(10\phi)}{0.5+|10\phi|},\quad b=\texttt{co2\_forcing\_pattern}(\phi)$$
$b$ is the zonal-mean CO$_2$ radiative forcing (W m$^{-2}$), interpolated linearly in $|\phi|$ from `CO2_ZONAL_FORCING`. The table is digitized every 5° from `figures/reference/co2_zonal_forcing.png`, with north and south averaged. It runs from 2.50 at the equator to 1.54 at the poles: positive everywhere and peaked at the equator.
$$w_k=a_k/\|a_k\|_2,\qquad \hat b=b/\|b\|_2\quad(\text{no area weighting})$$
The same convention holds downstream: `global_mean` is the plain mean over the $M$ grid points, not area weighted.
Gram (uncentered): $w_1^\top w_2=0.396,\ w_1^\top w_3=0.329,\ w_2^\top w_3=0.278,\ \hat b^\top w_{1,2,3}=0.650,\,0.489,\,0.478$. Modal coefficients $\gamma=W^{-1}\hat b$: $0.47$ (slow), $0.23,\,0.26$ (pair), complement norm $0.68$.

### Modal matrix: `make_W(M, seed=22)`
$$S=[w_1,w_2,w_3],\qquad Q_0=\text{last }M-3\text{ columns of the complete QR of }S,\qquad Q_\perp=Q_0O$$
$O$ is Haar orthogonal: the Q factor of a Gaussian matrix, with the signs of $\operatorname{diag}(R)$ absorbed.
$$W=[S,\ Q_\perp],\qquad W^{-1}=\begin{bmatrix}S^{+}\\ Q_\perp^\top\end{bmatrix}\quad(\text{exact since }Q_\perp^\top S=0)$$
The complement modes are not physical. Only their aggregate properties are meaningful.

### Propagator: `System.Lambda_R`, `System.A`
$$\Lambda_R=\operatorname{blkdiag}\!\left(\lambda_1,\ \begin{bmatrix}\rho\cos\theta&\rho\sin\theta\\-\rho\sin\theta&\rho\cos\theta\end{bmatrix},\ \operatorname{diag}(\lambda_4,\dots,\lambda_M)\right),\qquad A=W\Lambda_RW^{-1}$$
$$\sigma(A)=\{\lambda_1,\ \rho e^{\pm i\theta},\ \lambda_k\},\qquad \text{pair eigenvector}\propto w_2+iw_3$$
$$\tau(\lambda)=-1/\ln|\lambda|\ (\text{months}),\qquad \lambda=\texttt{eigenvalue}(\tau)=e^{-1/\tau},\qquad \text{period}=2\pi/\theta$$

### Model
$$x(t)=Ax(t-1)+B\,y(t)+\xi(t),\qquad \xi(t)\overset{iid}{\sim}\mathcal N(0,\,WDW^\top)$$
$$z=W^{-1}x:\qquad z(t)=\Lambda_Rz(t-1)+\gamma\,y(t)+\hat\xi(t),\qquad \gamma=W^{-1}B,\quad \hat\xi\sim\mathcal N(0,D)$$
$B$ is the scaled forcing matrix (see Forcing matrix); $\hat b$ is its unit-norm pattern.
Simulation runs in modal coordinates (`run_modal`), and the result is mapped back by $x=Wz$.

### Noise from modal variances: `System.modal_variances`, `System.noise_variances`
$$D=\operatorname{diag}(\sigma_1^2,\sigma_p^2,\sigma_p^2,\sigma_4^2,\dots,\sigma_M^2)$$
$$s_1^2=\tfrac{\sigma_1^2}{1-\lambda_1^2},\quad s_p^2=\tfrac{\sigma_p^2}{1-\rho^2},\quad s_k^2=\tfrac{\sigma_k^2}{1-\lambda_k^2}\quad\Longleftrightarrow\quad \sigma_1^2=s_1^2(1-\lambda_1^2),\ \ \sigma_p^2=s_p^2(1-\rho^2),\ \ \sigma_k^2=s_c^2(1-\lambda_k^2)$$
The inputs are the three target modal variances $(s_1^2,s_p^2,s_c^2)$, stored as `s1_sq`, `sp_sq`, `sc_sq`. Changing an eigenvalue recomputes $\sigma^2$, so the modal variance stays fixed.

### Forcing: `centered_forcing`, `forcing_series`
**Centering, always.** The forcing is centered on the observed interval, i.e. every month after the spin-up (the record, 1850–2014):
$$y(t)=a\Big(F(Y_0+t/12)-\tfrac1N\textstyle\sum_{s=0}^{N-1}F(Y_0+s/12)\Big),\qquad t=-T_f,\dots,N-1$$
- **One definition.** `centered_forcing` is the only place where the centering is done.
- **Same offset before the record.** The record mean is removed at all times, so the spin-up and the forcing history keep the same offset. This mirrors the real-world experiments, where `load_forcings_pullback` subtracts the record mean from both the record and the history.
- **Used everywhere.** This one series generates the data, is the forcing passed to the methods (`forcings()` returns slices of it, unchanged), and is what every forcing plot draws (`forcing.png`, `compare_forcings.png`, `forced_response_shape.png`).
- **Enforced.** `make_dataset` asserts a zero record mean, and `test_forcing_centered_everywhere` checks every sweep level and the plotted curves.

At the default, the past constant is $y=-0.666$ against a record range of $1.834$, and the history mean is $-0.586$. The forced response therefore starts in equilibrium with a negative forcing. For long $\tau_1$ it is still relaxing during the record and carries a large offset (`forced_response_tau1.png`).

Record month $t=0,\dots,N-1$ is the fractional year $t_{\rm yr}=Y_0+t/12$, with $Y_0=\texttt{record\_end\_year}-164$. The default `record_end_year = 2014` gives a record from Jan 1850 to Dec 2014, matching the real-data experiments (`utils/data_utils.py`). Every plotted time axis is in these calendar years (`record_years`, `annual_years`, `spinup_years`).

$a=$ `forcing_amplitude` is $1$: the overall scale is set by $B$ (below), not by the forcing. The forcing depends on absolute dates, so a longer spin-up or history only extends it backwards; the record values do not change.

**Default: exp + one dip** (`forcing_source = "analytic_gauss"`, `co2_forcing_gauss_model`):
$$F(t)=c+a\,e^{(t-2014)/\tau_G}+A_+e^{-(t-\mu_+)^2/2\sigma_+^2}-A_-e^{-(t-\mu_-)^2/2\sigma_-^2}$$
- **Default values.** $\tau_G=65$ yr, a dip $A_-=0.25$ at $\mu_-=1965$ ($\sigma_-=15$ yr), and no bump ($A_+=0$; its shape defaults to 1920 and $\sigma_+=20$ yr). These are round numbers, not a fit.
- **Provenance.** The joint least-squares fit of the exp and two Gaussians to the annual AR6 CO$_2$ values (`CO2_GAUSS_FIT`) gives:
  - $c=-0.0054$, $a=1.96$ and e-folding 63.2 yr;
  - a bump of 0.048 at 1921 ($\sigma$ 17.9 yr) and a dip of 0.137 at 1966 ($\sigma$ 14.1 yr);
  - an RMSE of 0.0092 W m$^{-2}$, against 0.038 for the exponential.

  The default deepens the dip and drops the bump. Against the true forcing (both centered), the default's record RMSE is $0.037$ W m$^{-2}$, against $0.046$ for the exponential, and its record-shape RMSE is $0.075$, against $0.093$.
- **Shape parameters.**
  - The parameters are `gauss_efold_yr`, `gauss_bump_amp`, `gauss_bump_year`, `gauss_bump_width_yr`, `gauss_dip_amp`, `gauss_dip_year` and `gauss_dip_width_yr`.
  - Amplitudes are in W m$^{-2}$ next to the fixed $a=1.96$; only their ratio to the exponential matters, and 0 removes that Gaussian.
  - Amplitudes must be $\ge0$, and widths and the e-folding $>0$.
  - $c$ and $a$ stay at the fit, since they drop out.
  - For example, `--set gauss_dip_amp=1` gives the amplified-dip run `dip1`.
- **Constant past.** The Gaussians vanish before about 1850, so the past is constant: $|F-c|<10^{-3}$ of the rise before 1500.

**Exp** (`forcing_source = "analytic"`, `co2_forcing_model`): $F(t)=c+a\,e^{(t-2014)/\tau_F}$, with $\tau_F=$ `forcing_efold_yr` $=60$ yr.
- **Provenance.** The fit is least squares to the annual values 1750–2019 at mid-year (`CO2_FIT`): $c=0.0191$, $a=1.915$ and $\tau_F=59.8$ yr. Its RMSE is $0.038$ W m$^{-2}$, and $0.046$ over the record. The largest misfit is the mid-century bump.
- **No polynomial.** A literal polynomial-plus-exponential diverges in the past, so the plain exponential is used.
- **Constant past.** $|F-c|/(F(2014)-c)<10^{-3}$ for every year before 1590. Over the longest spin-up (4000 yr, at $\tau_1=100$ yr), the forced slow mode varies by less than $10^{-3}$ of its record range between spin-up years 1000 and 3000. The `forcing` diagnostic shows this.

**File** (`forcing_source = "file"`): column `forcing_column` (default `co2`) of `forcing_file` (default `data_preparation/AR6_ERF_1750-2019.csv`; a relative path is relative to the repo root). `load_forcing_file` accepts two formats:
- an annual CSV with a `year` column, interpolated to months with `interpolate` as for the real data;
- a monthly CSV with a `time` column (`YYYY-MM-01`), used as is.

Outside the data, $F$ is held at its first value (for AR6, $\approx0$: a constant past) and continued after the data at its linear trend over the last 10 years. `forcing_efold_yr` only affects `analytic`, and `forcing_file`/`forcing_column` only affect `file`.

**Comparison and checks:**
- `plot_forcing_comparison.py` plots the true forcing and both models, all centered on the record (`figures/diagnostics/data/compare_forcings.png`).
- `plot_system_diagnostics.py` diagnostics:
  - `forced_response_tau1`: only the forced responses of the `slow_timescale_snr` datasets at $\tau_1=1,20,100$ yr, in tall panels;
  - `forced_response_shape`: the global-mean forced response and the forcing that drove it, both standardized, so only their shapes are compared (a slow mode lags the forcing and rounds its turns). The panel reports their correlation: $0.995$ at the starting point.

### Forcing matrix: `System.y0`, `System.b_scale`, `System.B`
$$y_0=y(-T_s),\qquad B=\frac{\hat b}{\|y_0\,(I-A)^{-1}\hat b\|_F},\qquad\text{so}\quad \|y_0\,(I-A)^{-1}B\|_F=1$$
The system starts the spin-up in quasi-equilibrium with the constant past forcing $y_0$, and $B$ is scaled so that this initial state has unit norm. That fixes the scale of everything: the forced response, and through `config_budget` the noise budget too.

- **$\hat b$ stays raw.** `System.b` is the unit-norm pattern (`B_HAT`, the Gram values above, the pattern panels). `System.B` is derived, never stored, so every `replace` that changes an eigenvalue, $W$ or a forcing field recomputes it. The dynamics, the oracle and the comparison against a fitted $B$ all use `System.B`.
- **Per system.** $A$ enters through $(I-A)^{-1}$, so the scale is recomputed at every sweep level. $\|(I-A)^{-1}\hat b\|$ grows roughly linearly in $\tau_1$ while $V^{(f)}$ grows more slowly (a slow mode cannot equilibrate within 165 yr), so $V^{(f)}$ now *falls* with $\tau_1$: $0.574$ at $\tau_1=1$ yr, $0.265$ at $20$ yr, $0.068$ at $100$ yr.
- **$y_0$ uses the noise spin-up $T_s$, not $T_f$.** $T_f$ is $T_s$ padded out to cover the methods' history, so keying $B$ to $T_f$ would let a method setting move the generated data. The forcing is near-constant that far back, so the two agree in practice.
- **Scale-invariant results.** Every budget except `_fixed_budget` is proportional to that level's own $V^{(f)}$, so rescaling $B$ scales the forced response, the noise and the data by one common factor, with the same random draws. All four methods are linear and every score is a ratio, so `total_snr`, `partial_snr_*`, `slow_timescale_snr`, `spatial_overlap` and `joint_snr_timescale` are **bit-identical** to the old $V^{(f)}=M$ calibration. `slow_timescale_modal_variance` is the exception: it holds the budget fixed while $V^{(f)}$ follows $\tau_1$, so its SNR inverts (see Sweeps).

### Spin-up and ground truth: `System.noise_spinup`, `System.spinup`, `forced_response`, `internal_variability`
$$T_s=\max\big(\lceil 40\,\max_k\tau(\lambda_k)\rceil,\ 1200\big),\qquad T_f=\max(T_s,\ \texttt{history})$$
The noise spin-up $T_s$ uses the longest decay time of any mode, so it still covers the pair when $\tau_1<\tau_p$. It does not depend on the history, so changing the methods' history leaves the realizations unchanged. The forcing series and the forced response start $T_f$ months before the record, which is at least as long as the forcing history passed to the method.
$$x^{(f)}(t)=Ax^{(f)}(t-1)+B\,y(t),\quad x^{(f)}(-T_f)=0\qquad(\text{shared by all realizations})$$
$$x^{(i)}(t)=Ax^{(i)}(t-1)+\xi(t),\quad x^{(i)}(-T_s)=0,\qquad x=x^{(f)}+x^{(i)}\ (\text{linear, so}\ =x-x^{(f)})$$
Times $t<0$ are discarded. Arrays are time-major: `forced` $(N,M)$, `internal` $(R,N,M)$, `data = forced + internal`.

**Common random numbers:** `SeedSequence(seed).spawn(2)` gives separate spin-up and record streams. Every level of every sweep therefore uses the same record noise $\varepsilon(t)$, $t\ge0$, with $\hat\xi=\sqrt D\,\varepsilon$.

### SNR: `theoretical_snr`, `empirical_snr`
$$V^{(f)}=\textstyle\sum_{i=1}^M\operatorname{Var}_t\,x^{(f)}_i(t),\qquad \mathrm{SNR}=\frac{V^{(f)}}{s_1^2+2s_p^2+(M-3)s_c^2}$$
The empirical SNR is $V^{(f)}/\sum_i\operatorname{Var}_t x^{(i)}_i$ per realization. It scatters around the theoretical value, especially for long $\tau_1$.

### Starting point: `REFERENCE`
| | value |
|---|---|
| step, record | monthly, $N=1980$ (165 yr), $M=20$ |
| patterns | `make_W(20, seed=22)` |
| forcing | $c+a\,e^{(t-2014)/65\,\text{yr}}-0.25\,e^{-(t-1965)^2/2\cdot15^2}$, centered on the record Jan 1850 – Dec 2014 (`record_end_year = 2014`); pattern $\hat b$ = zonal-mean CO$_2$ forcing, scaled to $B$ |
| slow | $\tau_1=20$ yr, $\lambda_1=e^{-1/240}=0.9958$ |
| pair | $\tau_p=2$ yr, $\rho=e^{-1/24}=0.959$; period 4 yr, $\theta=2\pi/48$ (ACF first zero at 12 months) |
| complement | $\lambda_k\overset{iid}{\sim}\mathcal U(0,0.1)$, 17 values, seed 20 |
| budget (`equal_budget`) | $s_1^2=V^{(f)},\ s_p^2=V^{(f)}/2,\ s_c^2=V^{(f)}/17\ \Rightarrow\ \mathrm{SNR}=1/3$; $V^{(f)}=0.2655$ |
| spin-up | $T_s=9601$ months |
| realizations | `N_REALIZATIONS = 100` per configuration, seed 0 |

### Sweeps
The patterns and eigenvalue seeds are fixed in every sweep. Only the noise realizations vary.

| study | swept | how |
|---|---|---|
| `total_snr_sweep` | $\mathrm{SNR}\in\{1/30,1/10,1/3,1,3,10,30\}$ | $(s_1^2,s_p^2,s_c^2)\times\frac{1/3}{\mathrm{SNR}}$; eigenvalues, patterns and forcing fixed. 10 and 30 extend the .tex grid toward a nearly noise-free end |
| `partial_snr_sweep(component)` | factor $\in\{1/4,1/2,1,2,4\}$ on $s_1^2$, $s_p^2$ or $s_c^2$ | the other two stay at the equal budget |
| `slow_timescale_sweep(hold="snr")` | $\tau_1\in\{1,2,5,10,20,50,100\}$ yr | `equal_budget` of each level's $V^{(f)}$, so SNR $=1/3$ |
| `slow_timescale_sweep(hold="modal_variance")` | same | budget fixed at the starting point, so SNR follows $V^{(f)}$, which under the $B$ scaling *falls* with $\tau_1$: SNR $0.721,\,0.692,\,0.566,\,0.459,\,0.333,\,0.174,\,0.085$ |
| `spatial_overlap_sweep` | $c\in\{0,0.25,0.457,0.75,0.9,0.95\}$ | $w_1\to\sqrt{1-c^2}\,u_\perp+c\,u_\parallel$, equal budget |

The slow-timescale sweep replaces the old temporal-overlap ablation: $\tau_1=1$–$2$ yr overlaps $\tau_p=2$ yr.

Spatial overlap (`tilt_slow`, `pair_plane_overlap`): $u_\parallel$ and $u_\perp$ are the unit parts of $w_1$ inside and outside $\operatorname{span}(w_2,w_3)$, and $c=\cos\angle(w_1,\operatorname{span}(w_2,w_3))$.
- $c=0.457$ is the default $w_1$.
- $w_1$ stays in $\operatorname{span}(S)$, so $Q_\perp$ is unchanged. $W^{-1}=[\operatorname{pinv}(S);Q_\perp^\top]$ is recomputed.

### Fitting interface: `SyntheticDataset.forcings(history=1200)`
$$\texttt{long\_forcings}=y(-\texttt{history}),\dots,y(N-1)\ \in\mathbb R^{(\texttt{history}+N)\times1},\qquad \texttt{short\_forcings}=y(0..N-1),\qquad \texttt{transition\_time}=\texttt{history}$$
The forcing enters at the target time, which matches `PullbackDMDc` ($x_t=Ax_{t-\text{lag}}+Bf_t$), so no shift is needed. The default 100-yr history is now shorter than the 165-yr record. Its truncation error $\sim e^{-100\,\text{yr}/\tau_1}$ becomes noticeable for $\tau_1$ of a few decades.

### Caveats
- **Sample variances.** For AR(1), the relative std of the sample variance over $N$ steps is $\approx\sqrt{2(1+\lambda^2)/((1-\lambda^2)N)}$, which is $0.63$ at $\tau_1=20$ yr. Mean removal also biases $\operatorname{Var}_t$ of the slow mode low.
- **Pair identifiability.** The pair covariance is isotropic, so evaluate the pair through its plane (principal angles) and its eigenvalue, not pattern by pattern.
- **Complement modes.** Individual modes and their forcing coefficients depend on the seed. Interpret only aggregates.
- **Lag-$\tau$ fits.** The propagator is $A^\tau$. The fitted forcing matrix approximates $(\sum_{m<\tau}A^m)B$, so compare forced responses, not $B$.
- **Performance.** The per-step loop is BLAS-thread bound. Run with `OMP_NUM_THREADS=1`, which is about 50× less CPU time on the cluster nodes.

### System checks: `tests/test_synthetic_pipeline.py` → `figures/tests/`
| check | figure(s) | asserts |
|---|---|---|
| `plot_system_check` | `system_check.png` | eigenvectors of $A$: slow $\parallel w_1$, pair $=c(w_2+iw_3)$ (1e-10) |
| `plot_ensemble_spaghetti` | `ensemble_spaghetti.png` | none (10 realizations + forced, annual means) |
| `check_dmdc_recovery` | `dmdc_recovery_check.png` | noise-free, white $y$: PullbackDMDc recovers $\lambda_1$, $\rho e^{i\theta}$, $\hat b$, $w_1$ and the plane $\operatorname{span}(w_2,w_3)$ (1e-8) |
| `check_forced_internal_split` | `forced_internal_spaghetti.png`, `forced_internal_convergence.png`, `modal_overview.png` | modal simulator equals a direct $x$-space simulation of the model (1e-9 relative); ensemble mean $\to$ forced at slope $<-0.4$. `modal_overview.png` (no asserts) has three panels: global means (raw, internal, forced), the spatial patterns, and the forcing |
| `check_forced_internal_recovery` | `forced_internal_recovery.png`, `forced_internal_recovery_mse.png` | starting point only (SNR 1/3, 100 realizations): mirrored errors asserted; skill (correlation, errors, slow eigenvalue) reported, not asserted |

The recovery check also prints an oracle: the true $A,B$ run over the same 100-yr history. That is the floor set by truncating the history ($\approx0.3\%$ at the starting point).
