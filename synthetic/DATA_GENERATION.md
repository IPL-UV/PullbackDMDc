# Synthetic data generation

Follows the synthetic-example appendix of the paper, except for the forcing: the time series is an exponential fit to the AR6 CO$_2$ ERF and the pattern is the zonal-mean CO$_2$ forcing profile (both replace the appendix's forcing). Code: `synthetic_system.py` (patterns, modal matrix) and `ablations.py` (dynamics, forcing, simulation, sweeps). Tests: `test_ablations.py`, `test_run_ablations.py`. Diagnostics: `plot_ablations.py` → `figures/diagnostics/data/`, `plot_system.py` → `figures/diagnostics/system/<name>/`. Ablation runs: `run_ablations.py`, `plot_ablation_results.py` → `figures/ablations/<run>/`. Shared figure style: `plot_style.py`.

## Grid and patterns: `lat_grid`, `raw_patterns`
$$\phi_i=-\tfrac{\pi}{2}+\tfrac{i\pi}{M-1},\quad i=0,\dots,M-1,\qquad M=20\ (\Delta\phi\approx 9.47^\circ,\ \text{poles included})$$
$$a_1=\cos(\phi-\tfrac{\pi}{4}),\quad a_2=\operatorname{sinc}(2\phi)=\tfrac{\sin(2\pi\phi)}{2\pi\phi},\quad a_3=\frac{0.5-\cos(10\phi)}{0.5+|10\phi|},\quad b=\texttt{co2\_forcing\_pattern}(\phi)$$
$b$ is the zonal-mean CO$_2$ radiative forcing (W m$^{-2}$), interpolated linearly in $|\phi|$ from `CO2_ZONAL_FORCING`. The table is digitized every 5° from `figures/reference/co2_zonal_forcing.png`, with north and south averaged. It runs from 2.50 at the equator to 1.54 at the poles: positive everywhere and peaked at the equator.
$$w_k=a_k/\|a_k\|_2,\qquad \hat b=b/\|b\|_2\quad(\text{no area weighting})$$
Gram (uncentered): $w_1^\top w_2=0.396,\ w_1^\top w_3=0.329,\ w_2^\top w_3=0.278,\ \hat b^\top w_{1,2,3}=0.650,\,0.489,\,0.478$. Modal coefficients $\gamma=W^{-1}\hat b$: $0.47$ (slow), $0.23,\,0.26$ (pair), complement norm $0.68$.

## Modal matrix: `make_W(M, seed=22)`
$$S=[w_1,w_2,w_3],\qquad Q_0=\text{last }M-3\text{ columns of the complete QR of }S,\qquad Q_\perp=Q_0O$$
$O$ is Haar orthogonal: the Q factor of a Gaussian matrix, with the signs of $\operatorname{diag}(R)$ absorbed.
$$W=[S,\ Q_\perp],\qquad W^{-1}=\begin{bmatrix}S^{+}\\ Q_\perp^\top\end{bmatrix}\quad(\text{exact since }Q_\perp^\top S=0)$$
The complement modes are not physical. Only their aggregate properties are meaningful.

## Propagator: `System.Lambda_R`, `System.A`
$$\Lambda_R=\operatorname{blkdiag}\!\left(\lambda_1,\ \begin{bmatrix}\rho\cos\theta&\rho\sin\theta\\-\rho\sin\theta&\rho\cos\theta\end{bmatrix},\ \operatorname{diag}(\lambda_4,\dots,\lambda_M)\right),\qquad A=W\Lambda_RW^{-1}$$
$$\sigma(A)=\{\lambda_1,\ \rho e^{\pm i\theta},\ \lambda_k\},\qquad \text{pair eigenvector}\propto w_2+iw_3$$
$$\tau(\lambda)=-1/\ln|\lambda|\ (\text{months}),\qquad \lambda=\texttt{eigenvalue}(\tau)=e^{-1/\tau},\qquad \text{period}=2\pi/\theta$$

## Model
$$x(t)=Ax(t-1)+\hat b\,y(t)+\xi(t),\qquad \xi(t)\overset{iid}{\sim}\mathcal N(0,\,WDW^\top)$$
$$z=W^{-1}x:\qquad z(t)=\Lambda_Rz(t-1)+\gamma\,y(t)+\hat\xi(t),\qquad \gamma=W^{-1}\hat b,\quad \hat\xi\sim\mathcal N(0,D)$$
Simulation runs in modal coordinates (`run_modal`), and the result is mapped back by $x=Wz$.

## Noise from modal variances: `System.modal_variances`, `System.noise_variances`
$$D=\operatorname{diag}(\sigma_1^2,\sigma_p^2,\sigma_p^2,\sigma_4^2,\dots,\sigma_M^2)$$
$$s_1^2=\tfrac{\sigma_1^2}{1-\lambda_1^2},\quad s_p^2=\tfrac{\sigma_p^2}{1-\rho^2},\quad s_k^2=\tfrac{\sigma_k^2}{1-\lambda_k^2}\quad\Longleftrightarrow\quad \sigma_1^2=s_1^2(1-\lambda_1^2),\ \ \sigma_p^2=s_p^2(1-\rho^2),\ \ \sigma_k^2=s_c^2(1-\lambda_k^2)$$
The inputs are the three target modal variances $(s_1^2,s_p^2,s_c^2)$, stored as `s1_sq`, `sp_sq`, `sc_sq`. Changing an eigenvalue recomputes $\sigma^2$, so the modal variance stays fixed.

## Forcing: `co2_forcing_model`, `forcing_series`
$F$ is an analytic fit to the AR6 CO$_2$ effective radiative forcing (`data_preparation/AR6_ERF_1750-2019.csv`, column `co2`, W m$^{-2}$):
$$F(t_{\rm yr})=c+a\,e^{(t_{\rm yr}-2014)/\tau_F},\qquad c=0.0191,\ \ a=1.915,\ \ \tau_F=\texttt{forcing\_efold\_yr}=59.8\ \text{yr}$$
- **Fit.** Least squares to the annual values 1750–2019, placed at mid-year (`CO2_FIT` in `ablations.py`). The RMSE against the monthly AR6 series is $0.038$ W m$^{-2}$, and $0.055$ over the record. The largest misfit is the mid-century bump: the model is $0.10$ W m$^{-2}$ high around 1965 and $0.08$ low around 1920. A literal polynomial-plus-exponential diverges in the past (by 2100 BC it reaches $-5.3$ W m$^{-2}$ with a linear term and $+144$ with a quadratic one), and a polynomial factor only lowers the RMSE to $0.029$. So the plain exponential is used.
- **Constant past.** $F-c$ decays as $e^{(t-2014)/\tau_F}$, so $|F-c|/(F(2014)-c)<10^{-3}$ for every year before 1600. Over the longest spin-up (4000 yr, at $\tau_1=100$ yr), the forced slow mode varies by less than $10^{-3}$ of its record range between spin-up years 1000 and 3000, once the zero start has decayed. The `forcing` diagnostic shows this.
- **One shape parameter.** The record mean is removed and $a$ is recalibrated (below), so $c$ and the fitted amplitude drop out. Only $\tau_F$ shapes the forcing; a smaller $\tau_F$ puts more of the rise into the last decades.

**Exp + Gaussians** (`forcing_source = "analytic_gauss"`, `co2_forcing_gauss_model`). The exponential plus one positive and one negative Gaussian, all fitted jointly to the same AR6 values (`CO2_GAUSS_FIT`):
$$F(t)=c+a\,e^{k(t-2014)/100}+A_+e^{-(t-\mu_+)^2/2\sigma_+^2}-A_-e^{-(t-\mu_-)^2/2\sigma_-^2}$$
The fitted values are $c=-0.0054$, $a=1.964$ and $k=1.582$ (e-folding 63 yr), with a bump $A_+=0.048$ at $\mu_+=1921$ ($\sigma_+=17.9$ yr) and a dip $A_-=0.137$ at $\mu_-=1966$ ($\sigma_-=14.1$ yr).
- **Fit:** the RMSE is $0.0092$ W m$^{-2}$, and $0.0101$ over the record, against $0.038$ and $0.055$ for the exponential. The record shape, after centering and scaling to unit std, has an RMSE of $0.022$ against $0.124$.
- **Past:** the Gaussians vanish before about 1850, so the past is still constant: $|F-c|<3\times10^{-4}$ of the rise before 1500. `forcing_efold_yr` does not affect this model.
- **Comparison:** `compare_forcings.py` plots the true forcing and both models (`figures/diagnostics/data/compare_forcings.png`). It also plots the forced response of one system under each forcing (`compare_forced_responses.png`).

**Forcing source** (`forcing_source`, default `"analytic"`). With `forcing_source = "file"`, $F$ is instead column `forcing_column` (default `co2`) of `forcing_file` (default `data_preparation/AR6_ERF_1750-2019.csv`; a relative path is relative to the repo root). `load_forcing_file` accepts two formats:
- an annual CSV with a `year` column, interpolated to months with `interpolate` as for the real data;
- a monthly CSV with a `time` column (`YYYY-MM-01`), used as is.

Outside the data, $F$ is held at its first value (for AR6, $\approx0$: a constant past) and continued after the data at its linear trend over the last 10 years. Centering and $a$ below are the same for both sources. `forcing_efold_yr` only affects `analytic`, and `forcing_file`/`forcing_column` only affect `file`.

Record month $t=0,\dots,N-1$ is the fractional year $t_{\rm yr}=Y_0+t/12$, with $Y_0=\texttt{record\_end\_year}-99$. The default `record_end_year = 2014` gives a record from Jan 1915 to Dec 2014.
$$y(t)=a\Big(F(Y_0+t/12)-\tfrac1N\textstyle\sum_{s=0}^{N-1}F(Y_0+s/12)\Big),\qquad t=-T_f,\dots,N-1$$
The record mean is removed at all times, including spin-up. $a=$ `FORCING_AMPLITUDE` is fixed once so that $V^{(f)}=M$ at the starting point, which makes the mean forced variance per latitude 1. It is the same in every sweep. The forcing depends on absolute dates, so a longer spin-up or history only extends it backwards; the record values do not change.

With $a=1$, $V^{(f)}=1.477\times10^3$ at $\tau_1=20$ yr and $7.376$ at $\tau_1=0.84$ yr. These differ from the appendix values ($\approx4.4\times10^3$ and $\approx17$), which were computed with the appendix's forcing.

## Spin-up and ground truth: `System.noise_spinup`, `System.spinup`, `forced_response`, `internal_variability`
$$T_s=\max\big(\lceil 40\,\max_k\tau(\lambda_k)\rceil,\ 1200\big),\qquad T_f=\max(T_s,\ \texttt{history})$$
The noise spin-up $T_s$ uses the longest decay time of any mode, so it still covers the pair when $\tau_1<\tau_p$. It does not depend on the history, so changing the methods' history leaves the realizations unchanged. The forcing series and the forced response start $T_f$ months before the record, which is at least as long as the forcing history passed to the method.
$$x^{(f)}(t)=Ax^{(f)}(t-1)+\hat b\,y(t),\quad x^{(f)}(-T_f)=0\qquad(\text{shared by all realizations})$$
$$x^{(i)}(t)=Ax^{(i)}(t-1)+\xi(t),\quad x^{(i)}(-T_s)=0,\qquad x=x^{(f)}+x^{(i)}\ (\text{linear, so}\ =x-x^{(f)})$$
Times $t<0$ are discarded. Arrays are time-major: `forced` $(N,M)$, `internal` $(R,N,M)$, `data = forced + internal`.

**Common random numbers:** `SeedSequence(seed).spawn(2)` gives separate spin-up and record streams. Every level of every sweep therefore uses the same record noise $\varepsilon(t)$, $t\ge0$, with $\hat\xi=\sqrt D\,\varepsilon$.

## SNR: `theoretical_snr`, `empirical_snr`
$$V^{(f)}=\textstyle\sum_{i=1}^M\operatorname{Var}_t\,x^{(f)}_i(t),\qquad \mathrm{SNR}=\frac{V^{(f)}}{s_1^2+2s_p^2+(M-3)s_c^2}$$
The empirical SNR is $V^{(f)}/\sum_i\operatorname{Var}_t x^{(i)}_i$ per realization. It scatters around the theoretical value, especially for long $\tau_1$.

## Starting point: `REFERENCE`
| | value |
|---|---|
| step, record | monthly, $N=1200$ (100 yr), $M=20$ |
| patterns | `make_W(20, seed=22)` |
| forcing | $c+a\,e^{(t-2014)/59.8\,\text{yr}}$ fitted to the AR6 CO$_2$ ERF, record Jan 1915 – Dec 2014 (`record_end_year = 2014`); pattern $\hat b$ = zonal-mean CO$_2$ forcing |
| slow | $\tau_1=20$ yr, $\lambda_1=e^{-1/240}=0.9958$ |
| pair | $\tau_p=2$ yr, $\rho=e^{-1/24}=0.959$; period 4 yr, $\theta=2\pi/48$ (ACF first zero at 12 months) |
| complement | $\lambda_k\overset{iid}{\sim}\mathcal U(0,0.1)$, 17 values, seed 20 |
| budget (`equal_budget`) | $s_1^2=V^{(f)},\ s_p^2=V^{(f)}/2,\ s_c^2=V^{(f)}/17\ \Rightarrow\ \mathrm{SNR}=1/3$ |
| spin-up | $T_s=9601$ months |
| realizations | `N_REALIZATIONS = 100` per configuration, seed 0 |

## Sweeps
The patterns and eigenvalue seeds are fixed in every sweep. Only the noise realizations vary.

| study | swept | how |
|---|---|---|
| `total_snr_sweep` | $\mathrm{SNR}\in\{1/30,1/10,1/3,1,3,10,30\}$ | $(s_1^2,s_p^2,s_c^2)\times\frac{1/3}{\mathrm{SNR}}$; eigenvalues, patterns and forcing fixed. 10 and 30 extend the .tex grid toward a nearly noise-free end |
| `partial_snr_sweep(component)` | factor $\in\{1/4,1/2,1,2,4\}$ on $s_1^2$, $s_p^2$ or $s_c^2$ | the other two stay at the equal budget |
| `slow_timescale_sweep(hold="snr")` | $\tau_1\in\{1,2,5,10,20,50,100\}$ yr | `equal_budget` of each level's $V^{(f)}$, so SNR $=1/3$ |
| `slow_timescale_sweep(hold="modal_variance")` | same | budget fixed at the starting point, so SNR follows $V^{(f)}$ |
| `spatial_overlap_sweep` | $c\in\{0,0.25,0.457,0.75,0.9,0.95\}$ | $w_1\to\sqrt{1-c^2}\,u_\perp+c\,u_\parallel$, equal budget |

The slow-timescale sweep replaces the old temporal-overlap ablation: $\tau_1=1$–$2$ yr overlaps $\tau_p=2$ yr.

Spatial overlap (`tilt_slow`, `pair_plane_overlap`): $u_\parallel$ and $u_\perp$ are the unit parts of $w_1$ inside and outside $\operatorname{span}(w_2,w_3)$, and $c=\cos\angle(w_1,\operatorname{span}(w_2,w_3))$.
- $c=0.457$ is the default $w_1$.
- $w_1$ stays in $\operatorname{span}(S)$, so $Q_\perp$ is unchanged. $W^{-1}=[\operatorname{pinv}(S);Q_\perp^\top]$ is recomputed.

## Fitting interface: `SyntheticDataset.forcings(history=1200)`
$$\texttt{long\_forcings}=y(-\texttt{history}),\dots,y(N-1)\ \in\mathbb R^{(\texttt{history}+N)\times1},\qquad \texttt{short\_forcings}=y(0..N-1),\qquad \texttt{transition\_time}=\texttt{history}$$
The forcing enters at the target time, which matches `PullbackDMDc` ($x_t=Ax_{t-\text{lag}}+Bf_t$), so no shift is needed. The default 100-yr history matches the main analysis. Its truncation error $\sim e^{-100\,\text{yr}/\tau_1}$ becomes noticeable for $\tau_1$ of a few decades.

## Caveats
- **Sample variances.** For AR(1), the relative std of the sample variance over $N$ steps is $\approx\sqrt{2(1+\lambda^2)/((1-\lambda^2)N)}$, which is $0.63$ at $\tau_1=20$ yr. Mean removal also biases $\operatorname{Var}_t$ of the slow mode low.
- **Pair identifiability.** The pair covariance is isotropic, so evaluate the pair through its plane (principal angles) and its eigenvalue, not pattern by pattern.
- **Complement modes.** Individual modes and their forcing coefficients depend on the seed. Interpret only aggregates.
- **Lag-$\tau$ fits.** The propagator is $A^\tau$. The fitted forcing matrix approximates $(\sum_{m<\tau}A^m)\hat b$, so compare forced responses, not $B$.
- **Performance.** The per-step loop is BLAS-thread bound. Run with `OMP_NUM_THREADS=1`, which is about 50× less CPU time on the cluster nodes.

## System checks: `test.py` → `figures/tests/`
| check | figure(s) | asserts |
|---|---|---|
| `plot_system_check` | `system_check.png` | eigenvectors of $A$: slow $\parallel w_1$, pair $=c(w_2+iw_3)$ (1e-10) |
| `plot_ensemble_spaghetti` | `ensemble_spaghetti.png` | none (10 realizations + forced, annual means) |
| `check_dmdc_recovery` | `dmdc_recovery_check.png` | noise-free, white $y$: PullbackDMDc recovers $\lambda_1$, $\rho e^{i\theta}$, $\hat b$, $w_1$ and the plane $\operatorname{span}(w_2,w_3)$ (1e-8) |
| `check_forced_internal_split` | `forced_internal_spaghetti.png`, `forced_internal_convergence.png`, `modal_overview.png` | modal simulator equals a direct $x$-space simulation of the model (1e-9 relative); ensemble mean $\to$ forced at slope $<-0.4$. `modal_overview.png` (no asserts) shows the same 10 realizations in modal coordinates: forcing, $z_1$ and $z_2$ (full and internal only), patterns |
| `check_forced_internal_recovery` | `forced_internal_recovery.png`, `forced_internal_recovery_mse.png` | starting point only (SNR 1/3, 100 realizations): mirrored errors asserted; skill (correlation, errors, slow eigenvalue) reported, not asserted |

The recovery check also prints an oracle: the true $A,\hat b$ run over the same 100-yr history. That is the floor set by truncating the history ($\approx0.5\%$ at the starting point).
