# Synthetic data generation

Follows the synthetic-example appendix of the paper. Code: `synthetic_system.py` (patterns, modal matrix, pattern figures) and `ablations.py` (dynamics, forcing, simulation, sweeps). Tests: `test_ablations.py`. Diagnostics: `plot_ablations.py` → `figures/ablations/`.

## Grid and patterns: `lat_grid`, `raw_patterns`
$$\phi_i=-\tfrac{\pi}{2}+\tfrac{i\pi}{M-1},\quad i=0,\dots,M-1,\qquad M=20\ (\Delta\phi\approx 9.47^\circ,\ \text{poles included})$$
$$a_1=\cos(\phi-\tfrac{\pi}{4}),\quad a_2=\operatorname{sinc}(2\phi)=\tfrac{\sin(2\pi\phi)}{2\pi\phi},\quad a_3=\frac{0.5-\cos(10\phi)}{0.5+|10\phi|},\quad b=1.2-\cos\phi$$
$$w_k=a_k/\|a_k\|_2,\qquad \hat b=b/\|b\|_2\quad(\text{no area weighting})$$
Gram (uncentered): $w_1^\top w_2=0.396,\ w_1^\top w_3=0.329,\ w_2^\top w_3=0.278,\ \hat b^\top w_{1,2,3}=0.365,\,0.142,\,0.309$.

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

## Forcing: `forcing_F`, `forcing_series`
$$F(t_{\rm yr})=\tfrac{1}{30}\big(t_{\rm yr}-9\tanh(0.1t_{\rm yr}-4.5)+0.25e^{0.05t_{\rm yr}}\big)$$
$$y(t)=a\Big(F(t/12)-\tfrac1N\textstyle\sum_{s=0}^{N-1}F(s/12)\Big),\qquad t=-T_s,\dots,N-1$$
The record mean is removed at all times, including spin-up. $a=$ `FORCING_AMPLITUDE` is fixed once so that $V^{(f)}=M$ at the starting point, which makes the mean forced variance per latitude 1. It is the same in every sweep.

With $a=1$, $V^{(f)}=4.48\times10^3$ at $\tau_1=20$ yr (tex: $\approx4.4\times10^3$). At $\tau_1=0.84$ yr it is $20.2$ (tex: $\approx17$; the difference depends on the pair settings).

## Spin-up and ground truth: `System.spinup`, `forced_response`, `internal_variability`
$$T_s=\max\big(\lceil 40\,\max_k\tau(\lambda_k)\rceil,\ \texttt{HISTORY}\big)$$
The spin-up uses the longest decay time of any mode, so it still covers the pair when $\tau_1<\tau_p$. It is at least as long as the forcing history passed to the method.
$$x^{(f)}(t)=Ax^{(f)}(t-1)+\hat b\,y(t),\quad x^{(f)}(-T_s)=0\qquad(\text{shared by all realizations})$$
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

## System checks: `test.py` → `figures/`
| check | figure(s) | asserts |
|---|---|---|
| `plot_system_check` | `system_check.png` | eigenvectors of $A$: slow $\parallel w_1$, pair $=c(w_2+iw_3)$ (1e-10) |
| `plot_ensemble_spaghetti` | `ensemble_spaghetti.png` | none (10 realizations + forced, annual means) |
| `check_dmdc_recovery` | `dmdc_recovery_check.png` | noise-free, white $y$: PullbackDMDc recovers $\lambda_1$, $\rho e^{i\theta}$, $\hat b$, $w_1$ and the plane $\operatorname{span}(w_2,w_3)$ (1e-8) |
| `check_forced_internal_split` | `forced_internal_spaghetti.png`, `forced_internal_convergence.png`, `modal_overview.png` | modal simulator equals a direct $x$-space simulation of the model (1e-9 relative); ensemble mean $\to$ forced at slope $<-0.4$. `modal_overview.png` (no asserts) shows the same 10 realizations in modal coordinates: forcing, $z_1$ and $z_2$ (full and internal only), patterns |
| `check_forced_internal_recovery` | `forced_internal_recovery.png`, `forced_internal_recovery_mse.png` | starting point only (SNR 1/3, 100 realizations): mirrored errors asserted; skill (correlation, errors, slow eigenvalue) reported, not asserted |

The recovery check also prints an oracle: the true $A,\hat b$ run over the same 100-yr history. That is the floor set by truncating the history ($\approx1\%$ at the starting point).
