# Synthetic experiments

A 20-latitude linear system with a slow mode, a fast oscillating (ENSO-like) pair and 17 noise modes, driven by a
CO₂-like forcing and observed with measurement noise. Four estimators (`PullbackDMDc`, `LIM`, `LIM-opt`, `LR`)
are fitted to every member of an ensemble, at every level of every ablation study. Each is scored on how well
it recovers the forced response and the slow mode.

The system is defined in **`synthetic_generator.py`**, written by Andrei Gavrilov. Its docstring is the
specification. Everything else in this folder is a thin layer on top of it. Please keep his file in his style,
and put new logic in the other files.

## Run it

```bash
./run_synthetic_experiments.sh   # both ablation runs and every figure
./run_synthetic_tests.sh         # the unit tests (run_tests.py; --only <substring>, --list)
```

The drivers activate the `dmdc_variants` env, `cd` into `synthetic/` and export `OMP_NUM_THREADS=1` (the per-step
simulation loop is BLAS-thread bound) and `MPLBACKEND=Agg`. Scripts are run from inside `synthetic/`. Use
`--n-jobs 12` for ablation runs on the shared node.

```bash
python run_ablation_studies.py --n-jobs 12                              # reference run: results/delta0.15/
python run_ablation_studies.py --set delta=0.5 --n-jobs 12              # deeper dip: results/delta0.5/
python plot_generator_diagnostics.py                                    # setup.pdf, pullback_results.pdf
python plot_generator_diagnostics.py --study obs_noise --level 1       # ... at any ablation level
python run_ablation_studies.py --name slow50 --set tau1=50 --n-jobs 12  # a run with a 50-yr slow mode
python plot_ablation_run_results.py --runs delta0.15 delta0.5           # both dip depths overlaid
python plot_ablation_run_results.py --runs delta0.15 --metric corr      # forced correlation instead (*_corr.png)
python plot_ablation_run_results.py --runs delta0.15 --metric r2        # forced R² instead (*_r2.png)
```

`--set KEY=VALUE` overrides any `Config` field. Values are Python literals, e.g. `--set "variances=(1, 1, 1, 0.9, 0.5)"`.
Runs and their figures are named by dip depth (`delta0.15`, `delta0.5`) unless `--name` is given.
`--from-run NAME` starts from a saved run's `config.json` instead of the defaults.

## Files

| file | what it is |
|---|---|
| `synthetic_generator.py` | **his**: `make_setup` (W(α), b₀, forcing), `generate_ensemble` (the ensemble dict), `fit_pullback_ensemble` |
| `plot_generator_diagnostics.py` | **his** plots (`plot_setup`: patterns and the forcing at δ = 0.15 and 0.5; `plot_pullback_results`) and a CLI → `figures/generator/delta<δ>/` |
| `config.py` | `Config`: the system fields are `generate_ensemble`'s arguments under the same names, plus seeds, sweep levels and methods |
| `ablations.py` | `STUDIES`: how each study bends the reference; `make_level(cfg, study, level)` returns `generate_ensemble`'s dict |
| `methods.py` | the four estimators, fitted one member at a time on identity EOFs |
| `run_ablation_studies.py` | fits and scores every level → `results/<run>/` |
| `plot_ablation_run_results.py` | sweep, phase-diagram, slow-mode-shape and forcing-change figures → `figures/ablations/<run>/` or `compare_<run>_vs_<run>/`; with several runs the first is solid with its interquartile band and the others dotted medians; no captions |
| `plot_style.py` | shared matplotlib style of the results figures |

## Reference system (`Config` defaults)

Notation follows the paper:
- `tau1=20` yr (τ₁);
- `variances=(1, 1, 1, 0.9, 0.1)`, the ratio V_f : V_1 : V_p : V_c : V_ε of the forced variance
  V_f = Σᵢ Var_t x_i^(f) and the variances of the slow mode, fast pair, noise modes and measurement noise, so
  SNR = 1/3. The measurement noise is white with the same σ_ε = √(V_ε / M) at every latitude;
- uniform forcing pattern b₀ = [1, …, 1]ᵀ, dip depth `delta=0.15` (δ); the forcing is
  F(t) = exp((t − 2014)/65) − δ exp(−½((t − 1965)/15)²), and every experiment is repeated at δ = 0.5;
- overlap `alpha=0` (α): the complement modes are rotated toward the slow pattern,
  q_k(α) = α w₁ + √(1−α²) q_k, with the slow and fast patterns unchanged; α = 0 is the orthonormal complement;
- `spin_up_factor=100` decay times;
- `tau_pair=1.5` yr and `period_pair=4` yr;
- 100 members over T = 1980 months, fitted at `lag=1`.

## Ablation studies

| study | levels | what changes (`generate_ensemble` argument) |
|---|---|---|
| `total_snr` | SNR = 1/100, 1/30, 1/10, 1/3 (ref), 1, 3, 10, 30, 100 | every noise component in `variances`, scaled by `snr_ref / snr` |
| `partial_snr_{slow,pair,complement}` | factor ¼ … 4 | one noise component in `variances` |
| `slow_timescale_snr` | τ1 = 1, 2, 5, 10, 20 (ref), 40, 70, 100 yr | `tau1`; SNR fixed, because `variances` is relative to each level's own V_f |
| `slow_timescale_modal_variance` | same τ1 levels | `tau1`, with the noise shares rescaled by V_f(ref) / V_f(τ1) so that the absolute variances stay fixed |
| `obs_noise` | V_ε/V_f = 0.003, 0.01, 0.03, 0.1 (ref), 0.3, 1, 3 | the measurement-noise component `variances[4]`, internal components fixed, so the SNR falls |
| `overlap` | α = 0 (ref), 0.3, 0.5, 0.7, 0.9, 0.99 | `alpha`, the overlap of every complement mode with w₁ |
| `joint_snr_timescale` | 9 × 8 grid (SNR × τ1 levels) | `tau1` and the `variances` scale together (phase diagram) |

The sweep figures draw `total_snr`, `slow_timescale_snr`, `obs_noise` and `overlap` by default. The partial-SNR and
fixed-variance studies are still run, and are drawn with `--studies`.

A variant of the whole system is a run, not a study. Every study is repeated at δ = 0.5 (`--set delta=0.5`).

## Outputs

`results/<run>/` holds:
- `config.json`, which fully determines the run's numbers;
- `provenance.json`, with the seeds, conventions and package versions;
- `ablations.csv`, with one row per (study, level, realization, method), the scores
  (`forced_rel_rmse`, `forced_corr`, `forced_r2`, `slow_eig`, `slow_tau_yr`, `slow_corr`, `pair_corr`, …) and `snr`/`tau1_yr`.
  The results figures show any forced-response score (`--metric rel_rmse`, the default, `corr` or `r2`):
  - `forced_rel_rmse = rms(est − truth) / rms(truth − its time mean)`, so a constant offset counts;
  - `forced_corr` is the correlation of estimate and truth after centering each latitude over time and
    vectorizing, so only the shape counts;
  - `forced_r2 = 1 − SSE/SST = 1 − forced_rel_rmse²`, the fraction of the forced variance explained, with an
    uncentered SSE and the truth's centered variance as SST; below 0 is worse than the truth's own mean;
- `slow_modes.npz`, the fitted slow-mode patterns plus the truth per level.

## Reproducibility

All randomness is seeded:
- `w_seed` sets the noise-mode patterns.
- `eig_seed` sets the noise-mode eigenvalues.
- `noise_seed` sets the `rng` of `generate_ensemble`.

Every level draws from the same `noise_seed`. So levels with the same spin-up length share their noise draws,
which is every study except the τ1 ones. A run is a pure function of its `config.json`, given the same environment.

## Notes

- **Centering conventions.** The forcing y is centered over the record. Nothing else is centered: not the
  ensemble members, not the true forced response, not the estimates. Each method is used as on real data. A
  realization's EOFs and data mean are computed once (`data_preparation/compute_eofs.run_eofs`) and passed to every
  class, which centers internally and adds the mean back in `predict()`. `forced_rel_rmse` has an uncentered
  numerator and a centered denominator: `rms(est − truth) / rms(truth − its time mean)`.
- **Eigen-metrics.** These come from each model's eigenvalues and its grid-space eigenvectors `eofs.T @ W`, as in the
  real-data scripts. LR and LIM-opt have no operator. The fitted slow mode is the largest real eigenvalue λ̂ of
  the lag-step propagator, chosen without the truth as it must be on real data. Its decay time is
  τ̂ = −lag / (12 ln λ̂) years. A fit with λ̂ ≥ 1, or with no positive real eigenvalue, is counted in
  `slow_unstable` and gets no decay time or slow-pattern score.
- The slow mode's variance over the 165-yr record, about the record mean, is about 0.8 of its stationary value
  at τ1 = 20 yr, so the empirical SNR of a member sits slightly above the theoretical one.
