# QRM 545 Midterm — Exam Recipe Book

> **How to use this file**: Read the exam question → match it to a Recipe below → copy the code → fill in the blanks → report results + paste the wording template.

## Imports (paste at top of every exam notebook)
```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
sys.path.insert(0, ".")  # adjust if needed

from qrm_lib import chapter1, chapter2, chapter3, chapter4, chapter5
from qrm_lib import exam_utils as eu
```

## Conventions (avoid silent point-loss)
- `alpha = 0.05` → left-tail 5% (i.e., 95% confidence)
- Report VaR / ES as **positive loss** numbers (bigger = worse)
- Matrix shapes: `(n_assets, n_assets)` for covariance, `(n_sim, n_assets)` for simulations
- Demean returns when the problem says "demean" or "zero-mean" — use `eu.demean(X)`

---

# Recipe A — Moments + Normal vs t Fit (AICc)

## Trigger keywords
"mean/variance/skew/kurtosis", "Normal vs t", "AICc", "fit distribution", "which distribution fits better"

## Code (copy → fill blanks → run)
```python
# ── load data ──
x = eu.load_returns_vector("YOURFILE.csv")
# or if data is already a column in a DataFrame:
# x = df["column_name"].dropna().to_numpy()

# ── run pipeline ──
res = eu.pipeline_moments_and_fit(x)
eu.print_moments(res)
```

## What to report
| Item | Value |
|---|---|
| Mean | `res["mean"]` |
| Variance | `res["variance"]` |
| Skewness | `res["skewness"]` |
| Excess Kurtosis | `res["excess_kurtosis"]` |
| Normal AICc | `res["fit"]["aicc_norm"]` |
| t AICc | `res["fit"]["aicc_t"]` |
| Best model | `res["best_model"]` |

## Wording template (fill blanks)
> "The sample has skewness {skew:.3f} and excess kurtosis {ek:.3f}. Positive excess kurtosis indicates heavier tails than the Normal distribution. I fit both Normal and Student-t via MLE and compare using AICc (small-sample corrected AIC). The {best_model} has lower AICc ({aicc_best:.2f} vs {aicc_other:.2f}), indicating better fit to the data."

## If the problem has a twist
- **Only moments, no fitting**: just use `chapter1.first4_moments(x)`
- **Custom distribution**: use `scipy.stats.<dist>.fit(x)` + `chapter1.calculate_aicc(ll, n, k)`

---

# Recipe B — Univariate VaR / ES

## Trigger keywords
"VaR at 5%", "ES at 5%", "parametric vs empirical", "compare Normal vs t risk"

## Code
```python
x = eu.load_returns_vector("YOURFILE.csv")
alpha = 0.05  # ← change if problem says different

res = eu.pipeline_var_es_univariate(x, alpha=alpha)
eu.print_var_es(res)
```

## What to report
| Metric | Empirical | Normal | t |
|---|---|---|---|
| VaR | `res["empirical_var"]` | `res["normal_var"]` | `res["t_var"]` |
| ES | `res["empirical_es"]` | `res["normal_es"]` | `res["t_es"]` |

## Wording template
> "VaR at {alpha*100:.0f}% is the loss threshold where only {alpha*100:.0f}% of outcomes are worse. ES is the average loss in that worst tail — it is always ≥ VaR."
>
> "Under the t-distribution, ES is notably larger than under Normal because the t has heavier tails, putting more probability mass in extreme losses. VaR changes less because it only marks the tail boundary, while ES averages over the entire tail."

## If the problem has a twist
- **Dollar VaR**: multiply return VaR by portfolio value
- **Multi-day VaR**: `chapter4.var_normal(mu, sigma, alpha, n_days=10)` (square-root-of-time)
- **Only empirical**: `eu.var_es_from_pnl(pnl_array, alpha)`

---

# Recipe C — Exponentially Weighted Covariance

## Trigger keywords
"EW covariance", "lambda", "exponential weights", "RiskMetrics", "recent weighting"

## Code
```python
X, cols = eu.load_returns_matrix("YOURFILE.csv")
lam = 0.97  # ← change per problem

res = eu.pipeline_ew_covariance(X, lam=lam, do_demean=True)

print("EW Covariance:\n", pd.DataFrame(res["ew_cov"], index=cols, columns=cols))
print("EW Correlation:\n", pd.DataFrame(res["ew_corr"], index=cols, columns=cols))
print(f"PSD: {res['is_psd']},  Min eigenvalue: {res['min_eig']:.8f}")
```

## What to report
- EW covariance matrix (and/or correlation matrix)
- Lambda used and whether matrix is PSD

## Wording template
> "Exponentially weighted estimates emphasize recent observations via decay factor λ={lam}. Higher λ → longer memory (smoother), lower λ → faster reaction to new shocks. This captures time-varying volatility, which is central to the RiskMetrics approach."

## If the problem has a twist
- **EW correlation + separate EW variances → combined cov**: compute corr from `res["ew_corr"]`, compute variances separately, then `cov = D @ corr @ D` where `D = diag(sqrt(var))`
- **Compare EW vs equal-weight**: run `np.cov(eu.demean(X), rowvar=False)` alongside

---

# Recipe D — Missing Data → PSD Fix → PCA Variance

## Trigger keywords
"pairwise covariance", "missing values", "not PSD", "Higham", "near_psd", "PCA explained variance"

## Code
```python
X, cols = eu.load_returns_matrix("YOURFILE.csv")

res = eu.pipeline_missing_psd_pca(X, fix_method="higham", do_demean=True)
# fix_method: "higham" (optimal) or "near_psd" (Rebonato-Jackel, faster)

print(f"Min eigenvalue BEFORE fix: {res['min_eig_before']:.8f}")
print(f"Min eigenvalue AFTER  fix: {res['min_eig_after']:.8f}")
print(f"\nPCA cumulative variance (first 10):")
print(res["pca_table"].head(10))
```

## What to report
| Item | Value |
|---|---|
| Min eigenvalue (pairwise) | `res["min_eig_before"]` — if negative → not PSD |
| Min eigenvalue (after fix) | `res["min_eig_after"]` — should be ≥ 0 |
| PCA explained variance | `res["pca_table"]` |
| k components for 75%/90% | find in `res["pca_cumulative"]` |

## Wording template
> "Pairwise covariance uses different observation subsets for each pair, which can produce a matrix that is not positive semi-definite (min eigenvalue = {min_before:.6f} < 0). Higham's alternating projection algorithm finds the nearest PSD matrix under Frobenius norm, yielding min eigenvalue = {min_after:.6f} ≥ 0."
>
> "PCA eigenvalues represent variance along orthogonal principal components. The first k={k} components explain {pct:.1f}% of total variance, indicating the effective dimension of the data."

## If the problem has a twist
- **Complete-case instead of pairwise**: `chapter3.missing_cov(X, skip_miss=True)`
- **Compare both methods**: run with `skip_miss=True` and `skip_miss=False`, compare eigenvalues
- **near_psd vs Higham comparison**: run both, compare Frobenius norms to original

---

# Recipe E — Copula Simulation → Portfolio VaR/ES

## Trigger keywords
"Gaussian copula", "fit marginals (t/Normal)", "Spearman", "simulate joint returns", "portfolio VaR/ES", "per-asset VaR/ES"

## Code (from prices CSV — most common exam format)
```python
alpha = 0.05
holdings = np.array([100, 200, 150])  # ← from problem
# dist_types per asset: ["t", "t", "normal"] or all "t"
dist_types = ["t", "t", "normal"]  # ← from problem

res = eu.pipeline_copula_portfolio_from_prices(
    prices_csv="PRICES.csv",
    holdings=holdings,
    dist_types=dist_types,
    alpha=alpha,
    n_sim=10000,
    method="spearman",  # ← or "pearson" if problem says
    seed=42,
    return_method="discrete",  # or "log"
    do_demean=True,
)

eu.print_portfolio_risk(res)
```

## Code (from returns array — if data already loaded)
```python
alpha = 0.05
R = ...          # (T, n_assets) returns matrix
holdings = ...   # (n_assets,) shares
cur_prices = ... # (n_assets,) current prices

res = eu.pipeline_copula_portfolio(
    returns=R,
    holdings=holdings,
    current_prices=cur_prices,
    dist_types=["t", "t", "normal"],
    alpha=alpha,
    n_sim=10000,
    method="spearman",
    seed=42,
)

eu.print_portfolio_risk(res)
```

## What to report
| Item | Where |
|---|---|
| Copula correlation matrix | `res["copula_corr"]` |
| Correlation method | `res["method"]` (spearman/pearson) |
| Marginal params | `res["marginals"]` (list of dicts with mu, sigma, nu) |
| Portfolio VaR ($) | `res["portfolio_var"]` |
| Portfolio ES ($) | `res["portfolio_es"]` |
| Per-asset VaR/ES ($) | `res["asset_var_es"]` (list of dicts) |

## Wording template
> "I model dependence using a Gaussian copula with {method} correlation (rank-based, robust to heavy tails). Each asset's marginal is fitted independently: {describe marginals}. After simulating {n_sim} joint return scenarios, I convert to dollar PnL using current holdings and prices, then compute VaR/ES empirically from the simulated PnL distribution."
>
> "Portfolio VaR at {alpha*100:.0f}% = ${var:,.2f}, ES = ${es:,.2f}. Note ES > VaR because ES averages over the entire tail."

## If the problem has a twist
- **No holdings / just returns**: skip `portfolio_pnl`, compute VaR/ES directly on simulated returns
- **Only copula correlation, no VaR**: use `chapter5.simulate_copula_mixed()` directly
- **Custom marginals**: use `chapter5.fit_copula_marginals()` + `chapter5.fit_gaussian_copula_corr()` + `chapter5.simulate_copula_from_fitted()` step by step
- **Multiple alpha levels (e.g. 5% AND 1%)**: use Recipe E2 below

---

# Recipe E2 — Copula VaR/ES at Multiple Alpha Levels

## Trigger keywords
"VaR/ES at 5% and 1%", "two confidence levels", "multiple alpha", "compare risk at different levels"

## Code
```python
alphas = [0.05, 0.01]  # ← adjust per problem
holdings = np.array([100, 200, 150])  # ← from problem
dist_types = ["t", "t", "normal"]

res = eu.pipeline_copula_multi_alpha_from_prices(
    prices_csv="PRICES.csv",
    holdings=holdings,
    alphas=alphas,
    dist_types=dist_types,
    n_sim=10000,
    method="spearman",
    seed=42,
    do_demean=True,
)

eu.print_multi_alpha_risk(res)

# Or access individual alpha results:
for a in alphas:
    r = res["results_by_alpha"][a]
    print(f"\nAlpha={a}: Portfolio VaR=${r['portfolio_var']:,.2f}, ES=${r['portfolio_es']:,.2f}")
```

## What to report
For each alpha level: portfolio VaR, portfolio ES, per-asset VaR/ES

## Wording template
> "I simulate once with a Gaussian copula ({n_sim} draws), then compute VaR/ES at multiple confidence levels from the same PnL distribution. At α=0.05 (95%), Portfolio VaR=${var5:,.2f}, ES=${es5:,.2f}. At α=0.01 (99%), VaR=${var1:,.2f}, ES=${es1:,.2f}. The 99% figures are larger because we are looking deeper into the tail."

---

# Recipe F — Cholesky vs PCA Simulation Comparison

## Trigger keywords
"compare Cholesky and PCA", "simulation speed vs accuracy", "Frobenius norm error"

## Code
```python
# cov = ... (from pipeline_ew_covariance, pipeline_missing_psd_pca, or np.cov)
cov = res_c["ew_cov"]  # or res_d["cov_fixed"] — use the PSD-fixed version!

res = eu.pipeline_cholesky_vs_pca(cov, n_sim=10000, pct_exp=0.75, seed=42)

print(f"Cholesky: time={res['time_cholesky']:.4f}s, Frobenius error={res['frobenius_cholesky']:.6f}")
print(f"PCA:      time={res['time_pca']:.4f}s, Frobenius error={res['frobenius_pca']:.6f}")
```

## What to report
| Metric | Cholesky | PCA |
|---|---|---|
| Time (s) | `res["time_cholesky"]` | `res["time_pca"]` |
| Frobenius error | `res["frobenius_cholesky"]` | `res["frobenius_pca"]` |

## Wording template
> "Cholesky uses the full covariance matrix (no approximation), so Frobenius error comes only from sampling noise. PCA keeps the top eigenvalues explaining {pct*100:.0f}% of variance, reducing dimensionality but introducing approximation error."
>
> "PCA is faster in high dimensions because it simulates fewer independent normals. The tradeoff: Cholesky has higher fidelity, PCA has speed + dimensionality reduction."

---

# Bonus Recipe — Conditional Distribution (Bivariate Normal)

## Trigger keywords
"conditional distribution", "X2 given X1", "conditional mean/variance", "bivariate normal"

## Code
```python
mean = np.array([mu1, mu2])
cov = np.array([[s11, s12],
                [s12, s22]])
x1_observed = ...  # given value

# Closed-form
mu_cond, var_cond = chapter3.conditional_bivariate_stats(mean, cov, x1_observed)
print(f"E[X2 | X1={x1_observed}] = {mu_cond:.4f}")
print(f"Var[X2 | X1={x1_observed}] = {var_cond:.4f}")

# Simulation verification
sim_x2 = chapter3.simulate_conditional_bivariate(mean, cov, x1_observed, nsim=10000)
print(f"Simulated mean: {sim_x2.mean():.4f}, var: {sim_x2.var():.4f}")
```

## Wording template
> "For bivariate normal (X1, X2), the conditional distribution X2|X1=x1 is also normal with mean μ₂ + (σ₁₂/σ₁₁)(x1 − μ₁) and variance σ₂₂ − σ₁₂²/σ₁₁. Information about X1 shifts the mean (toward the correlation direction) and reduces the variance."

---

# Bonus Recipe — AR/MA Model Selection

## Trigger keywords
"best AR order", "AICc comparison", "fit AR", "fit MA", "ACF/PACF"

## Code
```python
y = eu.load_returns_vector("YOURFILE.csv")

# Find best AR order by AICc
best_p, table = chapter2.select_best_ar_order_aicc(y, p_min=1, p_max=5)
print(f"Best AR order: {best_p}")
print(table)

# Fit the best model and get residuals
model = chapter2.fit_ar_model(y, p=best_p)
print(f"Coefficients: {model.params}")
print(f"AICc: {chapter2.model_result_aicc(model):.2f}")
residuals = model.resid
```

## Wording template
> "I scan AR orders from 1 to {max_p} and select the order minimizing AICc. AR({best_p}) has AICc = {aicc:.2f}, the lowest among candidates. AICc penalizes both poor fit (high likelihood) and model complexity (number of parameters), with a small-sample correction."

---

# Quick Decision Tree

```
Exam question mentions...
│
├─ "moments / skew / kurtosis / Normal vs t"  →  Recipe A
├─ "VaR / ES at X%"                           →  Recipe B (univariate)
│                                                 or Recipe E (portfolio)
├─ "EW / lambda / exponential weights"         →  Recipe C
├─ "missing data / pairwise / PSD / Higham"    →  Recipe D
├─ "copula / marginals / joint simulation"     →  Recipe E
├─ "Cholesky vs PCA / speed / Frobenius"       →  Recipe F
├─ "conditional distribution / X2|X1"          →  Bonus: Conditional
├─ "AR / MA / best order / AICc"              →  Bonus: AR/MA
├─ "cov↔corr / mixed lambda EW cov"          →  Bonus: Cov/Corr Utils
├─ "t VaR / simulate VaR/ES"                  →  Bonus: t Simulation Risk
├─ "regression with t errors / t-regression"   →  Bonus: t-Regression
└─ "simulate with Higham fix"                  →  Bonus: Higham Simulation
```

**Common combos**: D → F (fix matrix, then compare simulations), C → E (EW cov as input to copula), A → B (fit distribution then compute risk)

---

# Bonus Recipe — Covariance ↔ Correlation Conversion & Mixed-Lambda EW

## Trigger keywords
"convert covariance to correlation", "different lambda for variance and correlation", "EW cov from corr+var"

## Code
```python
# Convert cov → corr
corr = eu.cov_to_corr(cov_matrix)

# Convert corr → cov (given standard deviations)
cov_back = eu.corr_to_cov(corr, std_vector)

# Mixed-lambda EW covariance: λ_var for variances, λ_corr for correlations
cov_mixed = eu.ew_cov_mixed_lambda(X, lam_var=0.97, lam_corr=0.94, do_demean=True)
```

## Wording template
> "Using separate decay rates lets you capture different persistence in variance (λ_var={lam_v}) vs correlation (λ_corr={lam_c}). Variance often has slower decay (higher λ), while correlations may shift faster during crises."

---

# Bonus Recipe — t-Distribution VaR & Simulation-Based VaR/ES

## Trigger keywords
"t VaR standalone", "simulate VaR/ES from t", "Monte Carlo VaR for t distribution"

## Code
```python
# Standalone t VaR (closed-form, no ES)
nu, mu, sigma = 5.0, 0.001, 0.02  # ← from fitting
var_value = eu.var_t(nu, mu, sigma, alpha=0.05)

# Simulation-based VaR/ES (works when closed-form is hard)
sim_res = eu.simulate_t_var_es(nu, mu, sigma, alpha=0.05, n_sim=100000, seed=42)
print(f"Sim VaR: {sim_res['var']:.6f}, Sim ES: {sim_res['es']:.6f}")

# Generic: VaR/ES from any simulation output
var, es = eu.var_es_from_simulation(sim_array, alpha=0.05)
```

## Wording template
> "When closed-form ES is complex (e.g., for mixture distributions), simulation-based estimation provides a flexible alternative: simulate {n_sim} draws, then compute empirical VaR/ES from the simulated distribution."

---

# Bonus Recipe — Regression with t-Distributed Errors

## Trigger keywords
"t-regression", "regression with heavy-tailed errors", "MLE regression with t errors"

## Code
```python
X = ...  # (n, p) design matrix (features)
y = ...  # (n,) response vector

result = eu.mle_regression_t(X, y)
print(f"Beta: {result['beta']}")
print(f"Sigma: {result['sigma']:.6f}")
print(f"Nu (df): {result['nu']:.2f}")
print(f"AICc: {result['aicc']:.2f}")
residuals = result["residuals"]
```

## Wording template
> "Standard OLS assumes Gaussian errors, but financial data often has heavy tails. A t-regression fits beta, sigma, and degrees-of-freedom nu jointly via MLE, capturing tail heaviness. Here nu={nu:.2f}, indicating {'heavier tails than Normal' if nu < 30 else 'approximately Normal tails'}."

---

# Bonus Recipe — Cholesky Simulation with Higham PSD Fix

## Trigger keywords
"simulate with Higham", "Cholesky fails / not PSD", "force PSD then simulate"

## Code
```python
# If your covariance matrix isn't PSD, this auto-fixes via Higham before Cholesky
sim = eu.simulate_normal_higham(cov_matrix, nsim=10000, mean=None, seed=42)
# sim.shape = (10000, n_assets)
```

## Wording template
> "The pairwise covariance matrix was not PSD (min eigenvalue < 0), so I applied Higham's nearest PSD projection before Cholesky factorization. This ensures the simulation is valid while minimizing distortion to the original covariance structure."

---

# Atomic Functions Quick Lookup

If a recipe doesn't fit and you need to go manual:

| Need | Function | Import |
|---|---|---|
| 4 moments | `first4_moments(x)` | `chapter1` |
| Normal vs t AICc | `fit_normal_vs_t_aicc(x)` | `chapter1` |
| Pearson + Spearman | `calculate_correlations(x, y)` | `chapter2` |
| Fit AR(p) | `fit_ar_model(y, p)` | `chapter2` |
| Best AR order | `select_best_ar_order_aicc(y)` | `chapter2` |
| Simulate AR(p) | `simulate_ar_process(n, ar_params, c=0, sigma=1)` | `chapter2` |
| Missing-data cov | `missing_cov(X, skip_miss=False)` | `chapter3` |
| EW covariance | `ew_covariance(X, lam)` | `chapter3` |
| Fix PSD (fast) | `near_psd(A)` | `chapter3` |
| Fix PSD (optimal) | `higham_nearest_psd(A)` | `chapter3` |
| Cholesky simulation | `simulate_normal_cholesky(cov, nsim)` | `chapter3` |
| PCA simulation | `simulate_pca(cov, nsim, pct_exp=0.75)` | `chapter3` |
| PCA variance curve | `pca_cumulative_variance(cov)` | `chapter3` |
| Conditional X2\|X1 | `conditional_bivariate_stats(mean, cov, x1)` | `chapter3` |
| Prices → returns | `return_calculate(prices_df)` | `chapter4` |
| Normal VaR | `var_normal(mu, sigma, alpha)` | `chapter4` |
| Empirical VaR+ES | `calculate_es(data, alpha)` → `(VaR, ES)` tuple | `chapter5` |
| Normal ES | `calculate_es_normal(mu, sigma, alpha)` | `chapter5` |
| t ES | `calculate_es_t(nu, mu, sigma, alpha)` | `chapter5` |
| Fit t marginal | `fit_general_t(data)` | `chapter5` |
| Copula simulate | `simulate_copula_mixed(data, dist_types, n_sim, method)` → `(sim, corr, marginals)` | `chapter5` |
| Fit marginals → U | `fit_copula_marginals(data, dist_types)` → `(u_data, marginals)` | `chapter5` |
| Copula corr from U | `fit_gaussian_copula_corr(u_data, method)` | `chapter5` |
| Simulate from fitted | `simulate_copula_from_fitted(marginals, corr, n_sim)` | `chapter5` |
| **PnL from sim** | `portfolio_pnl(sim_R, holdings, prices)` | `exam_utils` |
| **VaR+ES from PnL** | `var_es_from_pnl(pnl, alpha)` | `exam_utils` |
| **Demean** | `demean(X)` | `exam_utils` |
| **PSD check** | `is_psd(A)` / `min_eigenvalue(A)` | `exam_utils` |
| **Cov → Corr** | `cov_to_corr(cov)` | `exam_utils` |
| **Corr → Cov** | `corr_to_cov(corr, std)` | `exam_utils` |
| **Mixed-λ EW cov** | `ew_cov_mixed_lambda(X, lam_var, lam_corr)` | `exam_utils` |
| **t VaR (closed-form)** | `var_t(nu, mu, sigma, alpha)` | `exam_utils` |
| **VaR+ES from sim** | `var_es_from_simulation(sim_data, alpha)` | `exam_utils` |
| **Sim t → VaR+ES** | `simulate_t_var_es(nu, mu, sigma, alpha, n_sim)` | `exam_utils` |
| **Higham + Cholesky sim** | `simulate_normal_higham(cov, nsim)` | `exam_utils` |
| **t-Regression MLE** | `mle_regression_t(X, y)` | `exam_utils` |
| **Multi-α copula** | `pipeline_copula_multi_alpha(R, h, p, alphas)` | `exam_utils` |
