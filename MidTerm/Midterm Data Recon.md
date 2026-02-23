# Midterm Data Recon — 考前数据侦察 + 预写代码（完整版）

> **目的**：我们有明天考试的 4 个 input CSV，但没有题目。通过分析数据结构，**预测可能的题型** 并 **预写代码骨架**，考试时只需填入具体参数。
>
> ⚠️ **重要原则**：每个 Problem 下面既有"一键 pipeline"代码，也有"分步 atomic"代码。考试如果问的是完整流程就用 pipeline；如果分成小问就用 atomic 逐步做。

---

# 通用 Import（每个 Problem 顶部粘贴）

```python
import numpy as np, pandas as pd, scipy.stats as stats, sys
sys.path.insert(0, "..")
from qrm_lib import chapter1, chapter2, chapter3, chapter4, chapter5
from qrm_lib import exam_utils as eu
```

---

# Problem 1 — 单变量 Returns

## 数据特征

| 属性 | 值 |
|---|---|
| 文件 | `problem1.csv` |
| Shape | (1000, 1)，列名 `X` |
| 数据类型 | 单资产 returns（看数值范围 ~±0.03） |
| Mean | -0.000532 |
| Std | 0.0317 |
| Skewness | **-0.24**（轻微左偏） |
| Excess Kurtosis | **+0.69**（heavy tails vs Normal） |
| Missing | 无 |

## 最可能的题型

> 单列 returns → 可能出 **完整流程** 或 **任意一小步**：

| 可能的小问 | 对应 atomic 函数 |
|---|---|
| (a) 计算 4 moments | `chapter1.first4_moments()` |
| (b) Fit Normal (MLE) | `stats.norm.fit()` 或直接用 mean/std |
| (c) Fit t (MLE) | `chapter5.fit_general_t()` |
| (d) AICc 比较 Normal vs t | `chapter1.fit_normal_vs_t_aicc()` |
| (e) Normal VaR | `chapter4.var_normal()` |
| (f) Normal ES | `chapter5.calculate_es_normal()` |
| (g) t-VaR | `eu.var_t()` 或 `-stats.t.ppf(alpha, df=nu, loc=mu, scale=sigma)` |
| (h) t-ES | `chapter5.calculate_es_t()` |
| (i) Empirical VaR | `chapter4.var_historical()` 或 `-np.quantile(x, alpha)` |
| (j) Empirical ES | `chapter5.calculate_es()` |
| (k) Simulation-based VaR/ES | `eu.simulate_t_var_es()` |
| (l) AR model selection | `chapter2.select_best_ar_order_aicc()` |
| (m) Plot histogram + fitted PDF | 画图题 |
| (n) Hypothesis test (Jarque-Bera) | `stats.jarque_bera()` |
| (o) QQ-plot | `stats.probplot()` |

## 预写代码 — 一键 Pipeline

```python
# ============ PROBLEM 1 — PIPELINE VERSION ============
x = eu.load_returns_vector("problem1.csv")

# Moments + Fit (Normal vs t)
res_a = eu.pipeline_moments_and_fit(x)
eu.print_moments(res_a)

# VaR / ES (all three methods at once)
alpha = 0.05  # ← 看题目
res_b = eu.pipeline_var_es_univariate(x, alpha=alpha)
eu.print_var_es(res_b)
```

## 预写代码 — 分步 Atomic（按小问复制）

### Step 1: 计算 4 Moments

```python
x = eu.load_returns_vector("problem1.csv")
mean, var, skew, ex_kurt = chapter1.first4_moments(x)
print(f"Mean={mean:.6f}, Var={var:.6f}, Skew={skew:.4f}, ExKurt={ex_kurt:.4f}")
```

**Wording**: "The sample mean is {mean:.6f}, variance is {var:.6f}. Skewness = {skew:.4f} indicates {'negative/left-tail asymmetry' if skew < 0 else 'positive/right-tail asymmetry'}. Excess kurtosis = {ex_kurt:.4f} {'> 0 indicates heavier tails than Normal (leptokurtic)' if ex_kurt > 0 else '≈ 0, consistent with Normal tails'}."

### Step 2: Fit Normal (MLE)

```python
mu_n = np.mean(x)
sigma_n = np.std(x, ddof=0)  # MLE uses ddof=0
print(f"Normal MLE: mu={mu_n:.6f}, sigma={sigma_n:.6f}")

# Log-likelihood
ll_n = np.sum(stats.norm.logpdf(x, loc=mu_n, scale=sigma_n))
print(f"Log-likelihood (Normal): {ll_n:.4f}")
```

**Wording**: "Under MLE for the Normal distribution, $\hat\mu$ = {mu:.6f} and $\hat\sigma$ = {sigma:.6f}. The log-likelihood is {ll:.2f}."

### Step 3: Fit t-distribution (MLE)

```python
mu_t, sigma_t, nu = chapter5.fit_general_t(x)
ll_t = np.sum(stats.t.logpdf(x, df=nu, loc=mu_t, scale=sigma_t))
print(f"t MLE: nu={nu:.2f}, mu={mu_t:.6f}, sigma={sigma_t:.6f}")
print(f"Log-likelihood (t): {ll_t:.4f}")
```

**Wording**: "The t-distribution MLE gives $\hat\nu$ = {nu:.2f} degrees of freedom, $\hat\mu$ = {mu:.6f}, $\hat\sigma$ = {sigma:.6f}. The finite degrees of freedom confirms the data has heavier tails than Normal."

### Step 4: AICc 比较

```python
n = len(x)
# Normal: 2 params (mu, sigma)
aicc_n = chapter1.calculate_aicc(ll_n, n, 2)
# t: 3 params (nu, mu, sigma)
aicc_t = chapter1.calculate_aicc(ll_t, n, 3)
print(f"AICc Normal: {aicc_n:.2f}")
print(f"AICc t:      {aicc_t:.2f}")
print(f"Best: {'t' if aicc_t < aicc_n else 'Normal'}")
```

**或者一步到位**:
```python
fit = chapter1.fit_normal_vs_t_aicc(x)
print(f"AICc Normal={fit['aicc_norm']:.2f}, AICc t={fit['aicc_t']:.2f}, Best={fit['best_model']}")
```

**Wording**: "AICc for Normal = {aicc_n:.2f}, AICc for t = {aicc_t:.2f}. Since AICc(t) < AICc(Normal), the t-distribution is preferred — it better captures the heavy tails observed in the data (excess kurtosis = {ek:.4f} > 0) with only one additional parameter (ν)."

**如果 Normal 赢**：
"AICc(Normal) < AICc(t), so the simpler Normal model is preferred. The excess kurtosis is close to 0, suggesting tails are not significantly heavier than Normal. The penalty for the extra parameter in the t-distribution outweighs its marginal improvement in fit."

### Step 5: Normal VaR

```python
alpha = 0.05  # ← 看题目
var_n = chapter4.var_normal(mu_n, sigma_n, alpha=alpha)
print(f"Normal VaR({alpha}): {var_n:.6f}")
```

**手算验证**：`VaR = -(mu + sigma * z_alpha)`，其中 `z_0.05 = -1.645`

**Wording**: "The parametric Normal VaR at α = {alpha} is {var:.6f}. This represents the {(1-alpha)*100:.0f}% confidence level: we expect losses to exceed this amount only {alpha*100:.0f}% of the time under the Normal assumption."

### Step 6: Normal ES

```python
es_n = chapter5.calculate_es_normal(mu_n, sigma_n, alpha=alpha)
print(f"Normal ES({alpha}): {es_n:.6f}")
```

**Wording**: "The Normal ES at α = {alpha} is {es:.6f}. ES gives the expected loss GIVEN that we are in the worst {alpha*100:.0f}% tail. ES > VaR always, because ES averages over the entire tail beyond VaR."

### Step 7: t-distribution VaR

```python
var_t = -stats.t.ppf(alpha, df=nu, loc=mu_t, scale=sigma_t)
print(f"t-VaR({alpha}): {var_t:.6f}")

# 或用 eu wrapper
var_t = eu.var_t(nu, mu_t, sigma_t, alpha)
```

**Wording**: "The parametric t-VaR at α = {alpha} is {var:.6f}. This is {'larger' if var_t > var_n else 'smaller'} than Normal VaR ({var_n:.6f}) because the t-distribution has heavier tails, putting more probability mass beyond the Normal quantile."

### Step 8: t-distribution ES

```python
es_t = chapter5.calculate_es_t(nu, mu_t, sigma_t, alpha=alpha)
print(f"t-ES({alpha}): {es_t:.6f}")
```

**Wording**: "The t-ES at α = {alpha} is {es:.6f}. The gap between t-ES and Normal ES is {'even larger than the VaR gap' if (es_t - es_n) > (var_t - var_n) else 'comparable to the VaR gap'}, reflecting how the heavy-tailed t-distribution concentrates more extreme losses in the deep tail."

### Step 9: Empirical VaR / ES

```python
# Method 1: Using chapter functions
emp_var, emp_es = chapter5.calculate_es(x, alpha=alpha)
print(f"Empirical VaR: {emp_var:.6f}, ES: {emp_es:.6f}")

# Method 2: Pure numpy (if asked to show work)
q = np.quantile(x, alpha)
emp_var_manual = -q
tail = x[x <= q]
emp_es_manual = -np.mean(tail)
print(f"Manual VaR: {emp_var_manual:.6f}, ES: {emp_es_manual:.6f}")
```

**Wording**: "The empirical VaR is the negative of the {alpha*100:.0f}th percentile of the return distribution: VaR = -Q({alpha}) = {emp_var:.6f}. The empirical ES is the negative mean of all returns falling at or below this quantile: ES = -E[X | X ≤ Q({alpha})] = {emp_es:.6f}, averaging over {len(tail)} tail observations."

### Step 10: Simulation-based VaR/ES (under t)

```python
sim_res = eu.simulate_t_var_es(nu, mu_t, sigma_t, alpha=alpha, n_sim=100000, seed=42)
print(f"Parametric t-VaR:  {sim_res['parametric_var']:.6f}")
print(f"Parametric t-ES:   {sim_res['parametric_es']:.6f}")
print(f"Simulated t-VaR:   {sim_res['simulated_var']:.6f}")
print(f"Simulated t-ES:    {sim_res['simulated_es']:.6f}")
```

**Wording**: "I simulate 100,000 draws from t(ν={nu:.2f}, μ={mu:.6f}, σ={sigma:.6f}) and compute the empirical quantile and tail mean. The simulated VaR ({s_var:.6f}) and ES ({s_es:.6f}) are close to the parametric values ({p_var:.6f}, {p_es:.6f}), confirming the analytical formulas are correct. Small differences are due to Monte Carlo noise."

### Step 11: AR Model Selection

```python
best_p, ar_table = chapter2.select_best_ar_order_aicc(x, p_min=1, p_max=5)
print(f"Best AR order: {best_p}")
print(ar_table)
```

**Wording**: "Scanning AR(1) through AR(5), the best model by AICc is AR({best_p}). If AR(0) (white noise) has lower AICc than all AR models, the returns show no significant autocorrelation."

### Step 12: Jarque-Bera Normality Test

```python
jb_stat, jb_pval = stats.jarque_bera(x)
print(f"Jarque-Bera stat: {jb_stat:.4f}, p-value: {jb_pval:.6f}")
```

**Wording**: "The Jarque-Bera test statistic is {jb:.4f} with p-value = {pval:.6f}. {'Since p < 0.05, we reject the null hypothesis of Normality at the 5% level — the data has significant skewness and/or excess kurtosis.' if pval < 0.05 else 'Since p > 0.05, we cannot reject Normality at the 5% level.'}"

### Step 13: Comparison Table 输出（如果要求列表）

```python
print(f"{'Method':<15} {'VaR':>10} {'ES':>10}")
print("-" * 37)
print(f"{'Normal':<15} {var_n:>10.6f} {es_n:>10.6f}")
print(f"{'t-dist':<15} {var_t:>10.6f} {es_t:>10.6f}")
print(f"{'Empirical':<15} {emp_var:>10.6f} {emp_es:>10.6f}")
```

**Wording (对比分析)**: "Comparing the three approaches: the t-distribution gives the largest VaR and ES, reflecting its heavier tails. The Normal model underestimates tail risk. The empirical estimates lie between the two parametric methods, which is expected since empirical is model-free and captures the actual tail shape. For risk management, the t-based estimates are most conservative and appropriate given the positive excess kurtosis."

---

# Problem 2 — 多资产 Returns Matrix

## 数据特征

| 属性 | 值 |
|---|---|
| 文件 | `problem2.csv` |
| Shape | (1000, 5)，列名 `x1..x5` |
| 数据类型 | 5 asset returns matrix |
| Missing | **无** |

## 最可能的题型（细分）

| 可能的小问 | 对应 atomic 函数 |
|---|---|
| (a) Sample covariance | `np.cov(X, rowvar=False)` |
| (b) Sample correlation | `np.corrcoef(X, rowvar=False)` |
| (c) Pearson vs Spearman | `chapter2.calculate_correlations()` |
| (d) EW covariance (单 λ) | `chapter3.ew_covariance()` |
| (e) EW variance + EW corr (双 λ) | `eu.ew_cov_mixed_lambda()` |
| (f) Cov → Corr 转换 | `eu.cov_to_corr()` |
| (g) Corr + std → Cov | `eu.corr_to_cov()` |
| (h) Check PSD | `eu.is_psd()`, `eu.min_eigenvalue()` |
| (i) Eigenvalue decomposition | `np.linalg.eigvalsh()` |
| (j) Cholesky simulation | `chapter3.simulate_normal_cholesky()` |
| (k) PCA cumulative variance | `chapter3.pca_cumulative_variance()` |
| (l) Cholesky vs PCA comparison | `eu.pipeline_cholesky_vs_pca()` |
| (m) Portfolio variance | 手算 `w.T @ cov @ w` |
| (n) Demean data | `eu.demean()` |
| (o) Conditional distribution | `chapter3.conditional_bivariate_stats()` |

## 预写代码 — 一键 Pipeline

```python
# ============ PROBLEM 2 — PIPELINE VERSION ============
X, cols = eu.load_returns_matrix("problem2.csv")

# EW Covariance
lam = 0.97  # ← 看题目
res_c = eu.pipeline_ew_covariance(X, lam=lam, do_demean=True)
print(pd.DataFrame(res_c["ew_cov"], index=cols, columns=cols))

# Cholesky vs PCA
res_f = eu.pipeline_cholesky_vs_pca(res_c["ew_cov"], n_sim=10000, pct_exp=0.75, seed=42)
```

## 预写代码 — 分步 Atomic

### Step 1: Load + Demean

```python
X, cols = eu.load_returns_matrix("problem2.csv")
X_dm = eu.demean(X)
print(f"Shape: {X.shape}, Columns: {cols}")
print(f"Before demean - means: {X.mean(axis=0)}")
print(f"After demean  - means: {X_dm.mean(axis=0)}")  # should be ~0
```

**Wording**: "I demean the returns by subtracting each column's mean before estimating covariance. This ensures the EW covariance estimator is unbiased and not inflated by non-zero mean returns."

### Step 2: Sample Covariance / Correlation

```python
# Sample covariance (unbiased, ddof=1)
cov_sample = np.cov(X, rowvar=False)
print("Sample Covariance:")
print(pd.DataFrame(cov_sample, index=cols, columns=cols).round(8))

# Sample correlation
corr_sample = np.corrcoef(X, rowvar=False)
print("\nSample Correlation:")
print(pd.DataFrame(corr_sample, index=cols, columns=cols).round(4))
```

**Wording**: "The sample covariance matrix is computed with Bessel's correction (dividing by n-1). The diagonal elements are the variances of each asset. Off-diagonal element cov(xi, xj) measures the linear co-movement between assets i and j."

### Step 3: Pearson vs Spearman

```python
pearson = np.corrcoef(X, rowvar=False)
spearman, _ = stats.spearmanr(X)
print("Pearson Correlation:")
print(pd.DataFrame(pearson, index=cols, columns=cols).round(4))
print("\nSpearman Correlation:")
print(pd.DataFrame(spearman, index=cols, columns=cols).round(4))
```

**Wording (Pearson vs Spearman)**:
"Pearson correlation measures linear dependence and is sensitive to outliers. Spearman correlation is rank-based, capturing monotonic (not just linear) dependence, and is robust to heavy tails and outliers. For copula fitting, Spearman is preferred because it is invariant under monotone transformations of marginals — a key requirement of copula theory."

### Step 4: EW Covariance (单 λ)

```python
lam = 0.97  # ← 看题目
X_dm = eu.demean(X)
ew_cov = chapter3.ew_covariance(X_dm, lam=lam)
if isinstance(ew_cov, pd.DataFrame):
    ew_cov = ew_cov.to_numpy(dtype=float)
print("EW Covariance (λ={lam}):")
print(pd.DataFrame(ew_cov, index=cols, columns=cols).round(8))
```

**Wording**: "The exponentially weighted covariance with λ = {lam} assigns weight (1-λ)λ^i to observation (T-1-i), where T is the sample size. Recent observations receive more weight. The effective window ≈ 1/(1-λ) = {1/(1-lam):.0f} observations. {'λ = 0.97 → ~33 day window, a common choice (RiskMetrics).' if lam == 0.97 else ''}"

### Step 5: EW Mixed-Lambda (双 λ) — Variance 和 Correlation 分开估计

```python
lam_var = 0.97   # ← 看题目
lam_corr = 0.94  # ← 看题目

cov_mixed, ew_var_vec, ew_corr_mat = eu.ew_cov_mixed_lambda(
    X, lam_var=lam_var, lam_corr=lam_corr, do_demean=True
)
print(f"EW Variances (λ_var={lam_var}):")
for i, c in enumerate(cols):
    print(f"  {c}: variance={ew_var_vec[i]:.8f}, std={np.sqrt(ew_var_vec[i]):.6f}")

print(f"\nEW Correlation (λ_corr={lam_corr}):")
print(pd.DataFrame(ew_corr_mat, index=cols, columns=cols).round(4))

print(f"\nCombined Cov = D(sqrt(var)) @ Corr @ D(sqrt(var)):")
print(pd.DataFrame(cov_mixed, index=cols, columns=cols).round(8))
```

**Wording**: "I estimate variances with λ_var = {lam_var} (shorter memory, ~{1/(1-lam_var):.0f} days, to respond quickly to volatility changes) and correlations with λ_corr = {lam_corr} (longer memory, ~{1/(1-lam_corr):.0f} days, because correlations are more stable). The combined covariance is reconstructed as Σ = D·R·D, where D = diag(σ₁,...,σₙ) and R is the correlation matrix."

### Step 6: Cov ↔ Corr 转换

```python
# Cov → Corr
corr_from_cov = eu.cov_to_corr(ew_cov)
print("Correlation (from cov):")
print(pd.DataFrame(corr_from_cov, index=cols, columns=cols).round(4))

# Corr + std → Cov
std_vec = np.sqrt(np.diag(ew_cov))
cov_rebuilt = eu.corr_to_cov(corr_from_cov, std_vec)
print("\nRebuilt Cov (should match original):")
print(pd.DataFrame(cov_rebuilt, index=cols, columns=cols).round(8))
print(f"Max difference: {np.max(np.abs(cov_rebuilt - ew_cov)):.2e}")
```

**Wording**: "To convert covariance to correlation: ρ_{ij} = σ_{ij} / (σ_i · σ_j). To reconstruct covariance from correlation: σ_{ij} = ρ_{ij} · σ_i · σ_j. This decomposition is useful when we want to estimate variances and correlations separately (e.g., with different λ values)."

### Step 7: Check PSD + Eigenvalues

```python
eigs = np.linalg.eigvalsh(ew_cov)
print(f"Eigenvalues: {eigs}")
print(f"Min eigenvalue: {eigs.min():.8f}")
print(f"Is PSD: {eu.is_psd(ew_cov)}")
```

**Wording (如果 PSD)**: "All eigenvalues are non-negative (min = {min_eig:.8f} ≥ 0), confirming the matrix is positive semi-definite. This is expected when all n observations are used simultaneously (equal-weight or EW without missing data)."

**Wording (如果 NOT PSD)**: "The minimum eigenvalue is {min_eig:.8f} < 0, so the matrix is NOT positive semi-definite. This typically occurs with pairwise estimation from data with missing values, where different observation subsets are used for different pairs."

### Step 8: Cholesky Simulation

```python
cov = ew_cov  # or cov_mixed
nsim = 10000  # ← 看题目
seed = 42     # ← 看题目

sim = chapter3.simulate_normal_cholesky(cov, nsim=nsim, seed=seed)
print(f"Simulated data shape: {sim.shape}")

# Verify covariance recovery
cov_sim = np.cov(sim, rowvar=False)
frob = np.linalg.norm(cov_sim - cov, 'fro')
print(f"Frobenius distance (sim cov vs input cov): {frob:.6f}")
```

**Wording**: "Cholesky simulation: decompose Σ = LL^T, generate Z ~ N(0,I), then X = LZ + μ. The simulated covariance closely matches the input (Frobenius distance = {frob:.6f}), confirming the simulation is correctly calibrated."

### Step 9: PCA Cumulative Variance

```python
cum = chapter3.pca_cumulative_variance(cov)
for k, c in enumerate(cum):
    print(f"  PC {k+1}: cumulative explained = {c:.4f} ({c*100:.1f}%)")

# How many PCs for X% explanation?
pct_target = 0.75  # ← 看题目
n_pcs = np.searchsorted(cum, pct_target) + 1
print(f"\nNeed {n_pcs} PCs to explain ≥ {pct_target*100:.0f}%")
```

**Wording**: "PCA decomposes the covariance into eigenvalue-eigenvector pairs, ordered by variance explained. The first {n_pcs} principal components explain {cum[n_pcs-1]*100:.1f}% of total variance. Using PCA simulation with {n_pcs} components reduces dimensionality from {len(cum)} to {n_pcs}, trading a small loss in covariance recovery for computational efficiency."

### Step 10: Cholesky vs PCA Comparison

```python
res_f = eu.pipeline_cholesky_vs_pca(cov, n_sim=10000, pct_exp=0.75, seed=42)
print(f"Cholesky: time={res_f['time_cholesky']:.4f}s, Frobenius={res_f['frobenius_cholesky']:.6f}")
print(f"PCA:      time={res_f['time_pca']:.4f}s, Frobenius={res_f['frobenius_pca']:.6f}")
```

**Wording**: "Cholesky simulation exactly preserves the covariance structure (Frobenius ≈ sampling noise only), while PCA(k) uses only the top k eigenvectors, introducing a deliberate approximation error. Cholesky Frobenius = {frob_c:.6f} vs PCA Frobenius = {frob_p:.6f}. PCA is faster when n is large and a few components capture most variance. For small n (like 5), Cholesky is preferred as it's already fast and exact."

### Step 11: Portfolio Variance（手算）

```python
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # ← 看题目（等权或给定）
port_var = weights @ cov @ weights
port_std = np.sqrt(port_var)
print(f"Portfolio variance: {port_var:.8f}")
print(f"Portfolio std:      {port_std:.6f}")
```

**Wording**: "Portfolio variance σ²_p = w^T Σ w = {port_var:.8f}. The portfolio standard deviation is {port_std:.6f}, which is less than the weighted average of individual standard deviations due to diversification (correlations < 1)."

### Step 12: Conditional Bivariate Distribution

```python
# If asked: given x1 = some value, what is E[x2] and Var(x2)?
# 需要从 5×5 cov 中提取 2×2 subblock
i, j = 0, 1  # ← 看题目说的是哪两个变量
mu_i, mu_j = np.mean(X[:, i]), np.mean(X[:, j])
var_i, var_j = cov[i, i], cov[j, j]
cov_ij = cov[i, j]
rho = cov_ij / np.sqrt(var_i * var_j)

x_given = 0.02  # ← 题目给的条件值

# Conditional E[Xj | Xi = x_given]
cond_mean = mu_j + rho * np.sqrt(var_j / var_i) * (x_given - mu_i)
# Conditional Var(Xj | Xi = x_given)
cond_var = var_j * (1 - rho**2)

print(f"Given {cols[i]} = {x_given}:")
print(f"  E[{cols[j]} | {cols[i]}={x_given}] = {cond_mean:.6f}")
print(f"  Var({cols[j]} | {cols[i]}={x_given}) = {cond_var:.8f}")
print(f"  Std({cols[j]} | {cols[i]}={x_given}) = {np.sqrt(cond_var):.6f}")
```

**Wording**: "Under bivariate normality, E[X₂|X₁=x] = μ₂ + ρ(σ₂/σ₁)(x - μ₁) = {cond_mean:.6f}. The conditional variance Var(X₂|X₁=x) = σ₂²(1 - ρ²) = {cond_var:.8f} does NOT depend on the conditioning value x — it only depends on the correlation ρ = {rho:.4f}. Higher correlation → more variance reduction from conditioning."

---

# Problem 3 — 5×5 Covariance Matrix (NOT PSD!)

## 数据特征

| 属性 | 值 |
|---|---|
| 文件 | `problem3.csv` |
| Shape | (5, 5)，symmetric |
| 数据类型 | **Covariance matrix**（对角线 ~0.0001–0.0004 = variance of returns） |
| Symmetric | ✓ |
| Eigenvalues | **[-4.56e-05**, 1.24e-04, 1.91e-04, 4.27e-04, 8.65e-04] |
| **PSD** | **✗ — 有一个负 eigenvalue！** |

## 最可能的题型（细分）

| 可能的小问 | 对应 atomic 函数 |
|---|---|
| (a) 验证是否 PSD | `eu.is_psd()` + `eu.min_eigenvalue()` |
| (b) 解释为什么不是 PSD | wording（pairwise + missing） |
| (c) near_psd 修复 | `chapter3.near_psd()` |
| (d) Higham 修复 | `chapter3.higham_nearest_psd()` |
| (e) 比较两种修复（Frobenius） | `np.linalg.norm(..., 'fro')` |
| (f) PCA cumulative variance | `chapter3.pca_cumulative_variance()` |
| (g) 从修复矩阵 simulate | `chapter3.simulate_normal_cholesky()` |
| (h) Cov → Corr | `eu.cov_to_corr()` |
| (i) Eigenvalue 分解 | `np.linalg.eigvalsh()` |
| (j) Cholesky vs PCA on fixed | `eu.pipeline_cholesky_vs_pca()` |

## 预写代码 — 一键 Pipeline

```python
# ============ PROBLEM 3 — PIPELINE VERSION ============
p3 = pd.read_csv("problem3.csv")
cov_raw = p3.values

# If it came from returns with missing data:
# res_d = eu.pipeline_missing_psd_pca(X_with_nan, fix_method="higham")

# Direct PSD fix + PCA:
cov_hig = chapter3.higham_nearest_psd(cov_raw)
res_f = eu.pipeline_cholesky_vs_pca(cov_hig, n_sim=10000, pct_exp=0.75, seed=42)
```

## 预写代码 — 分步 Atomic

### Step 1: Load + 验证 PSD

```python
p3 = pd.read_csv("problem3.csv")
cov_raw = p3.values
print(f"Shape: {cov_raw.shape}")
print(f"Symmetric: {np.allclose(cov_raw, cov_raw.T)}")

eigs = np.linalg.eigvalsh(cov_raw)
print(f"Eigenvalues: {eigs}")
print(f"Min eigenvalue: {eigs.min():.8e}")
print(f"Is PSD: {eu.is_psd(cov_raw)}")
```

**Wording**: "The matrix has eigenvalues {eigs}. Since the minimum eigenvalue is {eigs.min():.4e} < 0, the matrix is NOT positive semi-definite. A valid covariance matrix must have all eigenvalues ≥ 0."

### Step 2: 解释为什么不是 PSD

**Wording 选项 A (pairwise estimation)**:
"This matrix is not PSD because it was likely estimated using pairwise covariance with missing data. When different pairs use different observation subsets, the resulting matrix can violate the PSD requirement. This is a well-known issue in finance when assets have different trading histories or data availability."

**Wording 选项 B (mixed frequency)**:
"The matrix may have been constructed from correlation/covariance estimates from different time periods or different data sources. Combining inconsistent estimates can produce a non-PSD matrix."

**Wording 选项 C (numerical)**:
"A covariance matrix Σ is PSD if and only if x^T Σ x ≥ 0 for all x. Equivalently, all eigenvalues must be ≥ 0. When the minimum eigenvalue is negative (even slightly, like {eigs.min():.4e}), there exists a portfolio direction that produces negative variance — a mathematical impossibility."

### Step 3: near_psd 修复 (Rebonato-Jäckel)

```python
cov_near = chapter3.near_psd(cov_raw)
eigs_near = np.linalg.eigvalsh(cov_near)
frob_near = np.linalg.norm(cov_near - cov_raw, 'fro')

print(f"near_psd eigenvalues: {eigs_near}")
print(f"Min eigenvalue: {eigs_near.min():.8e}")
print(f"Frobenius distance to original: {frob_near:.8f}")
```

**Wording**: "The Rebonato-Jäckel near_psd method projects the correlation matrix onto PSD by setting negative eigenvalues to zero (or ε) and rescaling to unit diagonal. Frobenius distance = {frob_near:.8f}."

### Step 4: Higham 修复

```python
cov_hig = chapter3.higham_nearest_psd(cov_raw)
cov_hig = 0.5 * (cov_hig + cov_hig.T)  # enforce exact symmetry
eigs_hig = np.linalg.eigvalsh(cov_hig)
frob_hig = np.linalg.norm(cov_hig - cov_raw, 'fro')

print(f"Higham eigenvalues: {eigs_hig}")
print(f"Min eigenvalue: {eigs_hig.min():.8e}")
print(f"Frobenius distance to original: {frob_hig:.8f}")
```

**Wording**: "Higham's alternating projection method iteratively projects between the set of PSD matrices and the set of matrices with unit diagonal (for correlation). It is provably optimal — it finds the nearest PSD matrix in Frobenius norm. Frobenius distance = {frob_hig:.8f}."

### Step 5: 比较两种修复

```python
print(f"Frobenius distance comparison:")
print(f"  near_psd: {frob_near:.8f}")
print(f"  Higham:   {frob_hig:.8f}")
print(f"  Higham is closer: {frob_hig <= frob_near}")
print(f"  Improvement: {(frob_near - frob_hig) / frob_near * 100:.2f}%")
```

**Wording**: "Higham's method (Frobenius = {frob_hig:.8f}) produces a matrix strictly closer to the original than Rebonato-Jäckel (Frobenius = {frob_near:.8f}). This is guaranteed by Higham's theorem: alternating projection converges to the globally nearest PSD matrix. In practice, the difference is {(frob_near-frob_hig)/frob_near*100:.2f}%, which {'is small since the original matrix was only slightly non-PSD' if abs(eigs.min()) < 1e-3 else 'is significant'}."

### Step 6: Cov → Corr

```python
corr_raw = eu.cov_to_corr(cov_raw)
print("Correlation from raw covariance:")
print(pd.DataFrame(corr_raw).round(4))

corr_fixed = eu.cov_to_corr(cov_hig)
print("\nCorrelation from Higham-fixed covariance:")
print(pd.DataFrame(corr_fixed).round(4))
```

### Step 7: PCA Cumulative Variance（on fixed matrix）

```python
cum = chapter3.pca_cumulative_variance(cov_hig)
print("PCA cumulative explained variance:")
for k, c in enumerate(cum):
    print(f"  PC {k+1}: {c:.4f} ({c*100:.1f}%)")

pct_target = 0.75  # ← 看题目
n_pcs = np.searchsorted(cum, pct_target) + 1
print(f"\nNeed {n_pcs} PCs for ≥ {pct_target*100:.0f}%")
```

**Wording**: "After PSD correction, PCA shows that the first {n_pcs} principal components explain {cum[n_pcs-1]*100:.1f}% of the total variance. This suggests {'a strong dominant factor (likely market risk)' if cum[0] > 0.5 else 'variance is spread more evenly across components'}."

### Step 8: Simulate from Fixed Matrix

```python
nsim = 10000  # ← 看题目
sim = chapter3.simulate_normal_cholesky(cov_hig, nsim=nsim, seed=42)
print(f"Simulation shape: {sim.shape}")

# Verify
cov_sim = np.cov(sim, rowvar=False)
frob_recovery = np.linalg.norm(cov_sim - cov_hig, 'fro')
print(f"Frobenius (sim cov vs fixed cov): {frob_recovery:.6f}")
```

**也可以用 Higham wrapper**：
```python
sim = eu.simulate_normal_higham(cov_raw, nsim=10000, seed=42)
```

**Wording**: "I simulate {nsim} draws from N(0, Σ_fixed) using Cholesky decomposition of the Higham-corrected covariance. The simulated covariance recovers the fixed matrix with Frobenius error = {frob:.6f} (due to Monte Carlo sampling noise)."

### Step 9: Cholesky vs PCA on Fixed Matrix

```python
pct = 0.75  # ← 看题目
res = eu.pipeline_cholesky_vs_pca(cov_hig, n_sim=10000, pct_exp=pct, seed=42)
print(f"Cholesky — Frobenius: {res['frobenius_cholesky']:.6f}, Time: {res['time_cholesky']:.4f}s")
print(f"PCA({pct*100:.0f}%) — Frobenius: {res['frobenius_pca']:.6f}, Time: {res['time_pca']:.4f}s")
```

**Wording**: "Comparing simulation methods on the fixed matrix: Cholesky exactly replicates the covariance (Frobenius = {f_c:.6f}, sampling noise only), while PCA at {pct*100:.0f}% explained variance uses fewer dimensions but introduces approximation error (Frobenius = {f_p:.6f}). For a 5×5 matrix, Cholesky is preferred since dimensionality is already low."

---

# Problem 4 — Multi-Asset Prices (SPY, AAPL, MSFT, GOOGL, BABA)

## 数据特征

| 属性 | 值 |
|---|---|
| 文件 | `problem4.csv` |
| Shape | (1037, 6)，列名 `Date, SPY, AAPL, MSFT, GOOGL, BABA` |
| 数据类型 | **Daily prices**（真实股票，2022-01-03 ~ 2026-02-20） |
| Missing | 无 |
| Assets | 5 stocks：SPY, AAPL, MSFT, GOOGL, BABA |

## 最可能的题型（细分）

| 可能的小问 | 对应 atomic 函数 |
|---|---|
| (a) 计算 returns (discrete or log) | `chapter4.return_calculate()` |
| (b) Per-asset moments | `chapter1.first4_moments()` per column |
| (c) Per-asset Normal vs t fit | `chapter1.fit_normal_vs_t_aicc()` per column |
| (d) Spearman correlation | `stats.spearmanr()` |
| (e) Pearson correlation | `np.corrcoef()` |
| (f) Fit copula marginals | `chapter5.fit_copula_marginals()` |
| (g) Fit copula correlation | `chapter5.fit_gaussian_copula_corr()` |
| (h) Simulate from copula | `chapter5.simulate_copula_from_fitted()` |
| (i) Dollar PnL calculation | `eu.portfolio_pnl()` |
| (j) Per-asset VaR/ES | `eu.var_es_from_pnl()` per column |
| (k) Portfolio VaR/ES | `eu.var_es_from_pnl()` on sum |
| (l) Multi-alpha VaR/ES | 对多个 alpha 循环 |
| (m) Compare Spearman vs Pearson copula | 两次 simulate, 比较 |
| (n) Compare discrete vs log returns | 两次 return_calculate |
| (o) EW covariance on returns | `chapter3.ew_covariance()` |
| (p) VaR diversification benefit | sum of individual VaR vs portfolio VaR |
| (q) Marginal VaR contribution | numerical |

## 预写代码 — 一键 Pipeline

```python
# ============ PROBLEM 4 — PIPELINE VERSION ============
holdings = np.array([100, 200, 150, 100, 50])  # ← 看题目!!!
alpha = 0.05  # ← 看题目

# Single alpha
res = eu.pipeline_copula_portfolio_from_prices(
    "problem4.csv", holdings=holdings, dist_types=["t"]*5,
    alpha=alpha, n_sim=10000, method="spearman", seed=42,
    return_method="discrete", do_demean=True
)
eu.print_portfolio_risk(res)

# Multi-alpha
res2 = eu.pipeline_copula_multi_alpha_from_prices(
    "problem4.csv", holdings=holdings, alphas=[0.05, 0.01],
    dist_types=["t"]*5, n_sim=10000, method="spearman", seed=42
)
eu.print_multi_alpha_risk(res2)
```

## 预写代码 — 分步 Atomic

### Step 1: Load Prices

```python
prices_df = eu.load_prices("problem4.csv", date_col="Date")
print(f"Shape: {prices_df.shape}")
print(f"Date range: {prices_df['Date'].iloc[0]} to {prices_df['Date'].iloc[-1]}")
print(f"Last prices:")
for col in prices_df.columns[1:]:
    print(f"  {col}: ${prices_df[col].iloc[-1]:.2f}")
```

### Step 2: Compute Returns

```python
# Discrete returns: r_t = (P_t - P_{t-1}) / P_{t-1}
rets_df = chapter4.return_calculate(prices_df, date_column="Date", method="discrete")
R = rets_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
asset_names = rets_df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Returns shape: {R.shape}")
print(f"Assets: {asset_names}")
```

**如果题目问 log returns**:
```python
rets_log = chapter4.return_calculate(prices_df, date_column="Date", method="log")
R_log = rets_log.select_dtypes(include=[np.number]).to_numpy(dtype=float)
```

**Wording (discrete vs log)**: "Discrete returns r_t = (P_t/P_{t-1}) - 1 are exact for portfolio aggregation: portfolio return = Σ w_i · r_i. Log returns r_t = ln(P_t/P_{t-1}) are exact for time aggregation: multi-period return = Σ r_t. For daily risk management with short horizons, the difference is negligible. I use discrete returns for this portfolio VaR/ES calculation."

### Step 3: Per-Asset Moments

```python
print(f"{'Asset':<8} {'Mean':>10} {'Std':>10} {'Skew':>8} {'ExKurt':>8}")
print("-" * 50)
for j, name in enumerate(asset_names):
    m, v, s, ek = chapter1.first4_moments(R[:, j])
    print(f"{name:<8} {m:>10.6f} {np.sqrt(v):>10.6f} {s:>8.4f} {ek:>8.4f}")
```

**Wording**: "All five assets show positive excess kurtosis (heavier tails than Normal), supporting the use of t-distribution marginals. BABA shows the highest volatility and most negative skewness, reflecting emerging market risk."

### Step 4: Per-Asset Distribution Fit (Normal vs t)

```python
print("\n=== Distribution Fitting ===")
marginal_params = {}
for j, name in enumerate(asset_names):
    fit = chapter1.fit_normal_vs_t_aicc(R[:, j])
    best = fit["best_model"]
    print(f"{name}: best={best}, AICc_N={fit['aicc_norm']:.2f}, AICc_t={fit['aicc_t']:.2f}", end="")
    if best == "t":
        print(f", nu={fit['nu_t']:.2f}")
    else:
        print()
    marginal_params[name] = fit
```

**Wording**: "For each asset, I compare Normal vs t-distribution via AICc. {'All' if all assets are t else 'Most'} assets prefer the t-distribution, consistent with the well-known fact that financial returns exhibit fat tails. The degrees of freedom ν range from {min_nu:.1f} to {max_nu:.1f} — lower ν means heavier tails."

### Step 5: Fit Copula Marginals（底层函数）

```python
R_dm = eu.demean(R)
dist_types = ["t", "t", "t", "t", "t"]  # ← 看题目
u_data, marginals = chapter5.fit_copula_marginals(R_dm, dist_types=dist_types)
print("Fitted marginals:")
for m in marginals:
    print(f"  {m}")
```

**Wording**: "I fit each marginal separately via MLE. For t-distributed marginals, I estimate (ν, μ, σ) per asset. The copula framework allows different marginal distributions for different assets — we separate the modeling of individual behavior (marginals) from dependence structure (copula)."

### Step 6: Fit Copula Correlation

```python
# Spearman (recommended)
cop_corr_sp = chapter5.fit_gaussian_copula_corr(R_dm, method="spearman")
print("Spearman copula correlation:")
print(pd.DataFrame(cop_corr_sp, index=asset_names, columns=asset_names).round(4))

# Pearson (for comparison if asked)
cop_corr_pe = chapter5.fit_gaussian_copula_corr(R_dm, method="pearson")
print("\nPearson copula correlation:")
print(pd.DataFrame(cop_corr_pe, index=asset_names, columns=asset_names).round(4))
```

**Wording (Spearman vs Pearson for copula)**:
"For the Gaussian copula, Spearman correlation is preferred because: (1) it is rank-based, making it invariant to monotone transformations of the marginals — a fundamental property of copulas; (2) it is robust to outliers and heavy tails; (3) it captures the copula dependence directly without being distorted by marginal shapes. Pearson correlation, by contrast, is affected by the marginal distributions and can be biased when tails are heavy."

### Step 7: Simulate from Copula

```python
n_sim = 10000  # ← 看题目
seed = 42      # ← 看题目

sim_R = chapter5.simulate_copula_from_fitted(
    marginals=marginals,
    corr=cop_corr_sp,
    n_sim=n_sim,
    seed=seed
)
print(f"Simulated returns shape: {sim_R.shape}")
```

**或者一步到位**：
```python
sim_R, cop_corr, marginals = chapter5.simulate_copula_mixed(
    data=R_dm, dist_types=dist_types, n_sim=n_sim, method="spearman", seed=seed
)
```

**Wording**: "I simulate {n_sim} joint return scenarios from the fitted Gaussian copula. The process: (1) generate correlated uniform variables via the copula, (2) transform each uniform through the corresponding marginal inverse CDF. This preserves both the individual return distributions and their dependence structure."

### Step 8: Dollar PnL Calculation

```python
holdings = np.array([100, 200, 150, 100, 50])  # ← 看题目!!!
last_prices = prices_df.select_dtypes(include=[np.number]).iloc[-1].to_numpy(dtype=float)

# Portfolio value
port_value = np.sum(holdings * last_prices)
print(f"Current portfolio value: ${port_value:,.2f}")

# Per-asset dollar positions
for j, name in enumerate(asset_names):
    pos = holdings[j] * last_prices[j]
    print(f"  {name}: {holdings[j]} shares × ${last_prices[j]:.2f} = ${pos:,.2f} ({pos/port_value*100:.1f}%)")

# PnL matrix
a_pnl = eu.asset_pnl_matrix(sim_R, holdings, last_prices)  # (n_sim, 5)
p_pnl = a_pnl.sum(axis=1)  # (n_sim,)
print(f"\nPnL statistics:")
print(f"  Mean: ${p_pnl.mean():,.2f}")
print(f"  Std:  ${p_pnl.std():,.2f}")
print(f"  Min:  ${p_pnl.min():,.2f}")
print(f"  Max:  ${p_pnl.max():,.2f}")
```

**Wording**: "Dollar PnL for each asset = return × shares × current price. Portfolio PnL is the sum across assets. The portfolio value is ${port_value:,.2f}, with the largest position in {largest_asset} ({largest_pct:.1f}%)."

### Step 9: Portfolio VaR / ES

```python
alpha = 0.05  # ← 看题目
port_var, port_es = eu.var_es_from_pnl(p_pnl, alpha=alpha)
print(f"Portfolio VaR({alpha}): ${port_var:,.2f}")
print(f"Portfolio ES({alpha}):  ${port_es:,.2f}")
```

**Wording**: "At α = {alpha} ({(1-alpha)*100:.0f}% confidence), the portfolio VaR is ${var:,.2f} — we expect the 1-day loss to exceed this amount only {alpha*100:.0f}% of the time. The ES is ${es:,.2f} — the expected loss given we are in the worst {alpha*100:.0f}% tail. ES > VaR by definition, with the gap reflecting the severity of extreme losses."

### Step 10: Per-Asset VaR / ES

```python
print(f"\n{'Asset':<8} {'VaR':>12} {'ES':>12}")
print("-" * 34)
sum_var = 0
for j, name in enumerate(asset_names):
    vj, ej = eu.var_es_from_pnl(a_pnl[:, j], alpha=alpha)
    print(f"{name:<8} ${vj:>10,.2f} ${ej:>10,.2f}")
    sum_var += vj
print("-" * 34)
print(f"{'Sum':<8} ${sum_var:>10,.2f}")
print(f"{'Portfolio':<8} ${port_var:>10,.2f}")
print(f"{'Diversification benefit':<8}: ${sum_var - port_var:,.2f} ({(sum_var - port_var)/sum_var*100:.1f}%)")
```

**Wording (diversification benefit)**: "The sum of individual VaRs (${sum_var:,.2f}) exceeds the portfolio VaR (${port_var:,.2f}) by ${sum_var-port_var:,.2f}. This {(sum_var-port_var)/sum_var*100:.1f}% diversification benefit arises because assets are imperfectly correlated — losses don't all occur simultaneously. This is a key motivation for portfolio diversification. Note: VaR is NOT sub-additive in general (it can fail), but ES IS always sub-additive (coherent risk measure)."

### Step 11: Multi-Alpha Comparison

```python
alphas = [0.05, 0.01]  # ← 看题目
for a in alphas:
    v, e = eu.var_es_from_pnl(p_pnl, alpha=a)
    print(f"Alpha={a} ({(1-a)*100:.0f}%): VaR=${v:,.2f}, ES=${e:,.2f}")
```

**Wording**: "At 95% confidence (α=0.05), VaR = ${v05:,.2f} and ES = ${e05:,.2f}. At 99% confidence (α=0.01), VaR = ${v01:,.2f} and ES = ${e01:,.2f}. The 99% measures are significantly larger, reflecting the non-linear increase in tail risk. The ratio ES/VaR is {'larger at 99% than 95%, indicating the extreme tail is particularly fat' if ratio_01 > ratio_05 else 'similar at both levels'}."

### Step 12: Compare Spearman vs Pearson Copula

```python
# Spearman
sim_sp, _, _ = chapter5.simulate_copula_mixed(
    R_dm, dist_types=dist_types, n_sim=n_sim, method="spearman", seed=seed
)
pnl_sp = eu.portfolio_pnl(sim_sp, holdings, last_prices)
var_sp, es_sp = eu.var_es_from_pnl(pnl_sp, alpha=alpha)

# Pearson
sim_pe, _, _ = chapter5.simulate_copula_mixed(
    R_dm, dist_types=dist_types, n_sim=n_sim, method="pearson", seed=seed
)
pnl_pe = eu.portfolio_pnl(sim_pe, holdings, last_prices)
var_pe, es_pe = eu.var_es_from_pnl(pnl_pe, alpha=alpha)

print(f"Spearman: VaR=${var_sp:,.2f}, ES=${es_sp:,.2f}")
print(f"Pearson:  VaR=${var_pe:,.2f}, ES=${es_pe:,.2f}")
```

**Wording**: "Using Spearman correlation for the copula gives VaR = ${var_sp:,.2f} vs Pearson's ${var_pe:,.2f}. {'The difference is modest, suggesting the dependence structure is approximately linear.' if abs(var_sp-var_pe)/var_sp < 0.05 else 'The noticeable difference reflects the sensitivity of tail risk to how we measure dependence. Spearman is more robust and preferred for copula fitting.'}"

---

# General Wording Bank — 通用模板

## Concept Explanations（如果问概念题）

### VaR vs ES
"VaR is the α-quantile of the loss distribution — the minimum loss in the worst α% of scenarios. ES is the expected loss given we are in that worst α% tail: ES = E[Loss | Loss > VaR]. ES is always ≥ VaR and is a coherent risk measure (satisfying sub-additivity), while VaR is not."

### Coherent Risk Measures (4 Axioms)
"A risk measure ρ is coherent if it satisfies: (1) Monotonicity: if X ≤ Y always, then ρ(X) ≥ ρ(Y); (2) Sub-additivity: ρ(X+Y) ≤ ρ(X) + ρ(Y) — diversification never increases risk; (3) Positive homogeneity: ρ(λX) = λρ(X) for λ > 0; (4) Translation invariance: ρ(X + c) = ρ(X) - c. ES satisfies all four; VaR fails sub-additivity."

### Why VaR Fails Sub-Additivity
"Consider two zero-mean Bernoulli losses with P(loss=100) = 2%. Individual VaR(5%) = 0 for each (since 95th percentile shows no loss). But the combined portfolio can have P(loss=100) = ~4%, giving VaR(5%) > 0. Thus VaR(A+B) > VaR(A) + VaR(B), violating sub-additivity."

### Copula Theory
"By Sklar's theorem, any joint distribution F(x₁,...,xₙ) can be decomposed as F(x₁,...,xₙ) = C(F₁(x₁),...,Fₙ(xₙ)), where C is the copula and Fᵢ are marginal CDFs. The copula captures the pure dependence structure, separate from marginal behavior. The Gaussian copula uses a multivariate Normal to model dependence between the uniform transforms of marginals."

### EW Covariance vs Equal-Weight
"Equal-weight covariance treats all observations equally: σ̂ = (1/(n-1))Σ(xᵢ-x̄)². EW covariance assigns exponentially decaying weights: w_i = (1-λ)λⁱ, giving more influence to recent data. λ controls the decay rate — higher λ means slower decay (longer memory). RiskMetrics recommends λ=0.94 for daily data."

### Cholesky vs PCA Simulation
"Both methods generate correlated random variables from a covariance matrix. Cholesky: Σ = LL^T, X = LZ where Z ~ N(0,I) — exact but uses all dimensions. PCA: Σ = VΛV^T, keep top k eigenvalues, X ≈ V_k √Λ_k Z_k — approximate but lower dimension, useful when a few factors dominate."

### PSD Requirement
"A covariance matrix must be PSD so that portfolio variance w^T Σ w ≥ 0 for all weight vectors w. Without PSD, some portfolios would have negative variance — mathematically impossible. PSD also ensures Cholesky decomposition exists (needed for simulation)."

### Rebonato-Jäckel vs Higham
"Both methods 'fix' a non-PSD matrix to be PSD. Rebonato-Jäckel: sets negative eigenvalues to zero, rescales to preserve diagonal. Higham: alternating projections between PSD cone and unit-diagonal set, provably finds the nearest PSD matrix in Frobenius norm. Higham is always at least as good as Rebonato-Jäckel."

### AICc Formula
"AIC = 2k - 2ln(L), where k = number of parameters and L = maximum likelihood. AICc adds a finite-sample correction: AICc = AIC + 2k(k+1)/(n-k-1). Lower AICc is better. The correction penalizes overparameterization more strongly when n is small relative to k."

### Demean: When and Why
"Demeaning (subtracting column means) before covariance estimation is important when: (1) the EW covariance formula doesn't internally subtract the mean; (2) the mean is non-zero and would inflate the covariance estimate. For returns data with mean ≈ 0, the effect is small but technically correct."

## Error Handling Wording（如果结果看起来 weird）

### "Why is t-ES so much larger than Normal ES?"
"The t-distribution has polynomial tails (decaying as |x|^{-(ν+1)}) vs Normal's exponential tails (decaying as e^{-x²/2}). This means the t-distribution puts significantly more probability mass in the extreme tail. At the same VaR level, the expected shortfall under t is larger because the conditional tail average is pulled further by these extreme values."

### "Why does simulation VaR differ from parametric?"
"Monte Carlo VaR is a random variable — each simulation gives a slightly different quantile. The standard error of the simulated α-quantile ≈ √(α(1-α)/n) / f(q_α), where f is the PDF at the quantile. With 10,000 simulations, expect ~1-3% variation. More simulations → closer to parametric."

### "Why are Normal and t VaR almost the same but ES differs a lot?"
"VaR is a single quantile. For moderate α (like 5%), the Normal and t quantiles may be similar because the t's heavier tails are most prominent in the extreme. ES, however, averages over the entire tail beyond VaR — this is where the t-distribution's heavy tails contribute most. So ES amplifies the tail difference more than VaR does."

### "Why is Higham sometimes barely better than near_psd?"
"When the matrix is only slightly non-PSD (min eigenvalue ≈ 0), both methods produce similar corrections because the adjustment needed is small. The provable optimality of Higham matters more when the violation is severe."

---

# Quick Tips — 考试时的操作顺序

```
1. 读题 → 确认参数（alpha, holdings, dist_types, lambda, n_sim, seed 等）
2. 判断题目是 pipeline 还是 atomic：
   - 如果问完整流程（从数据到 VaR/ES）→ 用 pipeline
   - 如果分小步（"先算 moments"、"再 fit"、"然后 VaR"）→ 用 atomic
3. 复制对应代码段
4. 修改参数 → 运行
5. 从 output 中提取数字
6. 复制 wording 模板 → 替换数字 → 调整细节
```

## 参数 Checklist（考试时逐项确认）

- [ ] `alpha` = ? (0.05? 0.01? 两个都要?)
- [ ] `holdings` = ? (每个 asset 多少 shares?)
- [ ] `dist_types` = ? (all t? mixed? all Normal?)
- [ ] `lambda` = ? (0.97? 0.94? mixed? single?)
- [ ] `method` = ? (spearman? pearson?)
- [ ] `return_method` = ? (discrete? log?)
- [ ] `do_demean` = ? (True? False? 题目说了吗?)
- [ ] `n_sim` = ? (10000? 100000? 题目有规定?)
- [ ] `seed` = ? (42? 题目有规定?)
- [ ] `fix_method` = ? (higham? near_psd? both? compare?)
- [ ] `pct_exp` = ? (0.75? 0.90? for PCA)
- [ ] 输出要求? (table? single number? wording? plot?)

## 紧急 Debug 速查

| 症状 | 原因 | 解法 |
|---|---|---|
| `LinAlgError: Matrix not positive definite` | cov 不是 PSD | 用 `chapter3.higham_nearest_psd()` 修复后再 Cholesky |
| VaR 是负数 | 可能 alpha 用错了方向 | 确认 `VaR = -quantile(x, alpha)` |
| ES < VaR | 代码 bug | 确认 ES 是 tail 的 mean（条件在 ≤ quantile） |
| Copula simulation 结果不稳定 | n_sim 太小 | 增加到 100000 |
| `ModuleNotFoundError` | path 没设好 | 确认 `sys.path.insert(0, "..")` |
| AICc 是 inf 或 nan | n ≤ k+1 导致分母为 0 | 检查样本量和参数数量 |
| Mixed-lambda cov 不是 PSD | 理论上可能 | 用 Higham 修复 |
