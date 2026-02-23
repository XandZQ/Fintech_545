# Midterm Data Recon — 考前数据侦察 + 预写代码

> **目的**：我们有明天考试的 4 个 input CSV，但没有题目。通过分析数据结构，**预测可能的题型** 并 **预写代码骨架**，考试时只需填入具体参数。

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

> 单列 returns → 这几乎一定是 **Recipe A + B** 的组合题：
> 1. 计算 4 moments
> 2. Fit Normal vs t，用 AICc 选择
> 3. 计算 VaR / ES（Normal, t, empirical）
> 4. 可能加 AR fitting 或 hypothesis test

数据特征支持：excess kurtosis = 0.69 > 0 → **t 应该赢**；skewness = -0.24 → 负偏，说明左尾风险。

## 预写代码

```python
# ============ PROBLEM 1 ============
import numpy as np, pandas as pd, sys
sys.path.insert(0, "..")
from qrm_lib import chapter1, chapter2, chapter3, chapter4, chapter5
from qrm_lib import exam_utils as eu

# --- Load ---
x = eu.load_returns_vector("midterm/problem1.csv")

# --- (a) Moments ---
res_a = eu.pipeline_moments_and_fit(x)
eu.print_moments(res_a)

# --- (b) VaR / ES at alpha = 0.05 (change if needed) ---
alpha = 0.05  # ← 看题目
res_b = eu.pipeline_var_es_univariate(x, alpha=alpha)
eu.print_var_es(res_b)

# --- (c) If asked for t VaR/ES only ---
nu = res_a["fit"]["nu_t"]
mu = res_a["fit"]["mu_t"]
sigma = res_a["fit"]["sigma_t"]
print(f"\nt-VaR (closed form): {eu.var_t(nu, mu, sigma, alpha):.6f}")

# --- (d) If asked for simulation-based ---
sim_res = eu.simulate_t_var_es(nu, mu, sigma, alpha=alpha, n_sim=100000, seed=42)
print(f"Sim t-VaR: {sim_res['var']:.6f}, Sim t-ES: {sim_res['es']:.6f}")

# --- (e) If asked for AR model ---
best_p, ar_table = chapter2.select_best_ar_order_aicc(x, p_min=1, p_max=5)
print(f"\nBest AR order: {best_p}")
print(ar_table)
```

### Wording 备用

> "The sample has negative skewness (-0.24) and positive excess kurtosis (0.69), indicating left-tail risk and heavier tails than Normal. The t-distribution has lower AICc, confirming better fit. Under the t model, ES is notably larger than Normal ES because the t-distribution puts more probability mass in the extreme left tail."

---

# Problem 2 — 多资产 Returns Matrix

## 数据特征

| 属性 | 值 |
|---|---|
| 文件 | `problem2.csv` |
| Shape | (1000, 5)，列名 `x1..x5` |
| 数据类型 | 5 asset returns matrix |
| Missing | **无** |

## 最可能的题型

> 5 列无缺失 returns → 两种方向：
>
> **方向 A — Covariance + Simulation（Recipe C + F）**
> 1. 计算 EW covariance（给 λ）
> 2. Check PSD
> 3. Cholesky vs PCA simulation comparison
>
> **方向 B — Mixed-lambda EW（Recipe C + Bonus Cov/Corr）**
> 1. EW variance with λ_var
> 2. EW correlation with λ_corr
> 3. Combine → covariance matrix
>
> **方向 C — Copula（Recipe E）**
> 1. Fit marginals per column
> 2. Copula simulation → portfolio VaR/ES
> （但 Problem 4 更像 copula 题，因为它有 prices）

## 预写代码

```python
# ============ PROBLEM 2 ============
X, cols = eu.load_returns_matrix("midterm/problem2.csv")

# --- (a) EW Covariance ---
lam = 0.97  # ← 看题目
res_c = eu.pipeline_ew_covariance(X, lam=lam, do_demean=True)
print("EW Covariance:")
print(pd.DataFrame(res_c["ew_cov"], index=cols, columns=cols))
print(f"\nEW Correlation:")
print(pd.DataFrame(res_c["ew_corr"], index=cols, columns=cols))
print(f"PSD: {res_c['is_psd']}, Min eigenvalue: {res_c['min_eig']:.8f}")

# --- (b) Cov → Corr ---
corr = eu.cov_to_corr(res_c["ew_cov"])
print("\nCorrelation from EW Cov:")
print(pd.DataFrame(corr, index=cols, columns=cols))

# --- (c) Mixed-lambda EW ---
lam_var = 0.97   # ← 看题目
lam_corr = 0.94  # ← 看题目
cov_mixed = eu.ew_cov_mixed_lambda(X, lam_var=lam_var, lam_corr=lam_corr, do_demean=True)
print("\nMixed-lambda EW Cov:")
print(pd.DataFrame(cov_mixed, index=cols, columns=cols))

# --- (d) Cholesky vs PCA ---
cov = res_c["ew_cov"]  # or cov_mixed
res_f = eu.pipeline_cholesky_vs_pca(cov, n_sim=10000, pct_exp=0.75, seed=42)
print(f"\nCholesky: time={res_f['time_cholesky']:.4f}s, Frob={res_f['frobenius_cholesky']:.6f}")
print(f"PCA:      time={res_f['time_pca']:.4f}s, Frob={res_f['frobenius_pca']:.6f}")

# --- (e) If asked for simulation from this cov ---
sim_chol = chapter3.simulate_normal_cholesky(cov, nsim=10000, seed=42)
print(f"Cholesky sim shape: {sim_chol.shape}")
```

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

## 最可能的题型

> 给一个 non-PSD covariance matrix → 这几乎一定是 **Recipe D 系列**：
> 1. Verify it's not PSD（report min eigenvalue）
> 2. Fix with near_psd (Rebonato-Jäckel) and/or Higham
> 3. Compare both fixes（Frobenius norm to original）
> 4. PCA cumulative variance
> 5. Simulate from the fixed matrix（Cholesky or PCA）
> 6. 可能问"为什么不是 PSD"→ wording: pairwise estimation with missing data

**关键发现**：这个矩阵很可能是 Problem 2 的 returns 用 pairwise 方法估计出来的 covariance（但数值不完全一致，可能是另一组数据的产物）。

## 预写代码

```python
# ============ PROBLEM 3 ============
p3 = pd.read_csv("midterm/problem3.csv")
cov_raw = p3.values

# --- (a) Check PSD ---
print(f"Min eigenvalue: {eu.min_eigenvalue(cov_raw):.8f}")
print(f"Is PSD: {eu.is_psd(cov_raw)}")

# --- (b) Fix with near_psd ---
cov_near = chapter3.near_psd(cov_raw)
print(f"\nnear_psd min eigenvalue: {eu.min_eigenvalue(cov_near):.8f}")
frob_near = np.linalg.norm(cov_near - cov_raw, 'fro')
print(f"Frobenius distance (near_psd): {frob_near:.8f}")

# --- (c) Fix with Higham ---
cov_hig = chapter3.higham_nearest_psd(cov_raw)
print(f"\nHigham min eigenvalue: {eu.min_eigenvalue(cov_hig):.8f}")
frob_hig = np.linalg.norm(cov_hig - cov_raw, 'fro')
print(f"Frobenius distance (Higham): {frob_hig:.8f}")

# --- (d) Compare ---
print(f"\nHigham closer? {frob_hig < frob_near}")

# --- (e) PCA cumulative variance ---
cum = chapter3.pca_cumulative_variance(cov_hig)
for k, c in enumerate(cum):
    print(f"  k={k+1}: cumulative explained = {c:.4f}")

# --- (f) Simulate from fixed matrix ---
sim_hig = eu.simulate_normal_higham(cov_raw, nsim=10000, seed=42)
print(f"\nSimulation shape: {sim_hig.shape}")
# Verify covariance recovery
cov_sim = np.cov(sim_hig, rowvar=False)
frob_recovery = np.linalg.norm(cov_sim - cov_hig, 'fro')
print(f"Frobenius sim vs fixed: {frob_recovery:.6f}")

# --- (g) Cholesky vs PCA comparison on fixed matrix ---
res_f = eu.pipeline_cholesky_vs_pca(cov_hig, n_sim=10000, pct_exp=0.75, seed=42)
print(f"\nCholesky: Frob={res_f['frobenius_cholesky']:.6f}")
print(f"PCA(75%): Frob={res_f['frobenius_pca']:.6f}")
```

### Wording 备用

> "The input covariance matrix has min eigenvalue = -4.56e-05 < 0, meaning it is not positive semi-definite. This likely resulted from pairwise covariance estimation with missing data, where different observation subsets were used for different pairs. I apply Higham's alternating projection to find the nearest PSD matrix (Frobenius distance = {frob:.6f}). Higham is provably optimal — it produces a smaller Frobenius distance than Rebonato-Jäckel."

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

## 最可能的题型

> 真实股票 prices + 5 assets → 这几乎一定是 **考试的"大题" = Copula Pipeline（Recipe E / E2）**：
> 1. 计算 returns from prices
> 2. Fit marginals per asset（Normal or t）
> 3. Fit Gaussian Copula（Spearman correlation）
> 4. Simulate joint returns
> 5. Compute portfolio VaR / ES（给 holdings）
> 6. 可能要求多个 alpha levels（Recipe E2）
> 7. 可能要求 per-asset VaR/ES

还可能结合：
- "先 demean"
- "用 discrete vs log returns"
- "compare Spearman vs Pearson"

## 预写代码

```python
# ============ PROBLEM 4 ============

# --- (a) Load prices → returns ---
prices_df = eu.load_prices("midterm/problem4.csv", date_col="Date")
R, asset_names, last_prices = eu.prices_to_returns(prices_df, method="discrete", date_col="Date")
print(f"Returns shape: {R.shape}")
print(f"Asset names: {asset_names}")
print(f"Last prices: {last_prices}")

# --- (b) Fit marginals for each asset ---
print("\n=== Marginal Fits ===")
for j, name in enumerate(asset_names):
    r = R[:, j]
    res = eu.pipeline_moments_and_fit(r)
    print(f"\n{name}:")
    eu.print_moments(res)

# --- (c) Copula → Portfolio VaR/ES (single alpha) ---
alpha = 0.05       # ← 看题目
holdings = np.array([100, 200, 150, 100, 50])  # ← 看题目!!! 这个一定要从题目读
dist_types = ["t", "t", "t", "t", "t"]         # ← 看题目，可能 mix Normal/t

res_e = eu.pipeline_copula_portfolio(
    returns=R,
    holdings=holdings,
    current_prices=last_prices,
    dist_types=dist_types,
    alpha=alpha,
    n_sim=10000,
    method="spearman",
    seed=42,
    do_demean=True,
)
eu.print_portfolio_risk(res_e)

# --- (d) Multi-alpha version ---
alphas = [0.05, 0.01]  # ← 看题目
res_e2 = eu.pipeline_copula_multi_alpha(
    returns=R,
    holdings=holdings,
    current_prices=last_prices,
    alphas=alphas,
    dist_types=dist_types,
    n_sim=10000,
    method="spearman",
    seed=42,
    do_demean=True,
)
eu.print_multi_alpha_risk(res_e2)

# --- (e) If asked for copula correlation matrix ---
print("\nCopula Correlation:")
print(pd.DataFrame(res_e["copula_corr"], index=asset_names, columns=asset_names))

# --- (f) If asked for per-asset marginal parameters ---
print("\nMarginal Parameters:")
for m in res_e["marginals"]:
    print(m)
```

### Wording 备用

> "I convert prices to discrete returns, demean them, then fit per-asset marginals via MLE (t-distribution for all 5 assets, as excess kurtosis > 0 for each). I use Spearman correlation (rank-based, robust to heavy tails) to estimate the Gaussian copula correlation matrix. After simulating 10,000 joint return scenarios, I compute dollar PnL = return × holdings × current_price for each asset, sum to portfolio PnL, and take the empirical α-quantile for VaR and conditional tail mean for ES."

---

# Quick Tips — 考试时的操作顺序

```
1. 读题 → 确认 alpha, holdings, dist_types, lambda 等参数
2. 复制对应 Problem 的代码骨架
3. 修改参数 → 运行
4. 从 output 中提取数字填入 reporting table
5. 复制 wording 模板，替换数字
```

## 参数 Checklist（考试时逐项确认）

- [ ] `alpha` = ? (0.05? 0.01? 两个都要?)
- [ ] `holdings` = ? (每个 asset 多少 shares?)
- [ ] `dist_types` = ? (all t? mixed?)
- [ ] `lambda` = ? (0.97? 0.94? mixed?)
- [ ] `method` = ? (spearman? pearson?)
- [ ] `return_method` = ? (discrete? log?)
- [ ] `do_demean` = ? (True? False?)
- [ ] `n_sim` = ? (10000? 题目有规定?)
- [ ] `seed` = ? (42? 题目有规定?)
- [ ] `fix_method` = ? (higham? near_psd? both?)
- [ ] `pct_exp` = ? (0.75? 0.90? for PCA)
