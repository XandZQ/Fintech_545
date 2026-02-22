# QRM 545 Midterm Review — 总复习

> **使用方法**：这份文件是考前复习 + 考试中查阅的 **总地图**。每个知识点都链接到详细笔记和代码。考试时先定位问题类型 → 找到对应区域 → 跳转到详细笔记或 code recipe。

---

# Part 0 — 考试快速导航

## 看到题目后的决策流程

```
题目关键词 → 对应知识区域 → 操作
─────────────────────────────────────────────
"mean / variance / skew / kurtosis"
    → Part 1 (Moments) → Recipe A

"Normal vs t / AICc / fit distribution"
    → Part 2 (Distribution Fitting) → Recipe A

"VaR at X% / ES"  (单个 asset)
    → Part 3 (VaR/ES) → Recipe B

"EW / lambda / exponential weights"
    → Part 4 (Covariance) → Recipe C

"missing data / pairwise / not PSD / Higham"
    → Part 4 (Covariance) → Recipe D

"Cholesky / PCA simulation / compare"
    → Part 5 (Simulation) → Recipe F

"copula / marginals / portfolio VaR/ES"
    → Part 6 (Copula Pipeline) → Recipe E

"two alpha levels / multiple confidence"
    → Part 6 → Recipe E2

"conditional distribution / X2|X1"
    → Part 7 (Conditional) → Bonus: Conditional

"AR / MA / best order / ACF / PACF"
    → Part 8 (Time Series) → Bonus: AR/MA

"GARCH / volatility clustering"
    → Part 8 (Time Series)

"regression / beta / OLS / t-regression"
    → Part 9 (Regression)

"cov → corr / mixed lambda"
    → Part 4 → Bonus: Cov/Corr Utils
```

> 详细的 copy-paste 代码在 [[EXAM_QUICKREF]]。

---

# Part 1 — Moments（四个描述分布的数字）

> 详细笔记：[[Chapter 1 - Foundations Rebuild#4. Moments — 分布形状的四个数字]]

| Moment | 公式 | 含义 | Finance 意义 |
|---|---|---|---|
| Mean $\mu$ | $E[X]$ | 期望值 | Expected return（不是 risk measure） |
| Variance $\sigma^2$ | $E[(X-\mu)^2]$ | 离散程度 | Volatility |
| Skewness | $E[(X-\mu)^3]/\sigma^3$ | 左右不对称 | $<0$ = 左尾长 = 危险 |
| Excess Kurtosis | $E[(X-\mu)^4]/\sigma^4 - 3$ | 尾巴厚度 vs Normal | $>0$ = heavy tails = 极端事件频繁 |

### Sample Estimator 要点

- Variance 用 $n-1$ 分母（Bessel's correction）→ unbiased
- Kurtosis 的 sample estimator **最不稳定**，小样本 bias 显著
- 详细公式见 [[Chapter 1 - Foundations Rebuild#4.2 Sample Moments（样本矩）— 从数据中估计]]

### 考试操作

```python
res = eu.pipeline_moments_and_fit(x)
eu.print_moments(res)
```

---

# Part 2 — Distribution Fitting（Normal vs t）

> 详细笔记：[[Chapter 1 - Foundations Rebuild#7. Student's t Distribution]] 和 [[Chapter 2 - Multivariate and Regression#7.4 AIC 与 AICc — 模型选择]]

## Normal vs t 核心对比

| | Normal $\mathcal{N}(\mu, \sigma^2)$ | Student's t $t(\nu, \mu, \sigma)$ |
|---|---|---|
| 参数 | 2 个 ($\mu, \sigma$) | 3 个 ($\mu, \sigma, \nu$) |
| 尾巴 | Thin (exponential decay) | Heavy (polynomial decay) |
| Kurtosis | 0 (baseline) | $6/(\nu-4)$ when $\nu > 4$ |
| 适用 | Baseline / CLT | 真实金融数据 |

## 选模型：AICc

$$\text{AICc} = -2\ell(\hat{\theta}) + 2k + \frac{2k(k+1)}{n-k-1}$$

> **AICc 越小 = 越好**。t 多一个参数 $\nu$（$k=3$ vs $k=2$），但如果 heavy tail 确实存在，likelihood 提升足以补偿。

### 关键 wording

> "The t-distribution has lower AICc ({aicc_t:.2f} vs {aicc_norm:.2f}), indicating better fit. The estimated $\nu$ = {nu:.1f} confirms heavy tails — excess kurtosis under this t is $6/(\nu-4)$ = {ek:.2f}, matching the sample excess kurtosis."

---

# Part 3 — VaR & ES（Risk Metrics 核心）

> 详细笔记：[[Chapter 4 - Value-at-Risk (VaR)]] 和 [[Chapter 5 - Advanced VaR & Expected Shortfall]]

## 3.1 VaR 定义

$$\text{VaR}_\alpha = -F^{-1}(\alpha) \quad \text{(positive = loss)}$$

> "在 $\alpha$ 概率下，最多亏多少？" 例如 $\alpha = 0.05$：95% 的情况下损失不超过 VaR。

## 3.2 三种计算方法

| 方法 | 公式 / 思路 | 何时用 |
|---|---|---|
| **Normal** | $\text{VaR} = -(\mu + z_\alpha \cdot \sigma)$ | 假设 Normal |
| **t** | $\text{VaR} = -(\mu + t_\alpha(\nu) \cdot \sigma)$ | Fit t 后 |
| **Empirical** | 直接排序取 $\alpha$ 分位数 | 不假设分布 |

常用 $z_\alpha$：$z_{0.05} = -1.645$，$z_{0.01} = -2.326$

## 3.3 ES (Expected Shortfall)

$$\text{ES}_\alpha = -E[X \mid X \le F^{-1}(\alpha)]$$

> "如果损失超过了 VaR，**平均**会亏多少？" ES 永远 $\ge$ VaR。

### Normal ES 公式

$$\text{ES}_\alpha = -\mu + \sigma \cdot \frac{\phi(z_\alpha)}{\alpha}$$

> 详细推导在 [[Chapter 4 - Value-at-Risk (VaR)#6. Expected Shortfall]]：核心积分恒等式 $\int_{-\infty}^{c} z\,\phi(z)\,dz = -\phi(c)$。

### ES vs VaR 的关系

| 性质 | VaR | ES |
|---|---|---|
| 定义 | 尾部的"门槛" | 尾部的"平均" |
| Coherent risk measure? | **不是**（不满足 sub-additivity） | **是** |
| 对 tail shape 敏感? | 不敏感（只看一个点） | 敏感（看整个尾部） |

### 考试 wording

> "VaR 只告诉你 tail 的 boundary，ES 告诉你 tail 里面有多糟糕。t-distribution 下 ES 远大于 Normal 下的 ES，因为 t 的尾巴更厚。ES 是 coherent risk measure（满足 sub-additivity），VaR 不是。"

---

# Part 4 — Covariance Matrix（Portfolio Risk 基础）

> 详细笔记：[[Chapter 2 - Multivariate and Regression#3. Covariance Matrix — Portfolio Risk 的核心]] 和 [[Chapter 3 - Financial Data & Monte Carlo Simulation]]

## 4.1 核心公式

$$\text{Var}(\mathbf{w}^T\mathbf{X}) = \mathbf{w}^T \Sigma \, \mathbf{w}$$

> Portfolio risk 取决于 **所有 pairwise interactions**，不是个体 risk 的简单加总。

## 4.2 Covariance 估计方法

| 方法 | 特点 | 何时用 |
|---|---|---|
| Equal-weight | $\frac{1}{n-1}\sum(x-\bar{x})(x-\bar{x})^T$ | 默认 |
| EW ($\lambda$) | 近期数据权重更大 | "exponential weighting" / "RiskMetrics" |
| Pairwise | 每对用各自的非缺失观测 | 有 missing data |
| Complete-case | 只用全部非缺失的行 | 保守处理 missing |

详见 [[Chapter 3 - Financial Data & Monte Carlo Simulation#5. Exponential Weighting]]

## 4.3 Cov ↔ Corr 转换

$$C = D^{-1}\Sigma D^{-1}, \quad \Sigma = DCD$$

> 其中 $D = \text{diag}(\sigma_1, \dots, \sigma_n)$。代码：`eu.cov_to_corr()` / `eu.corr_to_cov()`

## 4.4 Mixed-Lambda EW（Test 2.3）

分别用不同 $\lambda$ 估计 variance 和 correlation，再组合：

```python
cov = eu.ew_cov_mixed_lambda(X, lam_var=0.97, lam_corr=0.94)
```

## 4.5 PSD 修正

Pairwise 估计可能 → 非 PSD（min eigenvalue $< 0$）。

| 方法 | 函数 | 特点 |
|---|---|---|
| Rebonato-Jäckel | `chapter3.near_psd()` | 快，但非最优 |
| Higham | `chapter3.higham_nearest_psd()` | Frobenius 最优 |

详细算法见 [[Chapter 3 - Financial Data & Monte Carlo Simulation#8. Non-PSD 修复]]

---

# Part 5 — Simulation（Cholesky & PCA）

> 详细笔记：[[Chapter 3 - Financial Data & Monte Carlo Simulation#4. Multivariate Normal Simulation]]

## 5.1 Cholesky Simulation

$$\mathbf{X} = L\mathbf{Z} + \boldsymbol{\mu}$$

- $\Sigma = LL^T$（Cholesky 分解）
- $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, I)$
- **精确**：no approximation
- 3×3 数值例子见 [[Chapter 3 - Financial Data & Monte Carlo Simulation#4.2 Cholesky 分解]]

## 5.2 PCA Simulation

1. 特征分解 $\Sigma = P\Lambda P^T$
2. 只保留前 $k$ 个 eigenvalues（explain ≥ 指定比例方差）
3. 模拟 $k$ 个独立标准正态 → 用 $P_k \sqrt{\Lambda_k}$ 映射回原空间

> **Trade-off**：PCA 更快（维度低），但引入 approximation error。

## 5.3 考试关键点

| | Cholesky | PCA |
|---|---|---|
| 精确度 | Exact（只有 sampling noise） | 有 truncation error |
| 速度 | $O(n^3)$ | 高维时更快 |
| 需要 PSD？ | 是 | 是 |
| 比较指标 | Frobenius norm of $\hat\Sigma - \Sigma$ | Same |

---

# Part 6 — Copula Pipeline（Portfolio VaR/ES 大题）

> 详细笔记：[[Chapter 5 - Advanced VaR & Expected Shortfall#5. Gaussian Copula]]

这通常是考试最大的综合题。完整 workflow：

```
原始 returns → demean → fit marginals (Normal/t) per asset
    → CDF transform → U (uniform) → fit copula correlation (Spearman)
    → simulate Z from MVN → back-transform to U → inverse CDF → simulated returns
    → dollar PnL = sim_returns × holdings × prices → VaR/ES from PnL
```

## 6.1 每一步的数学

| Step | 数学操作 | 代码 |
|---|---|---|
| Fit marginals | MLE fit Normal or t per column | `chapter5.fit_copula_marginals()` |
| CDF transform | $U_i = F_i(X_i;\hat\theta_i)$ | 同上（内部） |
| Copula corr | Spearman on U → $\hat\rho_S$ → $\rho = 2\sin(\frac{\pi}{6}\hat\rho_S)$ | `chapter5.fit_gaussian_copula_corr()` |
| Simulate | $Z \sim \mathcal{N}(0, \hat{C})$ → $U = \Phi(Z)$ → $X = F^{-1}(U)$ | `chapter5.simulate_copula_from_fitted()` |
| PnL | $\text{PnL}_j = r_j \times h_j \times p_j$，portfolio = $\sum$ | `eu.portfolio_pnl()` |
| VaR/ES | 排序 PnL，取 quantile / conditional mean | `eu.var_es_from_pnl()` |

## 6.2 Spearman vs Pearson for Copula

> **用 Spearman**（除非题目明确说 Pearson）：Spearman 是 rank-based，对 heavy-tailed marginals 更鲁棒，而且和 copula 的 concordance structure 直接对应。

详见 [[Chapter 2 - Multivariate and Regression#2.2 Spearman (Rank) Correlation]]

## 6.3 考试代码

```python
# 单一 alpha
res = eu.pipeline_copula_portfolio_from_prices("PRICES.csv", holdings, ...)

# 多 alpha (Test 9.1)
res = eu.pipeline_copula_multi_alpha_from_prices("PRICES.csv", holdings, alphas=[0.05, 0.01], ...)
```

---

# Part 7 — Conditional Distribution

> 详细笔记：[[Chapter 2 - Multivariate and Regression#5. Conditional Distribution（条件分布）]]

## 核心公式（Bivariate Normal）

$$X_2 | X_1 = x_1 \;\sim\; \mathcal{N}\!\left(\mu_2 + \frac{\sigma_{12}}{\sigma_1^2}(x_1 - \mu_1), \;\; \sigma_2^2 - \frac{\sigma_{12}^2}{\sigma_1^2}\right)$$

### 三个关键 insight

1. **条件均值** = 原均值 + 调整（$\rho$ 方向 × 偏离量）
2. **条件方差** < 无条件方差（观测 $X_1$ **总是** 减少对 $X_2$ 的 uncertainty）
3. **条件方差不依赖 $x_1$ 的具体值**（只依赖 correlation 强度）

---

# Part 8 — Time Series & GARCH

> 详细笔记：[[Chapter 2 - Multivariate and Regression#8. Time Series — 时间维度的依赖]] 和 [[Chapter 2 - Multivariate and Regression#9. GARCH — Volatility 也有 Memory]]

## 8.1 AR vs MA 速查

| | AR(p) | MA(q) |
|---|---|---|
| 公式 | $X_t = c + \sum \phi_i X_{t-i} + \varepsilon$ | $X_t = c + \varepsilon_t + \sum \theta_j \varepsilon_{t-j}$ |
| ACF | **缓慢衰减** | **lag $q$ 后截断 = 0** |
| PACF | **lag $p$ 后截断 = 0** | 缓慢衰减 |
| Stationarity | $|\phi_1| < 1$ for AR(1) | 总是 stationary |
| Memory | 状态持续，缓慢衰减 | Shock 存活 $q$ 期后消失 |

> **模型识别口诀**：ACF 截断 → MA；PACF 截断 → AR。

## 8.2 GARCH(1,1)

$$\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2$$

| 要点 | 公式 / 值 |
|---|---|
| Unconditional variance | $\bar\sigma^2 = \frac{\omega}{1-\alpha-\beta}$ |
| Stationarity | $\alpha + \beta < 1$ |
| Persistence | $\alpha + \beta$（越接近 1 → vol cluster 越持久） |

> **直觉**：$\alpha$ = "对昨天 shock 的反应速度"，$\beta$ = "昨天 vol 的延续强度"。

---

# Part 9 — Regression

> 详细笔记：[[Chapter 2 - Multivariate and Regression#6. Regression（回归分析）]]

## 9.1 OLS

$$\hat\beta = (X^TX)^{-1}X^TY$$

> "选择 $\beta$ 使残差平方和最小" = "最优线性解释"。

## 9.2 OLS = MLE (under Normal errors)

详见 [[Chapter 2 - Multivariate and Regression#7.2 OLS = MLE under Normal Errors]]

## 9.3 t-Regression

当 errors 是 heavy-tailed 时，用 $t$ 分布 MLE 替代 OLS：

```python
result = eu.mle_regression_t(X, y)
# Returns: beta, sigma, nu, aicc, residuals
```

> 和 OLS 对比：如果 $\hat\nu$ 较小（如 < 10），说明 errors 确实有 heavy tails，t-regression 比 OLS 更合适。

---

# Part 10 — Coherent Risk Measures（Wording 题必考）

> 详细笔记：[[Chapter 4 - Value-at-Risk (VaR)#7. Coherent Risk Measures]]

一个 coherent risk measure 必须满足 4 条公理：

| 公理 | 含义 | VaR | ES |
|---|---|---|---|
| Monotonicity | 更差的 portfolio → 更高 risk | ✓ | ✓ |
| Sub-additivity | $\rho(A+B) \le \rho(A) + \rho(B)$ | **✗** | ✓ |
| Positive Homogeneity | 加倍仓位 → 加倍 risk | ✓ | ✓ |
| Translation Invariance | 加现金 → 减少 risk | ✓ | ✓ |

### 关键 wording

> "VaR is **not** a coherent risk measure because it fails sub-additivity: diversification can appear to increase VaR, which contradicts economic intuition. ES is coherent — it always rewards diversification. This is a major reason regulators (Basel III) moved from VaR to ES."

---

# Part 11 — 常见陷阱 Checklist

考试前过一遍，避免扣分：

- [ ] **Demean**：题目说 "demean" 或 "zero-mean" → 先 `eu.demean(X)` 再做后续
- [ ] **VaR 符号**：Report as **positive loss**（bigger = worse）
- [ ] **Alpha 含义**：$\alpha = 0.05$ = 左尾 5% = 95% confidence
- [ ] **Spearman for copula**：除非题目指定 Pearson，默认用 Spearman
- [ ] **PSD check**：pairwise cov → check `eu.min_eigenvalue()` → if $< 0$ → Higham
- [ ] **AICc not AIC**：小样本必须用 AICc
- [ ] **t-distribution variance**：$\text{Var}(X) = \sigma^2 \cdot \frac{\nu}{\nu-2}$，**不是** $\sigma^2$
- [ ] **Lognormal mean**：$E[X] = e^{\mu + \sigma^2/2}$，**不是** $e^\mu$
- [ ] **AR stationarity**：$|\phi_1| < 1$，等于 1 是 random walk
- [ ] **GARCH stationarity**：$\alpha + \beta < 1$
- [ ] **Matrix shapes**：cov = $(n \times n)$，simulation = $(n_{\text{sim}} \times n_{\text{assets}})$

---

# Part 12 — 公式速查表

## Distributions

| 分布 | PDF 核心部分 | 参数 |
|---|---|---|
| Normal | $\exp(-\frac{(x-\mu)^2}{2\sigma^2})$ | $\mu, \sigma$ |
| t | $(1+\frac{t^2}{\nu})^{-(\nu+1)/2}$ | $\nu, \mu, \sigma$ |
| Lognormal | $\ln X \sim \mathcal{N}(\mu, \sigma^2)$ | $\mu, \sigma$ |

## VaR / ES

| | Normal | t | Empirical |
|---|---|---|---|
| VaR | $-(\mu + z_\alpha\sigma)$ | $-(\mu + t_\alpha(\nu)\sigma)$ | 排序取 quantile |
| ES | $-\mu + \sigma\frac{\phi(z_\alpha)}{\alpha}$ | 数值积分 / simulation | 尾部条件均值 |

## Key Constants

$z_{0.05} = -1.645, \quad z_{0.01} = -2.326, \quad z_{0.025} = -1.960$

## Covariance

| 操作 | 公式 |
|---|---|
| Portfolio variance | $\mathbf{w}^T\Sigma\mathbf{w}$ |
| Cov → Corr | $C = D^{-1}\Sigma D^{-1}$ |
| Corr → Cov | $\Sigma = DCD$ |
| EW cov | $\Sigma_t = (1-\lambda)\mathbf{r}_{t-1}\mathbf{r}_{t-1}^T + \lambda\Sigma_{t-1}$ |
| Unconditional GARCH var | $\bar\sigma^2 = \frac{\omega}{1-\alpha-\beta}$ |

## Conditional Distribution (Bivariate)

$$\mu_{2|1} = \mu_2 + \frac{\sigma_{12}}{\sigma_1^2}(x_1 - \mu_1), \quad \sigma^2_{2|1} = \sigma_2^2 - \frac{\sigma_{12}^2}{\sigma_1^2}$$

## Model Selection

$$\text{AICc} = -2\ell + 2k + \frac{2k(k+1)}{n-k-1} \quad \text{(smaller = better)}$$

---

# Part 13 — Code Quick Reference 入口

所有 copy-paste 代码都在 [[EXAM_QUICKREF]]：

| Recipe | 用途 | 关键函数 |
|---|---|---|
| A | Moments + Normal vs t | `eu.pipeline_moments_and_fit()` |
| B | Univariate VaR/ES | `eu.pipeline_var_es_univariate()` |
| C | EW Covariance | `eu.pipeline_ew_covariance()` |
| D | Missing → PSD fix → PCA | `eu.pipeline_missing_psd_pca()` |
| E | Copula → Portfolio VaR/ES | `eu.pipeline_copula_portfolio_from_prices()` |
| E2 | Multi-alpha Copula | `eu.pipeline_copula_multi_alpha_from_prices()` |
| F | Cholesky vs PCA | `eu.pipeline_cholesky_vs_pca()` |
| Bonus | Cov↔Corr, t-VaR, t-regression, Higham sim | 见 [[EXAM_QUICKREF]] Bonus 区域 |

---

# Part 14 — Wording 题模板库

考试 wording 部分可以直接改关键数字用：

### "Why is ES > VaR?"
> "ES averages over the entire tail beyond VaR, while VaR only marks the tail boundary. ES captures the severity of extreme losses, not just their threshold."

### "Why t over Normal?"
> "The t-distribution has lower AICc, indicating better fit. Its heavier tails (controlled by $\nu$) capture the excess kurtosis observed in financial returns, which the Normal distribution cannot."

### "Why Spearman for copula?"
> "Spearman correlation is rank-based and measures monotonic (not just linear) dependence. It is robust to heavy-tailed marginals and directly related to the copula's concordance structure."

### "Why EW over equal-weight?"
> "Exponentially weighted estimates give more weight to recent observations, capturing time-varying volatility. Equal-weight treats a shock from 3 years ago the same as yesterday's — unrealistic for dynamic markets."

### "Why is VaR not coherent?"
> "VaR fails sub-additivity: $\text{VaR}(A+B)$ can exceed $\text{VaR}(A) + \text{VaR}(B)$, meaning diversification appears to increase risk. This contradicts economic intuition and makes VaR unsuitable as a portfolio-level risk measure. ES satisfies all four coherence axioms."

### "What does Higham do?"
> "Higham's alternating projection finds the nearest PSD matrix to a given non-PSD matrix under Frobenius norm. It preserves the correlation structure as much as possible while ensuring all eigenvalues are non-negative, which is required for valid simulation via Cholesky."

### "Explain GARCH volatility clustering"
> "GARCH captures the empirical fact that large shocks increase future uncertainty ($\alpha$ term) and that current volatility persists ($\beta$ term). High $\alpha + \beta$ means volatility regimes are long-lived — crises don't end quickly."

### "PCA vs Cholesky tradeoff"
> "Cholesky uses the full covariance matrix with no approximation. PCA reduces dimensionality by keeping only the top eigenvalues, introducing truncation error but gaining computational speed. The Frobenius norm of the reconstructed vs original covariance measures this tradeoff."
