# Chapter 2 — Multivariate Statistics & Regression

> 这一章从 univariate 拓展到 **multivariate**：多个变量如何一起运动？然后引入 **regression**（解释 Y 为什么变化）和 **time series**（今天是否依赖昨天），最后是 **GARCH**（volatility 本身有 memory）。

---

# 1. Covariance — 最基本的"共同运动"度量

## 1.1 定义

$$\text{Cov}(X, Y) = E\big[(X - \mu_X)(Y - \mu_Y)\big]$$

展开：

$$= E[XY] - E[X]E[Y]$$

### 直觉

> 当 $X$ 高于均值时，$Y$ 是否也高于均值？

| $\text{Cov}(X,Y)$ | 含义 |
|---|---|
| $> 0$ | 同向运动（一起涨一起跌） |
| $< 0$ | 反向运动 |
| $\approx 0$ | 没有 **线性** 共同运动 |

### 为什么用"偏差"？

- 绝对值不重要，**surprise 才重要**
- Mean = expected，deviation = unexpected
- Covariance 衡量的是：**surprises 是否同时发生**

## 1.2 重要性质

1. $\text{Cov}(X, X) = \text{Var}(X)$
2. $\text{Cov}(X, Y) = \text{Cov}(Y, X)$（symmetric）
3. $\text{Cov}(aX + b, cY + d) = ac \cdot \text{Cov}(X, Y)$
4. Covariance 是 **scale-dependent**（单位改了值就变了）→ 这就是为什么需要 correlation

## 1.3 Sample Covariance

给定 $n$ 个观测 $(x_1, y_1), \dots, (x_n, y_n)$：

$$\hat{\text{Cov}}(X,Y) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$

> 分母用 $n-1$ 而非 $n$（[[Chapter 1 - Foundations Rebuild#Sample Variance|Bessel's correction]]），这样是 unbiased estimator。

---

# 2. Correlation — 标准化后的 Covariance

## 2.1 Pearson Correlation

$$\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \cdot \sigma_Y}$$

### 本质

> 把 covariance 用"标准差"做单位归一化，结果 **无量纲**、范围在 $[-1, +1]$。

| $\rho$ | 含义 |
|---|---|
| $+1$ | 完美正线性关系：$Y = a + bX,\; b > 0$ |
| $-1$ | 完美负线性关系：$Y = a + bX,\; b < 0$ |
| $0$ | 无 **线性** 关系（但可能有非线性依赖！） |

### 关键警告

$$\rho = 0 \;\not\Rightarrow\; \text{independence}$$

> PDF 第 2 页的图清楚展示了这一点：可以构造 $\rho = 0$ 但有强非线性依赖的例子。

## 2.2 Spearman (Rank) Correlation

Spearman 不看原始数值，而是看 **排名（ranks）**：

1. 把 $x_1, \dots, x_n$ 转化为 ranks $r_1, \dots, r_n$
2. 把 $y_1, \dots, y_n$ 转化为 ranks $s_1, \dots, s_n$
3. 对 ranks 算 Pearson correlation

$$\rho_S = \text{Pearson}(\text{rank}(X), \text{rank}(Y))$$

### Pearson vs Spearman 对比

|           | Pearson     | Spearman              |
| --------- | ----------- | --------------------- |
| 衡量什么      | **线性** 关系   | **单调** 关系（monotonic）  |
| 对 outlier | 敏感          | 鲁棒（用 rank 消除极端值影响）    |
| 适合场景      | Gaussian 模型 | 重尾分布 / copula fitting |

### 直觉

> Pearson 问："数据是否在一条 **直线** 上？"
> Spearman 问："数据是否在一条 **单调曲线** 上？"

在 risk management 中，Spearman 更常用于 [[Chapter 5 - Advanced VaR & Expected Shortfall#4.3 为什么用 Spearman Correlation？|copula fitting]]（Chapter 5），因为金融数据 rarely Gaussian。

---

# 3. Covariance Matrix — Portfolio Risk 的核心

## 3.1 定义

对 $n$ 个随机变量 $X_1, \dots, X_n$，covariance matrix $\Sigma$ 的第 $(i,j)$ 元素是：

$$\Sigma_{ij} = \text{Cov}(X_i, X_j)$$

写成矩阵形式：

$$\Sigma = E\big[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T\big]$$

结构：

$$\Sigma = \begin{pmatrix} \text{Var}(X_1) & \text{Cov}(X_1, X_2) & \cdots \\ \text{Cov}(X_2, X_1) & \text{Var}(X_2) & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}$$

- **对角线**：每个变量的 variance
- **非对角线**：两两之间的 covariance
- **对称矩阵**：$\Sigma = \Sigma^T$

## 3.2 Portfolio Variance — 最重要的公式

$$\text{Var}(\mathbf{w}^T \mathbf{X}) = \mathbf{w}^T \Sigma \, \mathbf{w}$$

其中 $\mathbf{w} = (w_1, \dots, w_n)^T$ 是 portfolio weights。

### 两资产的展开

$$\text{Var}(w_1 X_1 + w_2 X_2) = w_1^2 \sigma_1^2 + w_2^2 \sigma_2^2 + 2 w_1 w_2 \text{Cov}(X_1, X_2)$$

> **关键 insight**：portfolio risk 不是各 asset risk 的简单加和，而是取决于它们之间的 **interaction**（covariance）。Diversification 能降低 risk 正是因为 $\text{Cov}$ 可以是负的或很小。

## 3.3 Positive Semi-Definite (PSD)

一个合法的 covariance matrix 必须是 PSD：

$$\mathbf{w}^T \Sigma \, \mathbf{w} \ge 0 \quad \text{for all } \mathbf{w}$$

### 为什么？

> 因为 $\mathbf{w}^T \Sigma \, \mathbf{w} = \text{Var}(\mathbf{w}^T \mathbf{X})$，而 variance 不能是负数。

等价条件：$\Sigma$ 的所有 eigenvalues $\ge 0$。

> 如果估计出的 covariance matrix 不是 PSD（比如 [[Chapter 3 - Financial Data & Monte Carlo Simulation#5.2 两种常见处理方法|pairwise 估计]] + missing data），需要用 [[Chapter 3 - Financial Data & Monte Carlo Simulation#9. Fix 2：Higham Algorithm（最近 PSD 矩阵）|Higham]] / [[Chapter 3 - Financial Data & Monte Carlo Simulation#8. Fix 1：Rebonato–Jäckel Eigenvalue Cleaning|near_psd]] 修正。

## 3.4 Cov ↔ Corr 互转

从 covariance matrix 提取 correlation matrix：

$$\text{Corr}_{ij} = \frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii}} \cdot \sqrt{\Sigma_{jj}}}$$

矩阵表达：

$$C = D^{-1} \Sigma \, D^{-1}$$

其中 $D = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_n)$。

反过来从 correlation + 标准差重建 covariance：

$$\Sigma = D \, C \, D$$

> 这在 mixed-lambda EW estimation（Test 2.3）中很关键：先分别算 EW variance（$\lambda = 0.97$）和 EW correlation（$\lambda = 0.94$），再用 $D \, C \, D$ 组合。

---

# 4. Multivariate Normal Distribution

## 4.1 定义

$$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$$

密度函数（PDF）：

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

其中：
- $\boldsymbol{\mu}$：mean vector（$n \times 1$）
- $\Sigma$：covariance matrix（$n \times n$，PSD）
- $|\Sigma|$：determinant
- $(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})$：**Mahalanobis distance**（考虑 correlation 的"距离"）

## 4.2 关键性质

1. **Marginals are Normal**：任何一个 $X_i$ 单独看都是 Normal
2. **Linear combinations are Normal**：$\mathbf{a}^T \mathbf{X} \sim \mathcal{N}(\mathbf{a}^T \boldsymbol{\mu}, \; \mathbf{a}^T \Sigma \, \mathbf{a})$
3. **Conditional is Normal**：$X_1 | X_2 = x_2$ 仍然是 Normal（见下面）
4. **Uncorrelated ⟹ Independent**：这是 MVN 的特殊性质！一般分布中 uncorrelated $\not\Rightarrow$ independent

> 性质 4 是 Gaussian 世界的"便利性"：correlation = dependence。但真实市场中不成立，这是 [[Chapter 5 - Advanced VaR & Expected Shortfall#3. Breaking the Normality Assumption — Copulas|Copula]]（Chapter 5）存在的理由。

## 4.3 Simulation: $\mathbf{X} = L\mathbf{Z} + \boldsymbol{\mu}$

从 MVN 中模拟：

1. 对 $\Sigma$ 做 Cholesky 分解：$\Sigma = L L^T$
2. 生成独立标准正态向量 $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, I)$
3. 计算 $\mathbf{X} = L\mathbf{Z} + \boldsymbol{\mu}$

> **直觉**：$L$ 把独立的 shocks "染色"（inject correlation），变成有相关性的联合运动。

详细的 Cholesky 推导和 3×3 数值例子在 [[Chapter 3 - Financial Data & Monte Carlo Simulation#4. Cholesky Factorization（如何求 $L$）]] 中。

---

# 5. Conditional Distribution（条件分布）

## 5.1 Bivariate Normal Case

设 $(X_1, X_2)^T \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$：

$$\boldsymbol{\mu} = \begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}, \quad \Sigma = \begin{pmatrix} \sigma_1^2 & \sigma_{12} \\ \sigma_{12} & \sigma_2^2 \end{pmatrix}$$

则给定 $X_1 = x_1$，$X_2$ 的条件分布是：

$$X_2 \mid X_1 = x_1 \;\sim\; \mathcal{N}\!\left(\mu_{2|1},\; \sigma^2_{2|1}\right)$$

其中：

$$\mu_{2|1} = \mu_2 + \frac{\sigma_{12}}{\sigma_1^2}(x_1 - \mu_1)$$

$$\sigma^2_{2|1} = \sigma_2^2 - \frac{\sigma_{12}^2}{\sigma_1^2}$$

### 逐步推导

**Step 1**：写出 joint density 的 exponent 部分：

$$Q = (x_1 - \mu_1, \; x_2 - \mu_2) \, \Sigma^{-1} \begin{pmatrix} x_1 - \mu_1 \\ x_2 - \mu_2 \end{pmatrix}$$

**Step 2**：把 $x_1$ 视为已知常数，$Q$ 是关于 $x_2$ 的二次式，"配方"（complete the square）即可读出条件分布的均值和方差。

**Step 3**：配方结果：

$$Q = \frac{1}{\sigma^2_{2|1}}\left(x_2 - \mu_{2|1}\right)^2 + \text{只和 } x_1 \text{ 有关的常数}$$

因此 $X_2 | X_1$ 的 density 就是以 $\mu_{2|1}$ 为均值、$\sigma^2_{2|1}$ 为方差的 Normal。

### 直觉

- **条件均值** $\mu_{2|1}$：原始均值 $\mu_2$ 加了一个 **调整项** $\frac{\sigma_{12}}{\sigma_1^2}(x_1 - \mu_1)$
  - 如果 $x_1 > \mu_1$（$X_1$ 比预期高）且 $\sigma_{12} > 0$（正相关），则 $X_2$ 的条件均值也上调
  - 调整项 = $\rho \cdot \frac{\sigma_2}{\sigma_1} \cdot (x_1 - \mu_1)$（这就是 [[Chapter 2 - Multivariate and Regression#6. Regression（回归分析）|regression]] 的系数！）
- **条件方差** $\sigma^2_{2|1}$：**一定比无条件方差 $\sigma_2^2$ 小**
  - 减小的量 = $\frac{\sigma_{12}^2}{\sigma_1^2} = \rho^2 \sigma_2^2$
  - $|\rho|$ 越大 → 观察 $X_1$ 后 uncertainty 减少越多
  - **条件方差不依赖于 $x_1$ 的具体值**（只依赖相关性强度）

### 数值例子

$$\mu = \begin{pmatrix} 2 \\ 5 \end{pmatrix}, \quad \Sigma = \begin{pmatrix} 4 & 3 \\ 3 & 9 \end{pmatrix}$$

观测 $X_1 = 4$：

$$\mu_{2|1} = 5 + \frac{3}{4}(4 - 2) = 5 + 1.5 = 6.5$$

$$\sigma^2_{2|1} = 9 - \frac{9}{4} = 6.75$$

所以 $X_2 | X_1 = 4 \sim \mathcal{N}(6.5, \; 6.75)$。

> 无条件下 $X_2 \sim \mathcal{N}(5, 9)$；观察到 $X_1 = 4 > \mu_1$ 后，条件均值从 5 上移到 6.5（因为正相关），方差从 9 缩小到 6.75（信息减少了 uncertainty）。

## 5.2 General Case（多变量）

分块：$\mathbf{X} = (X_A, X_B)^T$，则：

$$X_A | X_B = x_B \;\sim\; \mathcal{N}\!\left(\mu_A + \Sigma_{AB}\Sigma_{BB}^{-1}(x_B - \mu_B), \;\; \Sigma_{AA} - \Sigma_{AB}\Sigma_{BB}^{-1}\Sigma_{BA}\right)$$

> 和 bivariate case 结构完全一样，只是标量 $\sigma_{12}/\sigma_1^2$ 变成了矩阵 $\Sigma_{AB}\Sigma_{BB}^{-1}$。

---

# 6. Regression（回归分析）

## 6.1 模型设定

$$Y = X\beta + \varepsilon$$

- $Y$：response vector（$n \times 1$）
- $X$：design matrix（$n \times (p+1)$，含 intercept 列）
- $\beta$：coefficient vector（$(p+1) \times 1$）
- $\varepsilon$：error vector，假设 $\varepsilon \sim \mathcal{N}(\mathbf{0}, \sigma^2 I)$

### Finance 解读

| 数学符号 | Finance 含义 |
|---|---|
| $\beta$ | Factor exposure（因子暴露） |
| $\varepsilon$ | Idiosyncratic / residual risk |
| $X\beta$ | Systematic risk（系统性风险） |
| $\sigma^2$ | Residual variance |

> Regression 是"有方向的 covariance"：不只说 $X$ 和 $Y$ 一起动，还说 "因为 $X$ 变了，所以 $Y$ 变了多少"。

## 6.2 OLS 估计量推导

### 目标

最小化 residual sum of squares（RSS）：

$$\hat{\beta} = \arg\min_\beta \|Y - X\beta\|^2 = \arg\min_\beta (Y - X\beta)^T(Y - X\beta)$$

### 展开

$$\text{RSS}(\beta) = Y^T Y - 2\beta^T X^T Y + \beta^T X^T X \beta$$

### 求导并令为零

$$\frac{\partial \text{RSS}}{\partial \beta} = -2X^T Y + 2X^T X \beta = 0$$

$$\Rightarrow \quad X^T X \beta = X^T Y$$

$$\boxed{\hat{\beta} = (X^T X)^{-1} X^T Y}$$

### 理解

- $(X^T X)$：regressors 之间的 "inner product matrix"
- $(X^T Y)$：regressors 和 response 的交叉项
- $(X^T X)^{-1}$：去除 regressors 之间的 redundancy

> OLS = **选择 $\beta$ 使得"解释不了的部分"尽可能小**。

### 条件

$(X^TX)$ 必须可逆 ↔ $X$ 列满秩 ↔ 没有完美 multicollinearity。

## 6.3 OLS 估计量的性质

$$E[\hat{\beta}] = \beta \quad \text{(unbiased)}$$

$$\text{Var}(\hat{\beta}) = \sigma^2 (X^T X)^{-1}$$

Residual variance 的无偏估计：

$$\hat{\sigma}^2 = \frac{1}{n - p - 1}\sum_{i=1}^{n} \hat{\varepsilon}_i^2 = \frac{\text{RSS}}{n - p - 1}$$

> 分母 $n - p - 1$ 是自由度（$n$ 个观测减去 $p+1$ 个参数）。

## 6.4 Multicollinearity（多重共线性）

当 regressors 之间高度相关时：

- $X^T X$ 的 condition number 很大（接近 singular）
- $\hat{\beta}$ 的方差爆炸（$\text{Var}(\hat{\beta}) = \sigma^2 (X^TX)^{-1}$ 中 $(X^TX)^{-1}$ 很大）
- 系数估计不稳定：加减一个观测值就会大幅改变 $\hat{\beta}$

> 本质问题：模型无法区分"是 $X_1$ 的影响还是 $X_2$ 的影响"——因为它们几乎总是一起动。

### 诊断

- 看 correlation matrix of regressors
- Variance Inflation Factor (VIF)
- 如果 $X^T X$ 的 eigenvalues 很小 → 有问题

---

# 7. Maximum Likelihood Estimation (MLE)

## 7.1 Likelihood Function

给定观测 $y_1, \dots, y_n$ 和模型参数 $\theta$：

$$L(\theta) = \prod_{i=1}^{n} f(y_i \,|\, \theta)$$

通常取 log：

$$\ell(\theta) = \sum_{i=1}^{n} \ln f(y_i \,|\, \theta)$$

> MLE 的思想：**选择使 observed data 出现概率最大的参数**。

## 7.2 OLS = MLE under Normal Errors

当 $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ 时：

$$f(y_i | x_i, \beta, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}\exp\!\left(-\frac{(y_i - x_i^T\beta)^2}{2\sigma^2}\right)$$

取 log-likelihood：

$$\ell(\beta, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - x_i^T\beta)^2$$

对 $\beta$ 求最大化：

> 只有最后一项含 $\beta$，且带负号 → 最大化 $\ell$ = 最小化 $\sum(y_i - x_i^T\beta)^2$ = OLS！

$$\boxed{\text{OLS} = \text{MLE under Gaussian errors}}$$

## 7.3 MLE with t-Distributed Errors

如果 error 服从 $t$ 分布（heavy tails）：

$$\varepsilon_i \sim t(\nu, 0, \sigma)$$

则 log-likelihood 变成：

$$\ell(\beta, \sigma, \nu) = \sum_{i=1}^{n} \ln f_t\!\left(\frac{y_i - x_i^T\beta}{\sigma};\, \nu\right) - n\ln\sigma$$

其中 $f_t(\cdot\,; \nu)$ 是标准 $t$ 分布 density。

> 此时 **不能** 用 OLS 公式，需要数值优化（Newton-Raphson / `scipy.optimize`）同时 fit $\beta, \sigma, \nu$。

这在实际金融数据中更常用，因为 returns 通常有 [[Chapter 1 - Foundations Rebuild#7. Student's t Distribution|heavy tails]]（excess kurtosis > 0）。

## 7.4 AIC 与 AICc — 模型选择

### AIC (Akaike Information Criterion)

$$\text{AIC} = -2\ell(\hat{\theta}) + 2k$$

- $\ell(\hat{\theta})$：最大 log-likelihood
- $k$：参数个数
- AIC 越小越好

### AICc — 小样本校正

$$\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1}$$

> 当 $n$ 远大于 $k$ 时 AICc $\approx$ AIC；当 $n$ 不够大时，AICc 对复杂模型惩罚更重。

### 使用原则

- 比较两个模型：先算各自的 AICc，**AICc 更小的 = better fit**
- 常见场景：Normal vs $t$（$k_{\text{Normal}} = 2$, $k_t = 3$）
- $t$ 模型多一个参数 $\nu$，但如果 heavy tail 确实存在，log-likelihood 提升足以补偿

---

# 8. Time Series — 时间维度的依赖

## 8.1 核心问题

前面讨论的是 cross-sectional dependence（不同资产之间）。

Time series 关心：

> **今天的值是否依赖于昨天的值？**

Autocorrelation 衡量这种 temporal dependence。

## 8.2 Autocorrelation Function (ACF)

$$\rho(h) = \text{Corr}(X_t, X_{t+h}) = \frac{\text{Cov}(X_t, X_{t+h})}{\text{Var}(X_t)}$$

- $h = 1$：lag-1 autocorrelation（今天 vs 昨天）
- $h = 2$：lag-2（今天 vs 前天）
- 要求 **stationarity**：$\rho(h)$ 只取决于 lag $h$，不取决于时间 $t$

## 8.3 AR(p) 模型 — Autoregressive

### AR(1)

$$X_t = c + \phi_1 X_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \text{WN}(0, \sigma^2)$$

### AR(p)

$$X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + \varepsilon_t$$

### Stationarity 条件

AR(1) stationary $\Leftrightarrow$ $|\phi_1| < 1$

> 如果 $\phi_1 = 1$ → random walk（non-stationary），方差随时间 $\to \infty$。

一般 AR(p)：特征方程 $1 - \phi_1 z - \cdots - \phi_p z^p = 0$ 的所有根的模 $> 1$。

### AR(1) 的 ACF

$$\rho(h) = \phi_1^h$$

> ACF **指数衰减**。$\phi_1$ 越接近 1 → 衰减越慢 → memory 越长。

### AR(1) 的 unconditional moments

$$E[X_t] = \frac{c}{1 - \phi_1}$$

$$\text{Var}(X_t) = \frac{\sigma^2}{1 - \phi_1^2}$$

**推导 unconditional mean**：

取 $E[X_t] = c + \phi_1 E[X_{t-1}] + 0$，在 stationarity 下 $E[X_t] = E[X_{t-1}] = \mu$：

$$\mu = c + \phi_1 \mu \;\Rightarrow\; \mu = \frac{c}{1 - \phi_1}$$

**推导 unconditional variance**：

$$\text{Var}(X_t) = \phi_1^2 \text{Var}(X_{t-1}) + \sigma^2$$

在 stationarity 下 $\text{Var}(X_t) = \text{Var}(X_{t-1}) = \gamma_0$：

$$\gamma_0 = \phi_1^2 \gamma_0 + \sigma^2 \;\Rightarrow\; \gamma_0 = \frac{\sigma^2}{1 - \phi_1^2}$$

## 8.4 MA(q) 模型 — Moving Average

### MA(1)

$$X_t = c + \varepsilon_t + \theta_1 \varepsilon_{t-1}$$

### MA(q)

$$X_t = c + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}$$

### 关键性质

- MA(q) 的 ACF 在 **lag $q$ 之后截断为 0**
  - MA(1)：只有 $\rho(1) \ne 0$，$\rho(h) = 0$ for $h \ge 2$
  - MA(q)：$\rho(h) = 0$ for $h > q$
- MA 过程 **总是 stationary**（无需条件）

### MA(1) 的 ACF

$$\rho(1) = \frac{\theta_1}{1 + \theta_1^2}, \quad \rho(h) = 0 \text{ for } h \ge 2$$

> AR 的 ACF 缓慢衰减，MA 的 ACF 突然截断 — 这是区分它们的关键。

## 8.5 PACF — Partial Autocorrelation Function

$$\text{PACF}(h) = \text{Corr}(X_t, X_{t+h} \,|\, X_{t+1}, \dots, X_{t+h-1})$$

> PACF 是 **控制了中间 lags 之后**的"纯" lag-$h$ 相关。

### 用 ACF + PACF 识别模型

| 模型 | ACF 特征 | PACF 特征 |
|---|---|---|
| AR(p) | 缓慢衰减（exponential / damped oscillation） | **在 lag $p$ 后截断** |
| MA(q) | **在 lag $q$ 后截断** | 缓慢衰减 |
| ARMA(p,q) | 都缓慢衰减 | 都缓慢衰减 |

> 实践中经常用 AICc 自动选阶，但 ACF/PACF 图给你直觉上的 sanity check。

## 8.6 Model Selection by AICc

对多个候选阶数 $p = 1, 2, \dots, p_{\max}$，fit 每个 AR(p)，计算 AICc，选最小的。

```
p=1: AICc = -1520.3
p=2: AICc = -1523.1  ← best
p=3: AICc = -1521.8
```

> 选 $p = 2$：it balances goodness-of-fit and complexity。

---

# 9. GARCH — Volatility 也有 Memory

## 9.1 为什么需要 GARCH？

OLS 和 standard time series 假设 **constant variance**（homoskedasticity）。

但金融数据中的 stylized fact：

> **Volatility clustering**：大波动之后往往跟着大波动，小波动之后跟着小波动。

GARCH 让方差本身变成一个 time-varying process。和 [[Chapter 3 - Financial Data & Monte Carlo Simulation#6. Exponentially Weighted Covariance（指数加权协方差）|Exponential Weighting]] 的思想类似：都强调近期数据更重要。

## 9.2 GARCH(1,1) 模型

$$r_t = \mu + \varepsilon_t, \quad \varepsilon_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0,1)$$

$$\boxed{\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2}$$

参数解读：

| 参数 | 含义 |
|---|---|
| $\omega > 0$ | 基础 variance level |
| $\alpha \ge 0$ | 对 **昨天 shock 平方**的反应（short-term） |
| $\beta \ge 0$ | 对 **昨天方差**的持续（persistence） |
| $\alpha + \beta$ | 总的 persistence（越接近 1 → volatility cluster 越持久） |

### 条件

- $\omega > 0, \; \alpha \ge 0, \; \beta \ge 0$
- $\alpha + \beta < 1$（stationarity condition）

## 9.3 Unconditional Variance 推导

在 stationarity 下，$E[\sigma_t^2] = E[\sigma_{t-1}^2] = \bar{\sigma}^2$：

$$\bar{\sigma}^2 = \omega + \alpha \, E[\varepsilon_{t-1}^2] + \beta \, \bar{\sigma}^2$$

因为 $E[\varepsilon_{t-1}^2] = E[\sigma_{t-1}^2 z_{t-1}^2] = \bar{\sigma}^2 \cdot E[z^2] = \bar{\sigma}^2$：

$$\bar{\sigma}^2 = \omega + \alpha \bar{\sigma}^2 + \beta \bar{\sigma}^2 = \omega + (\alpha + \beta)\bar{\sigma}^2$$

$$\bar{\sigma}^2 (1 - \alpha - \beta) = \omega$$

$$\boxed{\bar{\sigma}^2 = \frac{\omega}{1 - \alpha - \beta}}$$

> 长期平均方差由 $\omega$ 和 persistence $(\alpha + \beta)$ 共同决定。

### 数值例子

$\omega = 0.00001, \; \alpha = 0.06, \; \beta = 0.93$：

$$\bar{\sigma}^2 = \frac{0.00001}{1 - 0.06 - 0.93} = \frac{0.00001}{0.01} = 0.001$$

$$\bar{\sigma} = \sqrt{0.001} \approx 3.16\%$$

Persistence $= 0.99$，非常接近 1 → volatility shocks 衰减很慢。

## 9.4 直觉总结

$$\sigma_t^2 = \underbrace{\omega}_{\text{floor}} + \underbrace{\alpha \varepsilon_{t-1}^2}_{\text{react to yesterday's shock}} + \underbrace{\beta \sigma_{t-1}^2}_{\text{carry over yesterday's vol}}$$

- **Big shock yesterday** ($\varepsilon_{t-1}^2$ large) → $\sigma_t^2$ jumps → more uncertain tomorrow
- **Calm period** ($\varepsilon_{t-1}^2$ small) → $\sigma_t^2$ stays low
- **$\beta$ close to 1** → yesterday's vol level carries forward almost unchanged

> 这就是 risk management 中"regimes"的数学基础：高 vol 和低 vol 各自持续一段时间。

---

# 10. Chapter Summary — 全章知识地图

| 概念 | 核心问题 | 关键公式 |
|---|---|---|
| Covariance | 两变量是否一起动？ | $\text{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)]$ |
| Correlation | 归一化的共动程度 | $\rho = \text{Cov}/(σ_X σ_Y)$ |
| Pearson vs Spearman | 线性 vs 单调 | Spearman = Pearson on ranks |
| Cov Matrix | Portfolio risk 的数学载体 | $\text{Var}(\mathbf{w}^T\mathbf{X}) = \mathbf{w}^T\Sigma\mathbf{w}$ |
| MVN | 多变量联合 Gaussian | $f(\mathbf{x}) \propto \exp(-\frac{1}{2}Q)$ |
| Conditional | 观测后 belief 如何更新 | $\mu_{2|1} = \mu_2 + \frac{\sigma_{12}}{\sigma_1^2}(x_1 - \mu_1)$ |
| OLS | 最小化残差平方和 | $\hat\beta = (X^TX)^{-1}X^TY$ |
| MLE | 选择使数据最可能的参数 | $\hat\theta = \arg\max \ell(\theta)$ |
| AICc | 模型选择（fit vs complexity） | $\text{AIC} + \frac{2k(k+1)}{n-k-1}$ |
| AR(p) | 今天依赖过去 $p$ 天 | ACF 缓慢衰减，PACF 在 $p$ 后截断 |
| MA(q) | 今天依赖过去 $q$ 个 shock | ACF 在 $q$ 后截断 |
| GARCH(1,1) | Volatility 有 memory | $\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2$ |
