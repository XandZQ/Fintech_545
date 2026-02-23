# Chapter 5 — Advanced VaR & Expected Shortfall

> 这一章从 [[Chapter 4 - Value-at-Risk (VaR)|Chapter 4]] 的基础出发，深入两个方向：(1) **ES 的严格推导与性质**；(2) **打破正态假设** — 引入 Copula 来建模非线性依赖结构。

---

# 1. Expected Shortfall（回顾与深化）

## 1.1 回顾 ES 的定义

Expected Shortfall 回答的问题：

> **一旦损失超过 VaR，平均还会亏多少？**

形式化定义：

$$\text{ES}_\alpha = -\frac{1}{1-\alpha}\int_0^{1-\alpha} F^{-1}(p)\,dp$$

其中 $F^{-1}(p)$ 是 return distribution 的 quantile function。

### 等价写法（条件期望形式）

$$\text{ES}_\alpha = -E[R \mid R \le -\text{VaR}_\alpha]$$

直觉：

- VaR 画了一条线："你最多亏这么多（在 $\alpha$ 概率下）"
- ES 看线以下的区域："跌破了这条线之后，平均深度是多少？"

> ES 永远 $\ge$ VaR，因为它是 tail 的**平均**，而 VaR 只是 tail 的**入口**。

---

## 1.2 Normal Distribution 下的 ES（完整推导）

假设 $R \sim N(\mu, \sigma^2)$，我们要求 $E[R \mid R \le r^*]$，其中 $r^* = -\text{VaR}_\alpha = \mu - z_\alpha\sigma$。

**Step 1**：条件期望展开

$$E[R \mid R \le r^*] = \frac{\int_{-\infty}^{r^*} r \cdot f(r)\,dr}{P(R \le r^*)}$$

分母 $= F(r^*) = 1 - \alpha$。

**Step 2**：标准化变换

令 $z = \frac{r - \mu}{\sigma}$，则 $r = \mu + \sigma z$，$dr = \sigma\,dz$。

上限变为：$z^* = \frac{r^* - \mu}{\sigma} = -z_\alpha$

分子：

$$\int_{-\infty}^{-z_\alpha} (\mu + \sigma z) \cdot \frac{1}{\sigma}\phi(z) \cdot \sigma\,dz = \int_{-\infty}^{-z_\alpha} (\mu + \sigma z)\,\phi(z)\,dz$$

拆成两项：

$$= \mu\int_{-\infty}^{-z_\alpha}\phi(z)\,dz + \sigma\int_{-\infty}^{-z_\alpha} z\,\phi(z)\,dz$$

第一项 $= \mu \cdot \Phi(-z_\alpha) = \mu(1-\alpha)$。

**Step 3**：求解关键积分 $\int_{-\infty}^{c} z\,\phi(z)\,dz$

核心观察：$\phi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$，所以：

$$\frac{d\phi}{dz} = -z\phi(z) \quad \Longrightarrow \quad z\phi(z) = -\phi'(z)$$

因此：

$$\int_{-\infty}^{c} z\,\phi(z)\,dz = \int_{-\infty}^{c} -\phi'(z)\,dz = -\bigl[\phi(z)\bigr]_{-\infty}^{c} = -\phi(c) + 0 = -\phi(c)$$

代入 $c = -z_\alpha$：

$$\sigma \cdot (-\phi(-z_\alpha)) = -\sigma\phi(z_\alpha)$$

（利用对称性 $\phi(-z) = \phi(z)$）

**Step 4**：合并

$$E[R \mid R \le r^*] = \frac{\mu(1-\alpha) - \sigma\phi(z_\alpha)}{1-\alpha} = \mu - \frac{\sigma\phi(z_\alpha)}{1-\alpha}$$

因此：

$$\boxed{\text{ES}_\alpha = -E[R \mid R \le r^*] = -\mu + \frac{\sigma\phi(z_\alpha)}{1-\alpha}}$$

### 数值验证

取 $\alpha = 0.95$，$z_{0.95} = 1.645$，$\phi(1.645) \approx 0.1031$：

$$\frac{\text{ES}}{\sigma} = \frac{0.1031}{0.05} = 2.063 \quad \text{vs} \quad \frac{\text{VaR}}{\sigma} = 1.645$$

> ES 比 VaR 大约高 25% — 这就是 tail 的"平均深度"比"入口"更深。

---

## 1.3 Discrete (Historical Simulation) 下的 ES

当我们用历史数据（或 Monte Carlo samples）时，没有解析公式，直接用**经验平均**。

### 步骤

1. 将所有 simulated/historical portfolio returns 排序：$r_{(1)} \le r_{(2)} \le \cdots \le r_{(T)}$
2. 找 VaR cutoff index：$k = \lfloor T(1-\alpha) \rfloor$
3. ES = 最差的 $k$ 个 returns 的平均值（取负）：

$$\boxed{\text{ES}_\alpha = -\frac{1}{k}\sum_{i=1}^{k} r_{(i)}}$$

### 直觉

- VaR = 排序后的第 $k$ 个值
- ES = 第 1 到第 $k$ 个值的**平均**

> 如果 tail 里有一个极端 outlier，VaR 可能完全不变（因为只看第 $k$ 个），但 ES 会被拉高 — 这就是 ES 对 tail risk 更敏感的原因。

---

# 2. Model-Based Simulation（从拟合模型中做 Monte Carlo）

## 2.1 核心思路

在 [[Chapter 3 - Financial Data & Monte Carlo Simulation#1. 为什么需要 Monte Carlo？|Chapter 3]] 中我们学了 Monte Carlo 的底层工具。现在我们把它应用到 **fitted models**：

> 不直接 simulate raw returns，而是 simulate model 的 **residuals**，然后通过 model 重构 returns。

### 一般框架

假设我们对 asset return $Y$ 建了一个回归模型：

$$Y = X\beta + \epsilon$$

其中：

- $X$ = 解释变量（factors）
- $\beta$ = 回归系数（已估计）
- $\epsilon$ = residual（需要 simulate）

### 模拟步骤

1. **估计 model**：用历史数据做 OLS，得到 $\hat{\beta}$ 和 residuals $\hat{\epsilon}_t$
2. **Fit residual distribution**：对 $\hat{\epsilon}$ 拟合一个分布（normal、t 等）
3. **Simulate residuals**：从拟合的分布中抽取 $K$ 个 $\epsilon^{(k)}$
4. **Reconstruct returns**：$Y^{(k)} = X_{\text{next}}\hat{\beta} + \epsilon^{(k)}$
5. **计算 portfolio PnL** 并取 quantile → VaR / ES

---

## 2.2 单变量情况（Simple OLS + Normal Errors）

### Setup

$$Y_t = \beta_0 + \beta_1 X_t + \epsilon_t, \quad \epsilon_t \sim N(0, \sigma_\epsilon^2)$$

### Simulation

给定明天的 factor value $X_{t+1}$：

1. $\hat{Y}_{t+1} = \hat{\beta}_0 + \hat{\beta}_1 X_{t+1}$（确定性部分）
2. Simulate：$\epsilon^{(k)} \sim N(0, \hat{\sigma}_\epsilon^2)$，$k = 1, \dots, K$
3. $Y_{t+1}^{(k)} = \hat{Y}_{t+1} + \epsilon^{(k)}$

> 本质上：**model 给出 center，residual 给出 dispersion**。

---

## 2.3 多变量情况（Joint Simulation）

### 问题

当 portfolio 有多个 assets，每个 asset 有自己的 model：

$$Y_{i,t} = X_{i,t}\beta_i + \epsilon_{i,t}, \quad i = 1, \dots, n$$

**关键问题**：不同 asset 的 residuals 可能是 **correlated** 的！

### 解决方案：Joint Normal Residuals

假设 residual vector $\epsilon_t = (\epsilon_{1,t}, \dots, \epsilon_{n,t})'$ 服从：

$$\epsilon_t \sim N(0, \Sigma_\epsilon)$$

其中 $\Sigma_\epsilon$ 是 residual 的 covariance matrix。

### Simulation 步骤

**Step 1**：估计每个 asset 的 model → 得到 residuals $\hat{\epsilon}_{i,t}$

**Step 2**：计算 residual covariance matrix

$$\hat{\Sigma}_\epsilon = \frac{1}{T-1}\sum_{t=1}^{T}\hat{\epsilon}_t\hat{\epsilon}_t'$$

**Step 3**：Cholesky decomposition（在 [[Chapter 3 - Financial Data & Monte Carlo Simulation#4. Cholesky Factorization（如何求 $L$）|Chapter 3]] 中已详细推导）

$$\hat{\Sigma}_\epsilon = LL'$$

**Step 4**：Simulate correlated residuals

$$\epsilon^{(k)} = LZ^{(k)}, \quad Z^{(k)} \sim N(0, I)$$

**Step 5**：Reconstruct each asset's return

$$Y_i^{(k)} = X_{i,\text{next}}\hat{\beta}_i + \epsilon_i^{(k)}$$

**Step 6**：Portfolio return

$$R_p^{(k)} = w'Y^{(k)}$$

**Step 7**：对 $\{R_p^{(1)}, \dots, R_p^{(K)}\}$ 取 quantile → VaR，取 tail mean → ES

---

## 2.4 Multiple Independent Variables（多因子回归）

当单个 asset 的 model 有多个 factors 时：

$$Y_t = \beta_0 + \beta_1 X_{1,t} + \beta_2 X_{2,t} + \cdots + \beta_p X_{p,t} + \epsilon_t$$

### 两种 approach

**Approach A：只 simulate residuals**

- 假设明天的 $X_{1}, X_2, \dots, X_p$ 是 **已知的**（或用当前值代替）
- 只 simulate $\epsilon$
- 简单，但忽略了 factor uncertainty

**Approach B：同时 simulate factors 和 residuals**

- 给 factors 也建模（e.g., AR(1) 或 random walk）
- Simulate factors 和 residuals 的 **joint distribution**
- 更 realistic，但更复杂

### Joint Simulation 的完整 setup

定义 augmented error vector：

$$\eta_t = \begin{pmatrix} \epsilon_{Y,t} \\ \epsilon_{X_1,t} \\ \epsilon_{X_2,t} \\ \vdots \end{pmatrix} \sim N(0, \Sigma_\eta)$$

然后用 Cholesky 对 $\Sigma_\eta$ 做分解，simulate $\eta$，通过各自的 model 重构 $Y$ 和 $X$。

> ⚠️ **依赖顺序很重要**：如果 $Y$ 的 model 需要 $X_1$，那必须先 simulate $X_1$，再 simulate $Y$。

---

# 3. Breaking the Normality Assumption — Copulas

## 3.1 为什么需要 Copula？

到目前为止我们一直假设：

> Returns（或 residuals）服从 **multivariate normal distribution**

这有两个严重问题：

1. **Marginals 必须是 normal** — 但实际上各 asset 可能有不同的分布（fat tails, skew）
2. **Dependence structure 只能是 linear correlation** — 但实际的依赖关系可能是非线性的（e.g., tail dependence：crash 时所有资产同时暴跌）

> Copula 的核心思想：**把 marginal distributions 和 dependence structure 分开建模**。

---

## 3.2 Copula 的直觉

一个 joint distribution 包含两种信息：

| 信息 | 描述 | 例子 |
|---|---|---|
| **Marginals** | 每个变量自己的分布 | Stock A 服从 t-distribution，Stock B 服从 normal |
| **Dependence** | 变量之间如何 "连接" | Crash 时是否同时下跌？ |

Copula 就是那个 **"连接函数"**：

> Copula = 一个定义在 $[0,1]^n$ 上的 joint CDF，marginals 全是 Uniform(0,1)。

---

## 3.3 [[Sklar's Theorem]]（Copula 理论的基石）

### 定理陈述

对于任意 joint CDF $F(x_1, x_2, \dots, x_n)$，如果 marginal CDFs $F_1, F_2, \dots, F_n$ 是连续的，则 **存在唯一的 copula $C$** 使得：

$$\boxed{F(x_1, x_2, \dots, x_n) = C\bigl(F_1(x_1), F_2(x_2), \dots, F_n(x_n)\bigr)}$$

### 逐步理解

**Step 1**：每个 $F_i(x_i) \in [0,1]$，就是把 $x_i$ 变换到 "概率空间"

**Step 2**：$C$ 接收这些概率值，输出 joint probability

**Step 3**：反过来说，如果你知道了各 marginal 和 copula，就能完全重建 joint distribution

### 直觉比喻

> 想象做一道菜：**Marginals = 食材**（各种分布），**Copula = 食谱**（怎么把它们混在一起）。同样的食材，不同的食谱，做出完全不同的菜。

### 为什么这很强大？

- 你可以给 Stock A 用 t-distribution（fat tails）
- 给 Stock B 用 normal distribution
- 然后用一个 copula 来描述它们的 dependence
- **Marginals 和 dependence 完全独立选择**

---

## 3.4 数学推导：从 Sklar's Theorem 到 Simulation

### Probability Integral Transform（PIT）

如果 $X \sim F_X$（任意连续分布），则：

$$U = F_X(X) \sim \text{Uniform}(0,1)$$

**推导**：

$$P(U \le u) = P(F_X(X) \le u) = P(X \le F_X^{-1}(u)) = F_X(F_X^{-1}(u)) = u$$

这就是 Uniform 的 CDF。所以 $U \sim \text{Uniform}(0,1)$。✓

### 反向变换

如果 $U \sim \text{Uniform}(0,1)$，则：

$$X = F_X^{-1}(U) \sim F_X$$

> 这两个变换是 copula simulation 的关键工具：**任何分布都可以通过 Uniform 来 "转换"**。

---

# 4. Gaussian Copula（高斯 Copula）

## 4.1 定义

Gaussian Copula 用 **multivariate normal** 的 dependence structure，但允许 marginals 是任意分布。

$$C_{\text{Gauss}}(u_1, u_2, \dots, u_n; \Gamma) = \Phi_n\bigl(\Phi^{-1}(u_1), \Phi^{-1}(u_2), \dots, \Phi^{-1}(u_n); \Gamma\bigr)$$

其中：

- $\Phi^{-1}$：标准正态的 inverse CDF（把 uniform 变量转回 normal 空间）
- $\Phi_n(\cdot; \Gamma)$：$n$ 维标准正态的 joint CDF，correlation matrix 为 $\Gamma$
- $u_i \in [0,1]$：各 marginal 的概率值

### 逐步拆解

1. 你有 $u_1, u_2, \dots, u_n$（每个都是 Uniform）
2. 通过 $\Phi^{-1}$ 把它们映射到 normal 空间：$z_i = \Phi^{-1}(u_i)$
3. 在 normal 空间里，用 correlation matrix $\Gamma$ 描述它们的 joint behavior
4. 计算 joint probability

> 本质上，Gaussian Copula 说的是："**在概率空间里，变量的 dependence 结构跟 multivariate normal 一样**"。

---

## 4.2 Gaussian Copula 的 Fitting（拟合步骤）

### 完整流程

**Step 1：Fit Marginals（拟合各 marginal distribution）**

对每个 asset $i$，选择并拟合一个 marginal distribution $\hat{F}_i$。

可以是：

- Empirical CDF（非参数）
- Normal distribution
- Student's t distribution
- 或其他任何适合的分布

**Step 2：Transform to Uniform（概率积分变换）**

$$u_{i,t} = \hat{F}_i(x_{i,t})$$

将每个 asset 的历史 returns 转换为 $[0,1]$ 上的 uniform 值。

> 如果 marginal 拟合得好，$u_{i,t}$ 应该近似服从 $\text{Uniform}(0,1)$。

**Step 3：Transform to Normal（映射到正态空间）**

$$z_{i,t} = \Phi^{-1}(u_{i,t})$$

现在所有变量都在标准正态空间里了。

**Step 4：Estimate Correlation Matrix $\hat{\Gamma}$**

在 normal 空间里计算 **Spearman rank correlation**（不是 Pearson！原因见下节）：

$$\hat{\Gamma} = \text{Corr}(z_1, z_2, \dots, z_n)$$

> ⚠️ 这里用的是 transformed data 的 correlation，不是原始 returns 的 correlation。

---

## 4.3 为什么用 Spearman Correlation？

### Pearson vs Spearman 回顾（[[Chapter 2 - Multivariate and Regression#2.2 Spearman (Rank) Correlation|Chapter 2]] 的扩展）

| | Pearson | Spearman |
|---|---|---|
| 衡量什么 | Linear relationship | Monotonic relationship |
| 对 outliers | 敏感 | 稳健 |
| 对 marginal 变换 | **会变** | **不变** |

最后一点是关键：

### Spearman Correlation 对 monotonic 变换不变

**推导**：

Spearman correlation 的定义是 **ranks 的 Pearson correlation**。

如果 $g$ 是 monotonically increasing function，则 $\text{rank}(g(X_i)) = \text{rank}(X_i)$。

因此：

$$\rho_S(g(X), h(Y)) = \rho_S(X, Y)$$

对任意 monotonically increasing $g, h$。

### 为什么这对 Copula 重要？

在 copula fitting 中，我们做了 $X \to U = F(X) \to Z = \Phi^{-1}(U)$。

这是一系列 **monotonic transformations**。

- 如果用 **Pearson**：correlation 会在每次变换后改变 → 不 consistent
- 如果用 **Spearman**：correlation 在变换前后不变 → 可以直接用原始数据计算

> 实际操作中的简化：你可以跳过 Step 2 和 Step 3，**直接用原始 returns 的 Spearman correlation** 作为 $\hat{\Gamma}$。

---

## 4.4 Spearman Correlation 的计算

### 步骤

1. 对每个变量 $X_i$，把数据替换为 **ranks**：$R_{i,t} = \text{rank}(x_{i,t})$
2. 计算 **ranks 之间的 Pearson correlation**：

$$\rho_S(X, Y) = \rho_{\text{Pearson}}(\text{rank}(X), \text{rank}(Y))$$

### 展开公式

$$\rho_S = \frac{\sum_{t=1}^T (R_{X,t} - \bar{R}_X)(R_{Y,t} - \bar{R}_Y)}{\sqrt{\sum_{t=1}^T (R_{X,t} - \bar{R}_X)^2}\sqrt{\sum_{t=1}^T (R_{Y,t} - \bar{R}_Y)^2}}$$

其中 $\bar{R}_X = \frac{T+1}{2}$（ranks 的平均值）。

### 简化公式（无 ties 时）

如果没有 ties（即所有值不同），可以用经典公式：

$$\rho_S = 1 - \frac{6\sum_{t=1}^T d_t^2}{T(T^2-1)}$$

其中 $d_t = R_{X,t} - R_{Y,t}$ 是 rank differences。

---

# 5. Simulating from the Gaussian Copula（完整流程）

## 5.1 Simulation 算法

给定：fitted marginals $\hat{F}_1, \dots, \hat{F}_n$ 和 correlation matrix $\hat{\Gamma}$。

**Step 1：Cholesky Decomposition**

$$\hat{\Gamma} = LL'$$

（[[Chapter 3 - Financial Data & Monte Carlo Simulation#4.2 Cholesky 算法（逐列求解，完整公式）|Chapter 3 已详细介绍 Cholesky 算法]]）

**Step 2：Generate Independent Standard Normals**

$$Z^{(k)} = (Z_1^{(k)}, \dots, Z_n^{(k)})' \sim N(0, I), \quad k = 1, \dots, K$$

**Step 3：Induce Correlation**

$$\tilde{Z}^{(k)} = LZ^{(k)}$$

现在 $\tilde{Z}^{(k)} \sim N(0, \hat{\Gamma})$。

**Step 4：Transform to Uniform**

$$u_i^{(k)} = \Phi(\tilde{Z}_i^{(k)})$$

每个 $u_i^{(k)} \in [0,1]$，且 dependence structure 由 $\hat{\Gamma}$ 决定。

**Step 5：Transform to Original Marginals（反变换）**

$$x_i^{(k)} = \hat{F}_i^{-1}(u_i^{(k)})$$

现在 $x_i^{(k)}$ 服从 $\hat{F}_i$（各自的 marginal），且相互之间的 dependence 由 Gaussian Copula 决定。

### 流程图

$$Z \xrightarrow{L} \tilde{Z} \xrightarrow{\Phi} U \xrightarrow{F_i^{-1}} X$$

$$\text{独立正态} \to \text{相关正态} \to \text{相关 Uniform} \to \text{原始分布}$$

---

## 5.2 为什么这比 Multivariate Normal 更好？

| | Multivariate Normal | Gaussian Copula |
|---|---|---|
| Marginals | 必须全是 Normal | **任意分布** |
| Dependence | Linear correlation | Linear correlation (在 normal 空间) |
| Fat tails | 不支持 | **通过 marginals 实现** |
| Skewness | 不支持 | **通过 marginals 实现** |

> Gaussian Copula 本质上是 multivariate normal 的 **推广**：如果所有 marginals 都选 normal，Gaussian Copula 就退化为 multivariate normal。

### 局限性

Gaussian Copula 仍然有一个弱点：**tail dependence 为零**。

> 在极端情况下（e.g., 金融危机），assets 之间的 dependence 往往比 normal 时更强 — 这叫 **tail dependence**。Gaussian Copula 无法捕获这一点。如果需要 tail dependence，可以用 **t-Copula** 或 **Clayton Copula** 等。

---

# 6. Putting It All Together — 完整 VaR/ES 计算流程

## 6.1 使用 Gaussian Copula 的 End-to-End 流程

### Given

- $n$ 个 assets，历史 returns $\{r_{i,t}\}$
- Portfolio weights $w$

### Phase 1：Fit

1. 对每个 asset $i$，fit marginal distribution $\hat{F}_i$（e.g., t-distribution with $\hat{\nu}_i$ degrees of freedom）
2. 计算 Spearman correlation matrix $\hat{\Gamma}$ from raw returns
3. Cholesky decompose：$\hat{\Gamma} = LL'$

### Phase 2：Simulate

4. Generate $K$ independent standard normal vectors：$Z^{(k)} \sim N(0, I)$
5. $\tilde{Z}^{(k)} = LZ^{(k)}$（induce correlation）
6. $u_i^{(k)} = \Phi(\tilde{Z}_i^{(k)})$（to uniform）
7. $r_i^{(k)} = \hat{F}_i^{-1}(u_i^{(k)})$（to original marginal）

### Phase 3：Compute Risk Metrics

8. Portfolio return：$R_p^{(k)} = w' r^{(k)}$
9. Sort $\{R_p^{(1)}, \dots, R_p^{(K)}\}$
10. **VaR**：$\text{VaR}_\alpha = -R_p^{(\lfloor K(1-\alpha) \rfloor)}$
11. **ES**：$\text{ES}_\alpha = -\frac{1}{\lfloor K(1-\alpha) \rfloor}\sum_{i=1}^{\lfloor K(1-\alpha) \rfloor} R_p^{(i)}$

---

## 6.2 Example（Slides 中的综合例题）

### Setup

- 2 个 assets
- Asset 1：$r_1 \sim t(\nu_1)$（fat tails）
- Asset 2：$r_2 \sim N(\mu_2, \sigma_2^2)$
- Spearman correlation = 0.6
- Equal weights：$w = (0.5, 0.5)'$

### 执行

1. Fit marginals → $\hat{F}_1 = t(\hat{\nu}_1, \hat{\mu}_1, \hat{\sigma}_1)$，$\hat{F}_2 = N(\hat{\mu}_2, \hat{\sigma}_2^2)$
2. $\hat{\Gamma} = \begin{pmatrix} 1 & 0.6 \\ 0.6 & 1 \end{pmatrix}$
3. Cholesky：$L = \begin{pmatrix} 1 & 0 \\ 0.6 & \sqrt{1-0.36} \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0.6 & 0.8 \end{pmatrix}$
4. Simulate $K = 100{,}000$ scenarios
5. 对每个 scenario 做 $Z \to \tilde{Z} \to U \to r \to R_p$
6. 取 quantile 和 tail mean

> 这个流程可以处理任意数量的 assets、任意 marginal distributions，是现代 risk management 的标准工具。

---

# 7. Key Formulas 速查

$$\text{ES}_\alpha^{\text{normal}} = -\mu + \frac{\sigma\phi(z_\alpha)}{1-\alpha}$$

$$\text{ES}_\alpha^{\text{discrete}} = -\frac{1}{k}\sum_{i=1}^{k}r_{(i)}, \quad k = \lfloor T(1-\alpha)\rfloor$$

$$F(x_1, \dots, x_n) = C\bigl(F_1(x_1), \dots, F_n(x_n)\bigr) \quad \text{(Sklar's Theorem)}$$

$$C_{\text{Gauss}}(u_1, \dots, u_n) = \Phi_n\bigl(\Phi^{-1}(u_1), \dots, \Phi^{-1}(u_n); \Gamma\bigr)$$

$$\rho_S = 1 - \frac{6\sum d_t^2}{T(T^2-1)} \quad \text{(Spearman, no ties)}$$

$$\text{Simulation: } Z \xrightarrow{L} \tilde{Z} \xrightarrow{\Phi} U \xrightarrow{F_i^{-1}} X$$

---

# 8. 与前几章的联系

- **[[Chapter 1 - Foundations Rebuild|Chapter 1]]**：[[Chapter 1 - Foundations Rebuild#3. Quantile Function（分位数 = Inverse CDF）|Quantile function]]、[[Chapter 1 - Foundations Rebuild#2. PDF 与 CDF — 描述分布的两种视角|CDF]] → VaR 和 ES 的数学基础；[[Chapter 1 - Foundations Rebuild#7. Student's t Distribution|t 分布]] → marginal fitting
- **[[Chapter 2 - Multivariate and Regression|Chapter 2]]**：[[Chapter 2 - Multivariate and Regression#3. Covariance Matrix — Portfolio Risk 的核心|Covariance matrix]]、[[Chapter 2 - Multivariate and Regression#2.2 Spearman (Rank) Correlation|Pearson vs Spearman]] → Copula fitting 中 Spearman 的选择；[[Chapter 2 - Multivariate and Regression#6. Regression（回归分析）|Regression]] → model-based simulation 的基础
- **[[Chapter 3 - Financial Data & Monte Carlo Simulation|Chapter 3]]**：[[Chapter 3 - Financial Data & Monte Carlo Simulation#4. Cholesky Factorization（如何求 $L$）|Cholesky decomposition]]、[[Chapter 3 - Financial Data & Monte Carlo Simulation#1. 为什么需要 Monte Carlo？|Monte Carlo]] → Copula simulation 的核心引擎
- **[[Chapter 4 - Value-at-Risk (VaR)|Chapter 4]]**：[[Chapter 4 - Value-at-Risk (VaR)#3. VaR 的计算方法（三大方法）|VaR 的三种计算方法]] → 本章深化为 model-based 和 copula-based simulation；[[Chapter 4 - Value-at-Risk (VaR)#6. Expected Shortfall (ES) — VaR 的补充|ES 作为 VaR 的补充]] → 本章给出完整推导和 [[Chapter 4 - Value-at-Risk (VaR)#7. VaR 的性质：Coherent Risk Measures|coherent risk measure]] 的理论支撑
