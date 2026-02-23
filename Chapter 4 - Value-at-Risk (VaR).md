# Chapter 4 — Value-at-Risk (VaR)

> 这一章是整门课的核心：如何用一个数字回答 **"最坏能亏多少？"**

---

# 1. VaR 的定义与直觉

## 1.1 什么是 VaR？

Value-at-Risk 回答的问题：

> **在给定置信水平 $\alpha$ 和持有期 $\Delta t$ 下，投资组合的最大预期损失是多少？**

形式化定义（基于 loss distribution）：

$$\text{VaR}_\alpha = -F^{-1}(1-\alpha)$$

其中 $F^{-1}$ 是 **return distribution 的 quantile function（分位数函数）**。

### 等价写法

如果我们定义 loss $L = -R$，则：

$$P(L > \text{VaR}_\alpha) = 1 - \alpha$$

直觉：

- 95% VaR = **在 95% 的日子里，你的损失不会超过这个数**
- 剩下 5% 的日子？可能更惨 — 这就是 VaR 的局限

---

## 1.2 图形理解

想象 return distribution 的 PDF：

- 左尾（大额亏损）的面积 = $1 - \alpha$
- VaR 就是那个 **cutoff point**

$$P(R < -\text{VaR}_\alpha) = 1 - \alpha$$

> VaR 本质上就是 **CDF 的反函数**，在 [[Chapter 1 - Foundations Rebuild#3. Quantile Function（分位数 = Inverse CDF）|Chapter 1]] 中我们已经学过 quantile function 了，这里只是应用到了 loss 的语境。

---

# 2. Normal Distribution 下的 VaR（解析解）

## 2.1 单资产情况

假设 return $R \sim N(\mu, \sigma^2)$。

**Step 1**：标准化

$$Z = \frac{R - \mu}{\sigma} \sim N(0,1)$$

**Step 2**：找 quantile

我们要找 $r^*$ 使得 $P(R < r^*) = 1 - \alpha$。

$$P\left(\frac{R - \mu}{\sigma} < \frac{r^* - \mu}{\sigma}\right) = 1 - \alpha$$

$$\Phi\left(\frac{r^* - \mu}{\sigma}\right) = 1 - \alpha$$

其中 $\Phi$ 是标准正态的 CDF。

**Step 3**：求解

$$\frac{r^* - \mu}{\sigma} = \Phi^{-1}(1-\alpha) = -z_\alpha$$

这里 $z_\alpha = \Phi^{-1}(\alpha)$，例如 $z_{0.95} \approx 1.645$，$z_{0.99} \approx 2.326$。

$$r^* = \mu - z_\alpha \cdot \sigma$$

**Step 4**：VaR

$$\boxed{\text{VaR}_\alpha = -(r^*) = -\mu + z_\alpha \cdot \sigma}$$

> 如果 $\mu \approx 0$（短期内常见假设），则简化为：
> $$\text{VaR}_\alpha \approx z_\alpha \cdot \sigma$$

### 数值例子

假设 daily return: $\mu = 0.05\%$，$\sigma = 2\%$，portfolio value $V = \$1{,}000{,}000$

95% VaR:

$$\text{VaR}_{0.95} = -0.0005 + 1.645 \times 0.02 = 0.0324 = 3.24\%$$

Dollar VaR:

$$\text{Dollar VaR} = 0.0324 \times 1{,}000{,}000 = \$32{,}400$$

---

## 2.2 Portfolio VaR（多资产）

### Setup

- $w \in \mathbb{R}^n$：权重向量
- $R \sim N(\mu, \Sigma)$：asset returns
- Portfolio return: $R_p = w'R$

**Step 1**：Portfolio return 的分布

由 multivariate normal 的性质（[[Chapter 2 - Multivariate and Regression#4.2 关键性质|Chapter 2 已推导]]）：

$$R_p = w'R \sim N(w'\mu, \; w'\Sigma w)$$

所以：

$$\mu_p = w'\mu, \quad \sigma_p^2 = w'\Sigma w$$

**Step 2**：直接套用单资产 VaR 公式

$$\boxed{\text{VaR}_\alpha^{\text{portfolio}} = -w'\mu + z_\alpha \cdot \sqrt{w'\Sigma w}}$$

> 这个公式揭示了一个关键事实：**portfolio VaR 不是各资产 VaR 的简单加总**，因为 $\sqrt{w'\Sigma w}$ 包含了 covariance 的效应 — 这就是 diversification。

---

# 3. VaR 的计算方法（三大方法）

Slides 介绍了三种主流方法，各有优劣：

## 3.1 Parametric (Variance-Covariance) Method

### 思路

假设 returns 服从某个已知分布（通常是 normal），用 $\hat{\mu}$ 和 $\hat{\sigma}$ 代入公式。

### 步骤

1. 估计 $\hat{\mu}$ 和 $\hat{\Sigma}$（可以用 equal-weight 或 exponentially-weighted）
2. 计算 $\sigma_p = \sqrt{w'\hat{\Sigma}w}$
3. 计算 VaR：$\text{VaR}_\alpha = -w'\hat{\mu} + z_\alpha \cdot \sigma_p$

### 优点

- 计算速度快（closed-form）
- 容易理解

### 缺点

- **依赖分布假设**（如果 returns 不是 normal 呢？fat tails!）
- **对非线性 payoff 效果差**（e.g., options）

---

## 3.2 Historical Simulation

### 思路

> **不做任何分布假设，直接用历史数据 "重放"**

### 步骤

1. 收集过去 $T$ 天的 return 数据：$\{r_1, r_2, \dots, r_T\}$
2. 对于 portfolio，计算每天的 portfolio return：$r_{p,t} = w'r_t$
3. 将 $\{r_{p,1}, \dots, r_{p,T}\}$ **从小到大排序**
4. 找第 $\lfloor T(1-\alpha) \rfloor$ 个值（或线性插值）

### 数学表达

设排序后的 returns 为 $r_{(1)} \le r_{(2)} \le \cdots \le r_{(T)}$，则：

$$\text{VaR}_\alpha \approx -r_{(\lfloor T(1-\alpha) \rfloor)}$$

### 例子

$T = 1000$ 天，$\alpha = 0.99$

- 第 $\lfloor 1000 \times 0.01 \rfloor = 10$ 小的 return 就是 99% VaR 的 cutoff

### 优点

- 无分布假设（model-free）
- 自动捕获 fat tails、skewness
- 直觉简单

### 缺点

- **完全依赖历史数据**：如果过去没出现过的 scenario，它就看不到
- **对数据量很敏感**：尤其在高置信水平下（99% VaR 只用到最极端的 1% 数据）
- **ghost effect / cliff effect**：一个极端值掉出窗口时，VaR 会突然跳变

---

## 3.3 Monte Carlo Simulation

### 思路

> **人造大量 scenarios，然后像 historical sim 一样取 quantile**

### 步骤

1. 假设 returns 的 model（可以是 normal、t-distribution、GARCH 等）
2. 从 model 中 simulate $K$ 个 scenarios（$K$ 很大，如 10,000+）
3. 计算每个 scenario 的 portfolio PnL
4. 对 PnL 取 quantile

### 与 Historical Sim 的区别

| | Historical Sim | Monte Carlo |
|---|---|---|
| 数据来源 | 真实历史 | 模型生成 |
| 分布假设 | 无 | 有（但可灵活） |
| Scenario 数量 | 受限于历史长度 | 任意多 |
| 非线性 payoff | 可以处理 | 可以处理 |

### 优点

- 极其灵活（任何 model、任何 payoff）
- 可以生成 tail scenarios
- 在 [[Chapter 3 - Financial Data & Monte Carlo Simulation|Chapter 3]] 中我们已经学了 simulation 的底层工具（[[Chapter 3 - Financial Data & Monte Carlo Simulation#4. Cholesky Factorization（如何求 $L$）|Cholesky]]、[[Chapter 3 - Financial Data & Monte Carlo Simulation#10. PCA — 另一种 Simulation 引擎|PCA]]）

### 缺点

- 计算量大
- **Model risk**：garbage model → garbage VaR

---

# 4. Exponentially Weighted VaR（指数加权）

## 4.1 动机

Equal-weight historical sim 把 3 年前和昨天的数据一视同仁 — 这不合理。

**Recent events should matter more.**

## 4.2 权重构造

在 [[Chapter 3 - Financial Data & Monte Carlo Simulation#6. Exponentially Weighted Covariance（指数加权协方差）|Chapter 3]] 中我们已经推导过 exponentially weighted variance：

$$w_{t-i} = (1-\lambda)\lambda^{i-1}$$

归一化后：

$$w_{t-i} \leftarrow \frac{w_{t-i}}{\sum_{j=1}^{n} w_{t-j}}$$

## 4.3 Exponentially Weighted Historical Simulation

### 步骤

1. 给每个历史 return $r_{t-i}$ 赋权重 $w_{t-i}$
2. 将 returns **按大小排序**
3. 从最小的开始**累加权重**，直到累加值达到 $1 - \alpha$
4. 对应的 return 就是 VaR

### 与 Equal-weight 的对比

Equal-weight：每个历史观测权重 = $\frac{1}{T}$，VaR 就是第 $k$ 小的值

Exponentially weighted：近期数据权重大，远期数据权重小，VaR 的 cutoff 对应的是 **加权累积概率** 达到 $1 - \alpha$ 的位置

### $\lambda$ 的选择

- $\lambda = 0.94$（RiskMetrics 推荐，daily data）
- $\lambda = 0.97$（monthly data）
- $\lambda$ 越小 → 越 responsive to recent shocks
- $\lambda$ 越大 → 越 stable/smooth

---

# 5. VaR 的时间缩放（Scaling VaR）

## 5.1 Square-Root-of-Time Rule

如果 daily returns 是 **iid**（independent and identically distributed）：

$$R_{\Delta t} = R_1 + R_2 + \cdots + R_{\Delta t}$$

**Step 1**：Variance 的加法性（iid 情况）

$$\text{Var}(R_{\Delta t}) = \text{Var}(R_1) + \text{Var}(R_2) + \cdots = \Delta t \cdot \sigma_{\text{daily}}^2$$

**Step 2**：Standard deviation

$$\sigma_{\Delta t} = \sigma_{\text{daily}} \cdot \sqrt{\Delta t}$$

**Step 3**：VaR 缩放

$$\boxed{\text{VaR}_\alpha(\Delta t) = \text{VaR}_\alpha(1\text{ day}) \cdot \sqrt{\Delta t}}$$

### 例子

Daily 95% VaR = $\$100{,}000$

10-day 95% VaR = $100{,}000 \times \sqrt{10} \approx \$316{,}228$

> ⚠️ 这个 rule 的前提是 **iid returns**。如果 returns 有 autocorrelation 或 volatility clustering（GARCH），这个缩放就不准确了。

## 5.2 为什么 regulators 用 $\sqrt{10}$？

Basel 规定 banks 用 **10-day VaR** 来计算 capital requirements。

很多 banks 计算 1-day VaR 然后乘以 $\sqrt{10}$ — 这是一个 **近似**，不是精确值。

---

# 6. Expected Shortfall (ES) — VaR 的补充

## 6.1 VaR 的根本缺陷

VaR 只告诉你：**"threshold 是多少"**

VaR 不告诉你：**"超过 threshold 之后平均亏多少"**

> VaR 是门槛，不是深渊的深度。

## 6.2 ES 的定义

Expected Shortfall（也叫 CVaR, Conditional VaR, Tail VaR）：

$$\text{ES}_\alpha = -E[R \mid R < -\text{VaR}_\alpha]$$

直觉：

> **在最坏的 $(1-\alpha)$ scenarios 里，平均损失是多少？**

## 6.3 Normal Distribution 下的 ES（完整推导）

假设 $R \sim N(\mu, \sigma^2)$。

**Step 1**：条件期望的定义

$$E[R \mid R < -\text{VaR}_\alpha] = \frac{\int_{-\infty}^{-\text{VaR}_\alpha} r \cdot f(r) \, dr}{P(R < -\text{VaR}_\alpha)}$$

分母 = $1 - \alpha$。

**Step 2**：标准化

令 $z = \frac{r - \mu}{\sigma}$，则 $r = \sigma z + \mu$，$dr = \sigma \, dz$。

cutoff 变为：$z^* = \frac{-\text{VaR}_\alpha - \mu}{\sigma} = -z_\alpha$

分子变为：

$$\int_{-\infty}^{-z_\alpha} (\sigma z + \mu) \cdot \frac{1}{\sigma}\phi(z) \cdot \sigma \, dz = \int_{-\infty}^{-z_\alpha} (\sigma z + \mu) \phi(z) \, dz$$

拆开：

$$= \sigma \int_{-\infty}^{-z_\alpha} z \, \phi(z) \, dz + \mu \int_{-\infty}^{-z_\alpha} \phi(z) \, dz$$

第二项 = $\mu \cdot \Phi(-z_\alpha) = \mu(1-\alpha)$。

**Step 3**：关键积分

对于第一项，利用标准正态的性质：

$$\int_{-\infty}^{c} z \, \phi(z) \, dz = -\phi(c)$$

> 推导：$\phi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$，所以 $\frac{d\phi}{dz} = -z\phi(z)$，因此 $z\phi(z) = -\phi'(z)$。
>
> $$\int_{-\infty}^{c} z\phi(z) \, dz = \int_{-\infty}^{c} -\phi'(z) \, dz = -[\phi(z)]_{-\infty}^{c} = -\phi(c) + 0 = -\phi(c)$$

所以第一项 = $\sigma \cdot (-\phi(-z_\alpha))= -\sigma \phi(z_\alpha)$

（利用 $\phi(-z) = \phi(z)$，因为标准正态 PDF 是对称的）

**Step 4**：合并

$$E[R \mid R < -\text{VaR}_\alpha] = \frac{-\sigma\phi(z_\alpha) + \mu(1-\alpha)}{1-\alpha}$$

$$= \mu - \sigma \cdot \frac{\phi(z_\alpha)}{1-\alpha}$$

因此：

$$\boxed{\text{ES}_\alpha = -\mu + \sigma \cdot \frac{\phi(z_\alpha)}{1-\alpha}}$$

> 如果 $\mu \approx 0$：$\text{ES}_\alpha \approx \sigma \cdot \frac{\phi(z_\alpha)}{1-\alpha}$

### 数值例子

$\alpha = 0.95$，$z_{0.95} = 1.645$，$\phi(1.645) \approx 0.1031$

$$\frac{\text{ES}}{\sigma} = \frac{0.1031}{0.05} = 2.063$$

对比：$\frac{\text{VaR}}{\sigma} = 1.645$

> **ES 总是大于 VaR**（因为 ES 是 tail 的平均，而 VaR 只是 tail 的入口）

---

## 6.4 Historical Simulation 下的 ES

### 步骤

1. 排序 portfolio returns：$r_{(1)} \le r_{(2)} \le \cdots \le r_{(T)}$
2. 找到 VaR cutoff index $k = \lfloor T(1-\alpha) \rfloor$
3. ES = 那些最差的 $k$ 个 returns 的**平均值**（取负）

$$\text{ES}_\alpha = -\frac{1}{k}\sum_{i=1}^{k} r_{(i)}$$

---

# 7. VaR 的性质：Coherent Risk Measures

## 7.1 四条公理

一个 "好的" risk measure $\rho$ 应该满足：

| 公理 | 含义 | 直觉 |
|---|---|---|
| **Monotonicity** | 如果 $X \le Y$ always，则 $\rho(X) \ge \rho(Y)$ | 更差的 portfolio 应该有更高的 risk |
| **Sub-additivity** | $\rho(X+Y) \le \rho(X) + \rho(Y)$ | 分散投资不应增加 risk（diversification） |
| **Positive Homogeneity** | $\rho(\lambda X) = \lambda \rho(X)$ for $\lambda > 0$ | 加倍头寸 → 加倍风险 |
| **Translation Invariance** | $\rho(X + c) = \rho(X) - c$ | 加现金 → 减少风险 |

## 7.2 VaR 不是 Coherent 的！

> **VaR violates sub-additivity.**

这意味着：**合并两个 portfolio 的 VaR 可能比分别计算再相加还大**。

### 反例（直觉）

两个 portfolio 各有独立的小概率大额损失：

- 单独看：每个 VaR 都不大（因为大额损失概率 < $1-\alpha$）
- 合并后：至少一个出事的概率可能超过 $1-\alpha$ → VaR 突然变大

> 这在 credit risk 中特别常见 — 两个 bond 各有 3% 违约概率，合在一起并不是 "风险更小"。

## 7.3 ES 是 Coherent 的

Expected Shortfall **满足所有四条公理**，包括 sub-additivity。

> 这就是为什么 Basel III 从 VaR 转向 ES 作为 regulatory risk measure。

---

# 8. Backtesting VaR

## 8.1 什么是 Backtesting？

> **用历史数据验证 VaR model 是否准确**

如果 model 说 "99% VaR = $X$"，那么在过去的数据中，实际 loss 超过 $X$ 的天数应该约占 1%。

## 8.2 Violation Ratio

定义 **violation（breach）**：某天实际 loss > VaR

$$\text{Violation Ratio} = \frac{\text{Number of violations}}{T}$$

期望值 = $1 - \alpha$

## 8.3 Kupiec Test（比例检验）

### Setup

- $T$：total days
- $x$：violations 次数
- $p = 1 - \alpha$：期望 violation 概率

### Null Hypothesis

$$H_0: \text{true violation rate} = p$$

### Test Statistic（Likelihood Ratio）

$$LR = -2\ln\left[\frac{p^x(1-p)^{T-x}}{\hat{p}^x(1-\hat{p})^{T-x}}\right]$$

其中 $\hat{p} = x/T$。

**完整推导**：

$H_0$ 下的 likelihood：$L_0 = p^x(1-p)^{T-x}$

Unrestricted（MLE）的 likelihood：$L_1 = \hat{p}^x(1-\hat{p})^{T-x}$

$$LR = -2(\ln L_0 - \ln L_1) = -2\left[x\ln p + (T-x)\ln(1-p) - x\ln\hat{p} - (T-x)\ln(1-\hat{p})\right]$$

在 $H_0$ 下：

$$LR \xrightarrow{d} \chi^2(1)$$

> 如果 $LR > \chi^2_{1,\alpha_{\text{test}}}$（critical value），则 reject $H_0$，说明 model 不准。

---

# 9. VaR Mapping（风险映射）

## 9.1 为什么需要 Mapping？

实际 portfolio 可能包含成千上万的 positions。直接对每个 position 建模不现实。

**Mapping = 将复杂 portfolio 简化为少量 risk factors 的函数**

## 9.2 Linear Mapping

假设 portfolio value 可以近似为：

$$P \approx \sum_{i=1}^{n} \delta_i \cdot F_i$$

其中 $F_i$ 是 risk factors（如利率、汇率、股价指数），$\delta_i$ 是 sensitivity。

### Delta-Normal VaR

如果 risk factors $F \sim N(\mu_F, \Sigma_F)$：

$$\sigma_P^2 = \delta' \Sigma_F \delta$$

$$\text{VaR}_\alpha = -\delta'\mu_F + z_\alpha \sqrt{\delta' \Sigma_F \delta}$$

> 这与 portfolio VaR 公式结构完全相同，只是把 weights 换成了 sensitivities，把 asset returns 换成了 factor returns。

---

# 10. 总结对比表

| 特征 | VaR | ES |
|---|---|---|
| 定义 | Quantile of loss distribution | Average of losses beyond VaR |
| 回答的问题 | "门槛在哪？" | "超过门槛后平均有多深？" |
| Sub-additive | ❌ No | ✅ Yes |
| Coherent | ❌ No | ✅ Yes |
| 计算难度 | 相对简单 | 稍复杂 |
| Regulatory | Basel II (旧标准) | Basel III (新标准) |
| Normal 公式 | $z_\alpha \cdot \sigma$ | $\frac{\phi(z_\alpha)}{1-\alpha} \cdot \sigma$ |

---

# 11. Key Formulas 速查

$$\text{VaR}_\alpha^{\text{normal}} = -\mu + z_\alpha \sigma$$

$$\text{ES}_\alpha^{\text{normal}} = -\mu + \frac{\phi(z_\alpha)}{1-\alpha} \sigma$$

$$\text{VaR}(\Delta t) = \text{VaR}(1) \cdot \sqrt{\Delta t} \quad \text{(iid assumption)}$$

$$\sigma_p = \sqrt{w'\Sigma w}$$

$$\text{Kupiec LR} = -2\ln\left[\frac{p^x(1-p)^{T-x}}{\hat{p}^x(1-\hat{p})^{T-x}}\right] \sim \chi^2(1)$$

---

# 12. 与前几章的联系

- **[[Chapter 1 - Foundations Rebuild|Chapter 1]]**：[[Chapter 1 - Foundations Rebuild#3. Quantile Function（分位数 = Inverse CDF）|Quantile function]] → VaR 就是 quantile 的应用；[[Chapter 1 - Foundations Rebuild#5. Normal Distribution（正态分布）|Normal]] 和 [[Chapter 1 - Foundations Rebuild#7. Student's t Distribution|t 分布]] → VaR/ES 计算的基础
- **[[Chapter 2 - Multivariate and Regression|Chapter 2]]**：[[Chapter 2 - Multivariate and Regression#3.2 Portfolio Variance — 最重要的公式|$w'\Sigma w$]] → Portfolio VaR 的核心公式来自 covariance matrix
- **[[Chapter 3 - Financial Data & Monte Carlo Simulation|Chapter 3]]**：[[Chapter 3 - Financial Data & Monte Carlo Simulation#1. 为什么需要 Monte Carlo？|Monte Carlo simulation]] → VaR 的第三种计算方法；[[Chapter 3 - Financial Data & Monte Carlo Simulation#6. Exponentially Weighted Covariance（指数加权协方差）|EW covariance]] → VaR 估计中的权重方案
- **[[Chapter 5 - Advanced VaR & Expected Shortfall|Chapter 5]]**：ES 的深入推导 + [[Chapter 5 - Advanced VaR & Expected Shortfall#3. Breaking the Normality Assumption — Copulas|Copula]] simulation → portfolio VaR/ES
