# Chapter 1 — Univariate Statistics Foundations

> 这门课的灵魂：**我们关心的不只是 expected outcome，而是 outcomes 的整个分布**。Risk management = 研究分布的形状、尾巴、极端情况。

---

# 1. 随机变量与分布

## 1.1 Random Variable

> 一个数值结果，事先不知道具体是多少。

Finance 例子：
- 明天某只股票的 return
- 未来 10 天的 portfolio PnL
- 一个 credit default 的 loss

我们不关心某一次实现，而是关心 **所有可能结果的集合** — 这个集合由 **distribution** 描述。

## 1.2 Risk 的类型（PDF 第 1 页）

| Risk 类型 | 含义 |
|---|---|
| Market Risk | 市场价格变动导致的损失 |
| Credit Risk | 对手方违约 |
| Liquidity Risk | 无法以合理价格交易 |
| Operational Risk | 内部流程/系统失败 |
| Business Risk | 商业环境变化 |

> 本课主要关注 **Market Risk**：如何度量、建模、管理价格波动带来的风险。

---

# 2. PDF 与 CDF — 描述分布的两种视角

## 2.1 Probability Density Function (PDF)

$$f(x) = \text{在 } x \text{ 处的"概率密度"}$$

### 关键理解

- PDF $f(x)$ **不是** 概率！$P(X = x) = 0$（连续变量）
- PDF 是 **高度**，概率是 **面积**

$$P(a < X \le b) = \int_a^b f(x)\,dx$$

> 直觉："在 $x$ 附近，outcomes 有多密集？"

### 性质

1. $f(x) \ge 0$ for all $x$
2. $\int_{-\infty}^{\infty} f(x)\,dx = 1$

## 2.2 Cumulative Distribution Function (CDF)

$$F(x) = P(X \le x) = \int_{-\infty}^{x} f(t)\,dt$$

### 性质

1. $F(-\infty) = 0$，$F(+\infty) = 1$
2. $F(x)$ 单调不减（monotonically non-decreasing）
3. $P(a < X \le b) = F(b) - F(a)$

### 在 Risk 中的核心作用

- $F(-5\%) = P(X \le -5\%)$ → "损失超过 5% 的概率"
- CDF 的 tail behavior 直接决定了 VaR、ES 等 risk metrics

> **如果只记一个函数，记 CDF。**

## 2.3 PDF ↔ CDF 关系

$$f(x) = \frac{d}{dx}F(x) \qquad \text{(微分)}$$

$$F(x) = \int_{-\infty}^{x} f(t)\,dt \qquad \text{(积分)}$$

---

# 3. Quantile Function（分位数 = Inverse CDF）

## 3.1 定义

$$F^{-1}(p) = \inf\{x : F(x) \ge p\}$$

> 给定概率 $p \in (0,1)$，找到那个 $x$ 使得"$\le x$ 的概率 = $p$"。

## 3.2 Finance 意义

- $F^{-1}(0.05)$ = "最差 5% 的分界线" → 这就是 **95% VaR**（Chapter 4 详细讲）
- $F^{-1}(0.95)$ = "95% 分位数"
- Confidence intervals、stress thresholds 都是 quantiles

## 3.3 重要性质

- $F^{-1}(F(x)) = x$（CDF 和 quantile 互为逆函数）
- 这是 **Inverse Transform Sampling** 的基础（Chapter 3）：
  - $U \sim \text{Uniform}(0,1) \Rightarrow F^{-1}(U) \sim F$

---

# 4. Moments — 分布形状的四个数字

## 4.1 总体 Moments（Population Moments）

### 第 1 矩：Mean（均值）

$$\mu = E[X] = \int_{-\infty}^{\infty} x \, f(x)\,dx$$

> "Expected outcome"，但注意 risk management 中 mean 不是重点，**尾部** 才是。

### 第 2 中心矩：Variance（方差）

$$\sigma^2 = E[(X - \mu)^2] = \int_{-\infty}^{\infty} (x - \mu)^2 f(x)\,dx$$

$$\sigma = \sqrt{\sigma^2} \quad \text{(standard deviation，标准差)}$$

> "Outcomes 围绕 mean 的离散程度"。Finance 中 $\sigma$ = **volatility**。

注意：Variance 对上行和下行 **一视同仁** — 这是建模简化，不是现实。

### 第 3 标准化矩：Skewness（偏度）

$$\text{Skew}(X) = E\!\left[\left(\frac{X - \mu}{\sigma}\right)^3\right]$$

| Skewness | 含义 | Finance 例子 |
|---|---|---|
| $> 0$ | 右尾更长 → 偶尔有 extreme gains | 买 call option |
| $= 0$ | 对称 | Normal 分布 |
| $< 0$ | 左尾更长 → 偶尔有 extreme losses | 卖 put option、carry trade |

> **负 skewness 是最危险的**：平时稳定赚钱，偶尔巨亏（"picking up pennies in front of a steamroller"）。

### 第 4 标准化矩：Excess Kurtosis（超额峰度）

$$\text{ExKurt}(X) = E\!\left[\left(\frac{X - \mu}{\sigma}\right)^4\right] - 3$$

减 3 是因为 Normal 分布的 kurtosis = 3（作为 baseline）。

| Excess Kurtosis | 含义 |
|---|---|
| $= 0$ | 和 Normal 一样的尾巴厚度（mesokurtic） |
| $> 0$ | 尾巴比 Normal 更厚（leptokurtic）→ 极端事件更频繁 |
| $< 0$ | 尾巴比 Normal 更薄（platykurtic） |

> **金融数据几乎总是 excess kurtosis > 0**（heavy tails），这是 Normal 分布不够用的核心原因。

### General Central Moment（一般公式）

第 $k$ 个中心矩：

$$\mu_k = E[(X - \mu)^k]$$

第 $k$ 个标准化矩：

$$\tilde{\mu}_k = E\!\left[\left(\frac{X - \mu}{\sigma}\right)^k\right] = \frac{\mu_k}{\sigma^k}$$

## 4.2 Sample Moments（样本矩）— 从数据中估计

给定 $n$ 个观测 $x_1, \dots, x_n$：

### Sample Mean

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

> Unbiased estimator of $\mu$：$E[\bar{x}] = \mu$。

### Sample Variance

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

**为什么分母是 $n-1$ 而不是 $n$？（Bessel's Correction）**

如果用 $n$：

$$\hat{\sigma}^2_{\text{biased}} = \frac{1}{n}\sum(x_i - \bar{x})^2$$

这个估计是 **有偏的**（biased），因为：

$$E\!\left[\frac{1}{n}\sum(x_i - \bar{x})^2\right] = \frac{n-1}{n}\sigma^2 < \sigma^2$$

> 用 $\bar{x}$ 代替真实 $\mu$ 时，$(x_i - \bar{x})^2$ 系统性地偏小（因为 $\bar{x}$ 是 minimize $\sum(x_i - c)^2$ 的那个 $c$）。

所以除以 $n-1$ 来修正：

$$s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2 \quad \Rightarrow \quad E[s^2] = \sigma^2 \quad \text{(unbiased)}$$

> $n-1$ 也叫 **degrees of freedom**：$n$ 个观测用掉 1 个自由度来估计 $\bar{x}$。

### Sample Skewness

$$\hat{\text{Skew}} = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n}\!\left(\frac{x_i - \bar{x}}{s}\right)^3$$

> 带修正系数 $\frac{n}{(n-1)(n-2)}$ 来减少 bias。

### Sample Excess Kurtosis

$$\hat{\text{ExKurt}} = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^{n}\!\left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$$

> Kurtosis 的 sample estimator 是 **最不稳定的**（high variance），especially in small samples。需要大量数据才能可靠估计。

---

# 5. Normal Distribution（正态分布）

## 5.1 PDF

$$f(x) = \frac{1}{\sqrt{2\pi}\,\sigma}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

参数：$\mu$（均值）、$\sigma^2$（方差）。

### Standard Normal（标准正态）$Z \sim \mathcal{N}(0,1)$

$$\phi(z) = \frac{1}{\sqrt{2\pi}}\exp\!\left(-\frac{z^2}{2}\right)$$

CDF 记为 $\Phi(z)$，没有 closed-form（只能查表或数值计算）。

## 5.2 Standardization（标准化 / Z-score）

任何 $X \sim \mathcal{N}(\mu, \sigma^2)$ 都可以标准化：

$$Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$$

因此：

$$P(X \le x) = \Phi\!\left(\frac{x - \mu}{\sigma}\right)$$

> 这将任何 Normal 的概率问题 → 转化为查标准正态表。

### Risk 直觉

> "This loss is a 3σ event" 意思是：在 Normal 假设下，这种极端程度的事件概率 $\approx 0.13\%$。

## 5.3 Normal 的关键性质

1. **对称**：$\text{Skew} = 0$
2. **Excess Kurtosis = 0**（baseline for "thin tails"）
3. **线性组合仍是 Normal**：$aX + bY \sim \mathcal{N}(a\mu_X + b\mu_Y, \; a^2\sigma_X^2 + b^2\sigma_Y^2 + 2ab\text{Cov}(X,Y))$
4. **完全由 $\mu$ 和 $\sigma$ 决定**：知道这两个就知道一切
5. **Uncorrelated ⟹ Independent**（仅在 Normal 下成立！）

### 为什么用 Normal？

- Central Limit Theorem（CLT）：大量独立 rv 的 sum → Normal
- 数学方便：closed-form VaR, ES, option pricing
- 是所有比较的 **baseline**

### 为什么 Normal 不够？

> 金融市场 **violates normality exactly where risk matters**：
> - Heavy tails（extreme events 比 Normal 预测的更频繁）
> - Negative skewness
> - Volatility clustering

## 5.4 常用分位数速查

| Confidence | $z_\alpha = \Phi^{-1}(\alpha)$ |
|---|---|
| 5% (α=0.05) | $-1.645$ |
| 2.5% (α=0.025) | $-1.960$ |
| 1% (α=0.01) | $-2.326$ |
| 0.5% (α=0.005) | $-2.576$ |

> 这些值在 VaR 计算中反复出现（Chapter 4）。

---

# 6. Lognormal Distribution（对数正态分布）

## 6.1 定义与动机

如果 $\ln(X) \sim \mathcal{N}(\mu, \sigma^2)$，则 $X$ 服从 lognormal 分布。

等价地：

$$X = e^{\mu + \sigma Z}, \quad Z \sim \mathcal{N}(0,1)$$

### 为什么 Finance 用 Lognormal？

- **价格不能为负**：$X = e^{(\cdot)} > 0$ 永远成立
- 如果 **log returns 是 Normal**（$r_t = \ln(P_t/P_{t-1}) \sim \mathcal{N}$），则 **价格是 Lognormal**
- 这是 Black-Scholes 模型的核心假设

## 6.2 Lognormal 的 Moments

$$E[X] = e^{\mu + \sigma^2/2}$$

$$\text{Var}(X) = e^{2\mu + \sigma^2}(e^{\sigma^2} - 1)$$

$$\text{Skew}(X) = (e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1}$$

> 注意 skewness **总是正的**（right-skewed），因为 lognormal 有 long right tail。

### 重要警告

> $\mu$ 和 $\sigma$ 是 **log return 的** 均值和标准差，**不是** price 的均值和标准差！
>
> $E[X] = e^{\mu + \sigma^2/2} \ne e^\mu$（Jensen's inequality）

## 6.3 数值例子

$\mu = 0.05, \; \sigma = 0.20$（年化 log return mean 5%，vol 20%）：

$$E[X] = e^{0.05 + 0.04/2} = e^{0.07} \approx 1.0725$$

> 即使 expected log return = 5%，expected price ratio $\approx$ 7.25%（因为 Jensen's inequality 的 convexity correction $\sigma^2/2$）。

---

# 7. Student's t Distribution

## 7.1 定义与构造

如果 $Z \sim \mathcal{N}(0,1)$ 且 $V \sim \chi^2(\nu)$ 独立，则：

$$T = \frac{Z}{\sqrt{V/\nu}} \sim t(\nu)$$

$\nu$ = degrees of freedom（自由度）。

## 7.2 PDF

$$f(t;\nu) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\;\Gamma\!\left(\frac{\nu}{2}\right)}\left(1 + \frac{t^2}{\nu}\right)^{-(\nu+1)/2}$$

其中 $\Gamma(\cdot)$ 是 Gamma 函数。

> 和 Normal 对比：Normal 的尾巴是 $e^{-t^2/2}$（exponential decay），t 的尾巴是 $(1+t^2/\nu)^{-(\nu+1)/2}$（polynomial decay）→ **衰减慢得多** → heavy tail。

## 7.3 关键性质

| 性质 | 条件 / 值 |
|---|---|
| Mean | $0$（当 $\nu > 1$） |
| Variance | $\frac{\nu}{\nu - 2}$（当 $\nu > 2$；$\nu \le 2$ 时方差无穷大） |
| Skewness | $0$（对称，当 $\nu > 3$） |
| Excess Kurtosis | $\frac{6}{\nu - 4}$（当 $\nu > 4$；$\nu \le 4$ 时不存在） |
| $\nu \to \infty$ | $t(\nu) \to \mathcal{N}(0,1)$ |

### Moment 存在条件的意义

- $\nu = 3$：mean 存在，variance 存在，但 **kurtosis 不存在**（无穷大的尾巴风险！）
- $\nu = 5$：excess kurtosis $= 6/(5-4) = 6$（比 Normal 的 0 大很多）
- $\nu = 30$：excess kurtosis $= 6/26 \approx 0.23$（接近 Normal）

> **$\nu$ 越小 → 尾巴越厚 → 极端事件越频繁 → risk 越被 Normal 低估。**

## 7.4 Generalized t Distribution

在 MLE fitting 中用的是 "location-scale t"：

$$X = \mu + \sigma \cdot T, \quad T \sim t(\nu)$$

三个参数：
- $\mu$：location（类似 mean）
- $\sigma$：scale（类似 std，但 $\text{Var}(X) = \sigma^2 \cdot \frac{\nu}{\nu-2}$）
- $\nu$：degrees of freedom（控制 tail thickness）

> Exam 中的 "fit t distribution" 就是用 MLE 同时估计 $\mu, \sigma, \nu$。

## 7.5 为什么 Finance 用 t？

- 金融 returns 几乎总是 heavy-tailed（empirical excess kurtosis > 0）
- t 分布用一个额外参数 $\nu$ 就能 capture tail heaviness
- 和 Normal 比较：fit 两者，用 **AICc** 选更好的（Chapter 2 详细讲）

---

# 8. Maximum Likelihood Estimation (MLE) 简介

## 8.1 核心思想

给定观测数据 $x_1, \dots, x_n$ 和参数化分布 $f(x; \theta)$：

$$L(\theta) = \prod_{i=1}^{n} f(x_i; \theta)$$

$$\ell(\theta) = \sum_{i=1}^{n} \ln f(x_i; \theta) \quad \text{(log-likelihood)}$$

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \; \ell(\theta)$$

> 选择使 observed data "最不意外" 的参数。

## 8.2 Normal MLE

对 $X_i \sim \mathcal{N}(\mu, \sigma^2)$：

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

对 $\mu$ 求导令为零：

$$\hat{\mu}_{\text{MLE}} = \bar{x}$$

对 $\sigma^2$ 求导令为零：

$$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

> 注意 MLE 的方差估计用 $1/n$（biased！），而 sample variance 用 $1/(n-1)$（unbiased）。两者在大样本下差别很小。

## 8.3 t MLE

对 generalized $t(\nu, \mu, \sigma)$，log-likelihood 没有 closed-form solution for $\hat{\theta}$，需要 **数值优化**（Newton-Raphson, scipy.optimize 等）。

---

# 9. Hypothesis Testing 与 Monte Carlo Simulation

## 9.1 Kurtosis Estimator Bias（PDF 第 12 页的例子）

这个例子非常重要，它展示了 **Monte Carlo 思维**：

### 问题

> Sample kurtosis 是否是 true kurtosis 的 unbiased estimator？

### 方法（Monte Carlo 验证）

1. 设定 true distribution：$X \sim \mathcal{N}(0,1)$（true excess kurtosis = 0）
2. 重复 $M$ 次：
   - 生成 $n$ 个样本
   - 计算 sample excess kurtosis $\hat{K}_m$
3. 统计 $\bar{K} = \frac{1}{M}\sum_{m=1}^{M} \hat{K}_m$
4. 如果 $\bar{K} \approx 0$ → unbiased；如果 $\bar{K} \ne 0$ → biased

### 结论

> 当 $n$ 较小时（如 $n = 100$），sample kurtosis **有明显 bias**（通常偏低）。这就是为什么 sample estimator 中有复杂的修正系数 $\frac{n(n+1)}{(n-1)(n-2)(n-3)}$。

### 核心教训

- **所有 sample statistics 都有 estimation error**
- 高阶矩（skewness, kurtosis）尤其不稳定
- Risk metrics 是 "估计的估计" → 本身带有 uncertainty

## 9.2 t-Test 速查

$$T = \frac{\bar{X} - \mu_0}{s/\sqrt{n}}$$

- $\mu_0$：null hypothesis 下的均值
- $T \sim t(n-1)$ under $H_0$
- 如果 $|T|$ 很大 → reject $H_0$（数据和假设不一致）

> 直觉：$T$ 衡量 "观测到的偏差有多少个 standard error" → 用 $t$ 分布判断这个偏差是否 "显著"。

---

# 10. Chapter Summary — 全章知识地图

| 概念 | 核心问题 | 关键公式/记忆点 |
|---|---|---|
| Random Variable | 不确定的数值结果 | 由 distribution 描述 |
| PDF | 每个点附近有多少概率密度？ | $f(x) \ge 0$，面积 = 概率 |
| CDF | 累积到 $x$ 的概率？ | $F(x) = P(X \le x)$ |
| Quantile | 给定概率找值 | $F^{-1}(p)$ → VaR 的基础 |
| Mean | 期望值 | $\mu = E[X]$ |
| Variance | 离散程度（volatility） | $\sigma^2 = E[(X-\mu)^2]$，sample 用 $n-1$ |
| Skewness | 左右不对称程度 | $< 0$ = 左尾更长 = 危险 |
| Excess Kurtosis | 尾巴厚度 vs Normal | $> 0$ = heavy tails = 极端事件频繁 |
| Normal | Baseline 分布 | $\mu, \sigma$ 完全决定；薄尾；对称 |
| Lognormal | 价格分布 | $\ln X \sim \mathcal{N}$；$E[X] = e^{\mu+\sigma^2/2}$ |
| Student's t | Heavy tail 分布 | $\nu$ 越小 → 尾巴越厚；$\nu \to \infty \Rightarrow$ Normal |
| MLE | 选择最佳参数 | $\hat\theta = \arg\max \ell(\theta)$ |
| Monte Carlo | 用模拟代替解析 | 生成 → 计算 → 统计 |
