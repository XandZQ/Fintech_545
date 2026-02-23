

> 这一章解决一个核心问题：**如何从已知的分布假设中生成大量模拟 scenarios**，从而估计 portfolio 的 risk metrics（VaR, ES 等）。

---

# 1. 为什么需要 Monte Carlo？

## 1.1 核心动机

在 risk management 中，我们关心的量往往是 **随机变量的复杂函数**：

- Portfolio PnL = $f(\text{多个 asset 的 returns})$
- 那些 returns 是随机的
- $f$ 可能很复杂（非线性，含 options 等）
- 因此 PnL 的分布很难解析求得

> Monte Carlo 的核心思想：**用大量模拟代替解析计算**。

### 三步框架

1. **Simulate**：从假设的分布中抽取大量 random scenarios
2. **Evaluate**：对每个 scenario 计算 portfolio 的 outcome
3. **Aggregate**：对 outcomes 取 histogram / quantile / mean → VaR, ES

---

# 2. CDF、Quantile Function 与 Inverse Transform Sampling

## 2.1 CDF 回顾

$$F(x) = P(X \le x) = \int_{-\infty}^{x} f(t)\,dt$$

- $f(t)$：PDF（密度函数）— "单位长度上有多少概率"
- $F(x)$：CDF — "累计到 $x$ 的概率总量"

关键性质：

- $F(x) \in [0, 1]$
- $F(x)$ 单调递增（monotonically increasing）

## 2.2 Quantile Function（分位数函数 = Inverse CDF）

$$F^{-1}(p) = \text{满足 } P(X \le x) = p \text{ 的那个 } x$$

> 给定一个概率 $p \in (0,1)$，反查对应的值 $x$。

例子：

- $F^{-1}(0.05)$ = "最差 5% 的 cutoff" → 这就是 95% [[Chapter 4 - Value-at-Risk (VaR)#1. VaR 的定义与直觉|VaR]]
- $F^{-1}(0.95)$ = "95% 分位数"

## 2.3 Inverse Transform Sampling（核心 simulation 工具）

### 定理

如果 $U \sim \text{Uniform}(0,1)$，则：

$$X = F^{-1}(U)$$

使得 $X \sim F$（即 $X$ 服从目标分布）。

### 完整推导

**要证**：$P(X \le x) = F(x)$

$$P(X \le x) = P(F^{-1}(U) \le x)$$

因为 $F$ 是单调递增的，两边同时取 $F$：

$$= P(U \le F(x))$$

而 $U \sim \text{Uniform}(0,1)$，所以 $P(U \le u) = u$：

$$= F(x) \quad \checkmark$$

### 直觉

- $U$ 均匀地"撒"在 $[0,1]$ 上
- $F^{-1}$ 把这些均匀点"拉伸/压缩"成目标分布的形状
- PDF 密度高的区域 → CDF 变化快 → $F^{-1}$ 把更多 $U$ 值映射到这里

> 这个技巧的意义：**只要你有 Uniform 随机数和 $F^{-1}$ 的公式，就能从任何分布中 simulate**。

---

# 3. Multivariate Normal Simulation（多维正态模拟）

## 3.1 问题的提出

现实中 portfolio 包含多个 assets，它们的 returns 是 **correlated** 的。

我们需要 simulate 的不是单个随机变量，而是一个 **随机向量**：

$$X = (X_1, X_2, \dots, X_n)' \sim N(\mu, \Sigma)$$

其中：

- $\mu \in \mathbb{R}^n$：均值向量
- $\Sigma \in \mathbb{R}^{n \times n}$：协方差矩阵
	- 对角线 = 各 asset 的 variance
	- 非对角线 = 两两之间的 covariance

## 3.2 核心构造：$X = LZ + \mu$

### 思路

> **把独立的标准正态变量 "混合" 成相关的正态变量**。

存在 $L \in \mathbb{R}^{n \times n}$ 使得：

$$X = LZ + \mu$$

其中 $Z = (Z_1, \dots, Z_n)'$，每个 $Z_i \sim N(0,1)$ 且**彼此独立**。

### 为什么 $\Sigma = LL'$？（完整推导）

**Step 1**：$X$ 的协方差

$$\text{Cov}(X) = \text{Cov}(LZ + \mu)$$

**Step 2**：常数 $\mu$ 不影响协方差

$$= \text{Cov}(LZ)$$

**Step 3**：矩阵线性变换的协方差公式

> 对于任意常数矩阵 $A$ 和随机向量 $Y$：$\text{Cov}(AY) = A\,\text{Cov}(Y)\,A'$

$$= L\,\text{Cov}(Z)\,L'$$

**Step 4**：$Z$ 的各分量独立且方差为 1

$$\text{Cov}(Z) = I \quad \text{（单位矩阵）}$$

**Step 5**：代入

$$\text{Cov}(X) = LIL' = LL'$$

所以只要选择 $L$ 使得 $LL' = \Sigma$，就能保证 simulate 出的 $X$ 具有正确的协方差结构。$\blacksquare$

> 类比：1 维中，$\sigma^2$ 是 variance，$\sigma$ 是 standard deviation。多维中，$\Sigma$ 是 "variance"，$L$ 就是 "standard deviation matrix"。

---

## 3.3 Simulation 算法（矩阵维度详解）

要 simulate $K$ 个 scenarios（每个 scenario 包含 $n$ 个 asset 的 return）：

**Step 1**：对 $\Sigma$ 做分解，得到 $L$（$n \times n$）

**Step 2**：生成独立标准正态矩阵 $Z$（$n \times K$）

- 每列是一个 scenario 的 $n$ 个独立标准正态

**Step 3**：计算

$$X = LZ + \mu \mathbf{1}'$$

- $LZ$：$n \times K$（每列是一个相关正态向量）
- $\mu\mathbf{1}'$：把均值加到每列上
- 结果 $X$ 的 shape = $n \times K$

**Step 4**：转置

$$X^T \in \mathbb{R}^{K \times n}$$

每行 = 一个 scenario，每列 = 一个 asset → 标准 "数据表" 格式。

---

# 4. Cholesky Factorization（如何求 $L$）

## 4.1 Cholesky 是什么？

对于 **positive definite (PD)** 的矩阵 $\Sigma$，存在**唯一**的下三角矩阵 $L$ 使得：

$$\Sigma = LL'$$

$L$ 的结构（下三角）：

$$L = \begin{pmatrix} L_{11} & 0 & 0 \\ L_{21} & L_{22} & 0 \\ L_{31} & L_{32} & L_{33} \end{pmatrix}$$

## 4.2 Cholesky 算法（逐列求解，完整公式）

### 对角元素（$j = i$）

$$L_{j,j} = \sqrt{\Sigma_{j,j} - \sum_{k=1}^{j-1} L_{j,k}^2}$$

> 直觉：$\Sigma_{j,j}$ 是总方差，减去前面各列已经 "解释" 的部分，剩下的开方。

### 非对角元素（$i > j$）

$$L_{i,j} = \frac{1}{L_{j,j}}\left(\Sigma_{i,j} - \sum_{k=1}^{j-1} L_{i,k} L_{j,k}\right)$$

> 直觉：$\Sigma_{i,j}$ 是总协方差，减去前面各列已经 "解释" 的部分，再除以当前列的 "缩放因子" $L_{j,j}$。

### 算法流程

对 $j = 1, 2, \dots, n$（逐列）：

1. 先算对角元素 $L_{j,j}$
2. 再算该列下方的所有 $L_{i,j}$（$i = j+1, \dots, n$）

---

## 4.3 Worked Example（$3 \times 3$）

设 covariance matrix：

$$\Sigma = \begin{pmatrix} 1 & 0.5 & 0.3 \\ 0.5 & 1 & 0.4 \\ 0.3 & 0.4 & 1 \end{pmatrix}$$

### Column 1（$j = 1$）

对角：

$$L_{1,1} = \sqrt{\Sigma_{1,1}} = \sqrt{1} = 1$$

（$j=1$ 时没有前面的列，所以 $\sum = 0$）

下方：

$$L_{2,1} = \frac{\Sigma_{2,1}}{L_{1,1}} = \frac{0.5}{1} = 0.5$$

$$L_{3,1} = \frac{\Sigma_{3,1}}{L_{1,1}} = \frac{0.3}{1} = 0.3$$

### Column 2（$j = 2$）

对角：

$$L_{2,2} = \sqrt{\Sigma_{2,2} - L_{2,1}^2} = \sqrt{1 - 0.25} = \sqrt{0.75} \approx 0.8660$$

下方：

$$L_{3,2} = \frac{\Sigma_{3,2} - L_{3,1}L_{2,1}}{L_{2,2}} = \frac{0.4 - 0.3 \times 0.5}{0.8660} = \frac{0.25}{0.8660} \approx 0.2887$$

### Column 3（$j = 3$）

对角：

$$L_{3,3} = \sqrt{\Sigma_{3,3} - L_{3,1}^2 - L_{3,2}^2} = \sqrt{1 - 0.09 - 0.0833} = \sqrt{0.8267} \approx 0.9092$$

### 最终结果

$$L = \begin{pmatrix} 1 & 0 & 0 \\ 0.5 & 0.8660 & 0 \\ 0.3 & 0.2887 & 0.9092 \end{pmatrix}$$

### 验证

$$LL' = \begin{pmatrix} 1 & 0.5 & 0.3 \\ 0.5 & 1 & 0.4 \\ 0.3 & 0.4 & 1 \end{pmatrix} = \Sigma \quad \checkmark$$

---

## 4.4 PD vs PSD：为什么 Cholesky 有时会失败？

### Positive Definite (PD)

$$w'\Sigma w > 0 \quad \text{for all } w \ne 0$$

含义：任何方向上都有正的 variance → Cholesky 开方时 **根号下始终为正**。

### Positive Semi-Definite (PSD)

$$w'\Sigma w \ge 0 \quad \text{for all } w$$

含义：某些方向上 variance = 0（完美线性依赖）→ 某个 $L_{j,j}$ 的根号下 = 0 → **除法出现 $\div 0$**。

### 处理方法

如果在计算 $L_{j,j}$ 时根号下 $\le 0$：

- 设 $L_{j,j} = 0$
- 该列所有 $L_{i,j} = 0$

> 这等价于说："第 $j$ 个方向上没有新的信息，可以被前面的列完全解释。"

---

# 5. Missing Data 与 Covariance Matrix 估计

## 5.1 金融数据的现实问题

不同市场有不同的交易日历（假日不同、时区不同）。所以多个 asset 的 return 时间序列经常 **不完全对齐**。

### 例子

| Date | Asset A | Asset B | Asset C |
|---|---|---|---|
| Jan 2 | 0.5% | 0.3% | — |
| Jan 3 | -0.2% | — | 0.1% |
| Jan 4 | 0.1% | 0.4% | 0.2% |

→ 只有 Jan 4 三个 asset 都有数据。

## 5.2 两种常见处理方法

### 方法 1：Complete Cases（只用完全重叠的日子）

只保留 **所有 asset 都有数据** 的行。

- ✅ 优点：得到的 $\hat{\Sigma}$ 一定是 PSD（数学保证）
- ❌ 缺点：可能丢掉大量数据（尤其 asset 多的时候）

### 方法 2：Pairwise Covariance

对每对 $(i, j)$，用它们 **共同有数据的日子** 分别估计 $\hat{\Sigma}_{i,j}$。

- ✅ 优点：充分利用所有可用数据
- ❌ 缺点：拼出来的矩阵 **可能不是 PSD** → Cholesky 会失败

> 这就是为什么我们需要后面的 "non-PSD repair" 方法（Section 7–9）。

---

# 6. Exponentially Weighted Covariance（指数加权协方差）

## 6.1 动机

传统的 sample variance 对所有历史数据 **等权处理**：

$$\hat{\sigma}^2 = \frac{1}{T-1}\sum_{t=1}^{T}(x_t - \bar{x})^2$$

问题：3 年前的数据和昨天的数据权重一样 — 对 risk management 来说不合理。

> **Recent shocks should matter more.**

## 6.2 Exponentially Weighted Variance（递推形式）

$$\boxed{\sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)(x_{t-1} - \bar{x})^2}$$

其中 $\lambda \in (0,1)$ 是 **衰减因子（decay factor）**。

### 直觉拆解

$$\sigma_t^2 = \underbrace{\lambda\,\sigma_{t-1}^2}_{\text{保留昨天的估计（记忆）}} + \underbrace{(1-\lambda)(x_{t-1} - \bar{x})^2}_{\text{加入昨天的新信息（surprise）}}$$

- $\lambda$ 大（如 0.97）→ 记忆长，反应慢 → 平滑
- $\lambda$ 小（如 0.90）→ 记忆短，反应快 → 波动

## 6.3 Back-Substitution 展开（完整推导）

### 递推展开

$$\sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)\,\epsilon_{t-1}^2$$

其中 $\epsilon_{t-i} = x_{t-i} - \bar{x}$（deviation from mean）。

将 $\sigma_{t-1}^2$ 继续展开：

$$\sigma_t^2 = \lambda\bigl[\lambda\,\sigma_{t-2}^2 + (1-\lambda)\,\epsilon_{t-2}^2\bigr] + (1-\lambda)\,\epsilon_{t-1}^2$$

$$= \lambda^2\,\sigma_{t-2}^2 + (1-\lambda)\lambda\,\epsilon_{t-2}^2 + (1-\lambda)\,\epsilon_{t-1}^2$$

再展开一层：

$$= \lambda^3\,\sigma_{t-3}^2 + (1-\lambda)\lambda^2\,\epsilon_{t-3}^2 + (1-\lambda)\lambda\,\epsilon_{t-2}^2 + (1-\lambda)\,\epsilon_{t-1}^2$$

### 归纳得到通项

$$\sigma_t^2 = \lambda^n\,\sigma_{t-n}^2 + (1-\lambda)\sum_{i=1}^{n}\lambda^{i-1}\,\epsilon_{t-i}^2$$

当 $n \to \infty$（且 $\lambda < 1$），$\lambda^n \to 0$：

$$\boxed{\sigma_t^2 = (1-\lambda)\sum_{i=1}^{\infty}\lambda^{i-1}\,\epsilon_{t-i}^2}$$

### 直觉

Variance 是过去所有 squared surprises 的 **加权平均**，权重随时间 **指数衰减**。

---

## 6.4 权重为什么 sum to 1？（几何级数证明）

权重序列：$(1-\lambda),\;(1-\lambda)\lambda,\;(1-\lambda)\lambda^2,\;\dots$

$$\sum_{i=1}^{\infty}(1-\lambda)\lambda^{i-1} = (1-\lambda)\sum_{i=0}^{\infty}\lambda^{i} = (1-\lambda)\cdot\frac{1}{1-\lambda} = 1 \quad \checkmark$$

> 所以权重构成一个合法的 "概率分布" across time — 每个历史时点分到一个非负权重，总和为 1。

## 6.5 有限样本的权重定义与归一化

### 定义权重

$$w_{t-i} = (1-\lambda)\lambda^{i-1}, \quad i = 1, 2, \dots, n$$

### 有限样本问题

当只有 $n$ 个观测时，$\sum_{i=1}^{n}w_{t-i} < 1$（截断了尾巴）。

解决方法 — **归一化**：

$$w_{t-i} \leftarrow \frac{(1-\lambda)\lambda^{i-1}}{\sum_{j=1}^{n}(1-\lambda)\lambda^{j-1}} = \frac{\lambda^{i-1}}{\sum_{j=1}^{n}\lambda^{j-1}} = \frac{\lambda^{i-1}}{\frac{1-\lambda^n}{1-\lambda}}$$

> 归一化确保权重仍然 sum to 1，即使样本有限。

## 6.6 加权 Variance 和 Covariance 估计量

### Weighted Variance

$$\hat{\sigma}_t^2 = \sum_{i=1}^{n}w_{t-i}(x_{t-i} - \bar{x})^2$$

### Weighted Covariance

$$\widehat{\text{cov}}_t(x,y) = \sum_{i=1}^{n}w_{t-i}(x_{t-i} - \bar{x})(y_{t-i} - \bar{y})$$

> 结构完全相同：recent data gets bigger weight。

### 构建 Exponentially Weighted Covariance Matrix

$$\hat{\Sigma}_t = \begin{pmatrix} \hat{\sigma}_{1,t}^2 & \widehat{\text{cov}}_{12,t} & \cdots \\ \widehat{\text{cov}}_{21,t} & \hat{\sigma}_{2,t}^2 & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}$$

这个矩阵可以直接用于 Cholesky → simulation → [[Chapter 4 - Value-at-Risk (VaR)|VaR]]/[[Chapter 5 - Advanced VaR & Expected Shortfall|ES]]。

---

## 6.7 与 GARCH(1,1) 的关系

GARCH(1,1)：

$$\sigma_t^2 = \omega + \alpha\,\epsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2$$

RiskMetrics（JP Morgan 的 exponentially weighted 模型）是 GARCH 的特殊情况：

$$\omega = 0, \quad \alpha = 1-\lambda, \quad \beta = \lambda$$

因此 $\alpha + \beta = 1$。

### ⚠️ 重要警告

当 $\alpha + \beta = 1$（即 $\omega = 0$）时：

- [[Chapter 2 - Multivariate and Regression#9.3 Unconditional Variance 推导|GARCH]] 的 **长期方差（unconditional variance）** 为：

$$\bar{\sigma}^2 = \frac{\omega}{1 - \alpha - \beta}$$

- 分母 = 0 → 长期方差 **不存在（undefined）**

> 这意味着 RiskMetrics 模型 **不能用于长期预测** — 它没有 mean-reversion，variance 会永远 "漂移"。

---

# 7. Non-PSD 矩阵问题（为什么 Cholesky 会失败）

## 7.1 产生 Non-PSD 的原因

1. **Pairwise estimation**：用不同子集的数据估计各个 $\hat{\Sigma}_{i,j}$，拼出来的矩阵不保证 PSD
2. **浮点误差**：数值计算中的舍入误差可能把一个 "几乎 PSD" 的矩阵变成 non-PSD

### 症状

Cholesky 算法中，$L_{j,j} = \sqrt{\cdot}$ 里的值为 **负数** → 无法开方 → 算法崩溃。

---

# 8. Fix 1：Rebonato–Jäckel Eigenvalue Cleaning

## 8.1 基本思路

> **负的 eigenvalue = "某个方向上的 variance 是负的" — 这物理上不可能。把它 clip 到 0。**

## 8.2 完整步骤

### Step 1：对 correlation matrix $C$ 做 eigendecomposition

$$C = S\Lambda S^T$$

其中：

- $S = [s_1, s_2, \dots, s_n]$：eigenvectors（正交矩阵）
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)$：eigenvalues

### Step 2：Clip 负 eigenvalues

$$\lambda_i' = \max(\lambda_i, 0)$$

构建 $\Lambda' = \text{diag}(\lambda_1', \lambda_2', \dots, \lambda_n')$。

### Step 3：构建 Scaling Matrix $T$

Correlation matrix 要求对角线 = 1。Clip 之后这个性质可能被破坏，需要 rescale。

定义 $B_0 = S\sqrt{\Lambda'}$，则 $B_0 B_0^T$ 的对角线可能不是 1。

构建对角矩阵 $T = \text{diag}(t_1, \dots, t_n)$，其中：

$$t_i = \frac{1}{\sqrt{\sum_{j=1}^{n} S_{i,j}^2 \cdot \lambda_j'}}$$

> $t_i$ 的作用：rescale 第 $i$ 行，使得 $\hat{C}$ 的第 $(i,i)$ 元素 = 1。

### Step 4：构建修复后的矩阵

$$B = T\,S\,\sqrt{\Lambda'}$$

$$\hat{C} = BB^T$$

$\hat{C}$ 是 PSD 且对角线为 1 → 合法的 correlation matrix。

### 直觉总结

$$\text{原始 } C \xrightarrow{\text{eigen-decompose}} \text{找到方向和 variance} \xrightarrow{\text{clip 负值}} \text{去掉不可能的成分} \xrightarrow{\text{rescale}} \text{修复对角线}$$

---

# 9. Fix 2：Higham Algorithm（最近 PSD 矩阵）

## 9.1 思路

Rebonato-Jäckel 是一种 heuristic。Higham 提出了一个 **优化问题**：

> 找到与 $C$ 最"近"的 PSD correlation matrix。

$$\min_{A} \|A - C\|_F \quad \text{subject to } A \text{ is PSD and } A_{ii} = 1$$

### Frobenius Norm

$$\|A\|_F = \sqrt{\sum_{i=1}^{n}\sum_{j=1}^{n} a_{i,j}^2}$$

> 就是把矩阵当作一个长向量，计算 Euclidean distance。

## 9.2 Alternating Projections（交替投影法）

Higham 的算法在两个约束之间 **来回投影**：

### Projection 1：$P_S(A)$ — 强制 PSD

1. 对 $A$ 做 eigendecomposition：$A = S\Lambda S^T$
2. Clip：$\lambda_i' = \max(\lambda_i, 0)$
3. 重建：$P_S(A) = S\Lambda' S^T$

### Projection 2：$P_U(A)$ — 强制对角线 = 1

$$[P_U(A)]_{i,j} = \begin{cases} 1 & \text{if } i = j \\ A_{i,j} & \text{if } i \ne j \end{cases}$$

> 直接把对角线设为 1，off-diagonal 不动。

### 迭代过程

$$\Delta S_0 = 0, \quad Y_0 = C$$

For $k = 0, 1, 2, \dots$：

1. $R_k = Y_k - \Delta S_k$
2. $X_k = P_S(R_k)$（投影到 PSD）
3. $\Delta S_{k+1} = X_k - R_k$（Dykstra correction）
4. $Y_{k+1} = P_U(X_k)$（投影到对角线 = 1）

重复直到 $\|Y_{k+1} - Y_k\|_F < \text{tolerance}$。

### 为什么需要 Dykstra Correction？

直接交替投影可能不收敛到真正的最近点。Dykstra correction 项 $\Delta S_k$ 记录了每次投影的 "误差"，确保算法收敛到 **真正的最优解**。

> Higham 比 Rebonato-Jäckel 更精确，但计算更慢。实践中两者都常用。

---

# 10. PCA — 另一种 Simulation 引擎

## 10.1 PCA 与 Cholesky 的关系

从 eigendecomposition 出发：

$$C = S\Lambda S^T$$

定义：

$$B = S\sqrt{\Lambda}$$

则：

$$BB^T = S\sqrt{\Lambda}\sqrt{\Lambda}S^T = S\Lambda S^T = C \quad \checkmark$$

> 所以 $B$ 也是一个合法的 "square root"，可以替代 Cholesky 的 $L$ 用于 simulation。

### Cholesky vs PCA 对比

| | Cholesky | PCA |
|---|---|---|
| $L$ 的结构 | 下三角 | $S\sqrt{\Lambda}$（不一定三角） |
| 分解结果 | 唯一 | 取决于 eigenvector 排列 |
| 计算速度 | 快 | 稍慢（需要 eigen 分解） |
| 降维 | 不支持 | ✅ **支持**（核心优势） |

## 10.2 Dimensionality Reduction（降维 — PCA 的核心优势）

### 思路

如果 $\Sigma$ 有 $n$ 个 eigenvalues，但只有前 $K$ 个是 "大的"（比如前 3 个 eigenvalue 占总方差的 95%），那我们可以：

- 只保留前 $K$ 个 eigenvectors 和 eigenvalues
- 只需要 simulate $K$ 个独立标准正态（而不是 $n$ 个）

### Percent Variance Explained

将 eigenvalues 按从大到小排列：$\lambda_{(1)} \ge \lambda_{(2)} \ge \cdots \ge \lambda_{(n)}$

$$\text{PctExplained}(K) = \frac{\sum_{i=1}^{K}\lambda_{(i)}}{\sum_{j=1}^{n}\lambda_{(j)}}$$

### 例子

假设 10 个 assets，eigenvalues 为：$5.2, 2.1, 1.3, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.07$

总和 = 10

前 3 个：$(5.2 + 2.1 + 1.3)/10 = 86\%$

> 只用 3 个 component 就能解释 86% 的 variance → 极大节省计算量。

### Truncated Simulation

$$B_K = S_K\sqrt{\Lambda_K}$$

其中 $S_K$ 是前 $K$ 列 eigenvectors，$\Lambda_K$ 是前 $K$ 个 eigenvalues。

$$X \approx B_K Z_K + \mu, \quad Z_K \sim N(0, I_K)$$

> 在 asset 数量很多（如 500 个 bond）时，PCA 能把 simulate 维度从 500 降到 3–5，效率提升巨大。

---

# 11. Simulating from Models（基于模型的 Monte Carlo）

## 11.1 一般框架

不直接 simulate raw returns，而是：

1. 对每个变量建模：$x_{i,t} = f_i(\Omega_t, \epsilon_{i,t})$
	- $\Omega_t$：information set（可能包含其他变量的值）
	- $\epsilon_{i,t}$：error term
2. 假设 errors 联合正态：$\epsilon \sim N(0, \Sigma_\epsilon)$
3. 用 $X = LZ + \mu$ 方法 simulate $\epsilon$
4. 代入 model function $f_i$ 重构 $x_{i,t}$

## 11.2 Dependency Ordering（依赖顺序）

### 问题

如果 $x_2$ 的 model 需要 $x_1$ 作为输入（e.g., $x_2 = \beta x_1 + \epsilon_2$），则必须：

1. **先 simulate $x_1$**
2. **再用 $x_1$ 的值 simulate $x_2$**

### Circular Dependency

如果 A 依赖 B 且 B 依赖 A → 需要 iterative solver（超出本课范围）。

> 在实际 risk management 中，通常可以通过选择 model 结构来避免 circular dependency。

---

# 12. 六种 Covariance Matrix 估计方法总结

Slides 中提到了多种构建 $\hat{\Sigma}$ 的方法，这里做一个统一比较：

| 方法 | 思路 | 优点 | 缺点 |
|---|---|---|---|
| **Equal-weight sample** | $\frac{1}{T-1}\sum(x_t-\bar{x})(x_t-\bar{x})'$ | 简单，PSD guaranteed | 对远期和近期一视同仁 |
| **Exponentially weighted** | 近期权重大，$\lambda$ 控制衰减 | 反应快 | 需要选 $\lambda$ |
| **Complete cases** | 只用全部 asset 都有数据的日子 | PSD guaranteed | 浪费数据 |
| **Pairwise** | 各 pair 分别用共有数据 | 充分利用数据 | 可能 non-PSD |
| **Rebonato-Jäckel** | Clip 负 eigenvalues + rescale | 快速修复 | Heuristic，不是最优 |
| **Higham** | 优化问题，交替投影 | 最优（最近 PSD） | 计算较慢 |

---

# 13. Key Formulas 速查

$$X = LZ + \mu, \quad \Sigma = LL' \quad \text{（Multivariate Normal Simulation）}$$

$$L_{j,j} = \sqrt{\Sigma_{j,j} - \sum_{k=1}^{j-1}L_{j,k}^2} \quad \text{（Cholesky 对角元素）}$$

$$L_{i,j} = \frac{\Sigma_{i,j} - \sum_{k=1}^{j-1}L_{i,k}L_{j,k}}{L_{j,j}} \quad \text{（Cholesky 非对角元素）}$$

$$\sigma_t^2 = \lambda\sigma_{t-1}^2 + (1-\lambda)\epsilon_{t-1}^2 \quad \text{（Exponentially Weighted Variance）}$$

$$\sigma_t^2 = (1-\lambda)\sum_{i=1}^{\infty}\lambda^{i-1}\epsilon_{t-i}^2 \quad \text{（展开形式）}$$

$$w_{t-i} = (1-\lambda)\lambda^{i-1} \quad \text{（Exponential Weights）}$$

$$\text{PctExplained}(K) = \frac{\sum_{i=1}^{K}\lambda_{(i)}}{\sum_{j=1}^{n}\lambda_{(j)}} \quad \text{（PCA Variance Explained）}$$

---

# 14. 与前后章节的联系

- **[[Chapter 1 - Foundations Rebuild|Chapter 1]]**：[[Chapter 1 - Foundations Rebuild#2. PDF 与 CDF — 描述分布的两种视角|CDF]]、[[Chapter 1 - Foundations Rebuild#3. Quantile Function（分位数 = Inverse CDF）|Quantile function]] → 本章 Inverse Transform Sampling 的基础
- **[[Chapter 2 - Multivariate and Regression|Chapter 2]]**：[[Chapter 2 - Multivariate and Regression#3. Covariance Matrix — Portfolio Risk 的核心|Covariance matrix]]、[[Chapter 2 - Multivariate and Regression#4. Multivariate Normal Distribution|MVN]] → 本章 simulation 对象；[[Chapter 2 - Multivariate and Regression#9. GARCH — Volatility 也有 Memory|GARCH]] → RiskMetrics 是其特殊情况
- **[[Chapter 4 - Value-at-Risk (VaR)|Chapter 4]]**：[[Chapter 4 - Value-at-Risk (VaR)#3.3 Monte Carlo Simulation|Monte Carlo]] 是 VaR 三大计算方法之一；EW $\hat{\Sigma}$ → 用于 [[Chapter 4 - Value-at-Risk (VaR)#3.1 Parametric (Variance-Covariance) Method|parametric VaR]]
- **[[Chapter 5 - Advanced VaR & Expected Shortfall|Chapter 5]]**：Cholesky 和 PCA → [[Chapter 5 - Advanced VaR & Expected Shortfall#5. Simulating from the Gaussian Copula（完整流程）|Copula simulation]] 的核心引擎；[[Chapter 5 - Advanced VaR & Expected Shortfall#2. Model-Based Simulation（从拟合模型中做 Monte Carlo）|Model-based simulation]] → 从拟合模型中做 Monte Carlo
