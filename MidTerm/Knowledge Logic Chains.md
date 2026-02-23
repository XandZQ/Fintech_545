# Knowledge Logic Chains — 知识逻辑链 Mind Map

> **怎么用**：每条 chain 从一个"为什么"出发，一步步推到最终的工具/方法。考试时如果忘了某个概念的来龙去脉，顺着 chain 走一遍就能想起来。
> 所有 `[[]]` 都是 Obsidian 链接，可以点击跳转。

---

# Chain 0: 全局大图 — 从"风险是什么"到"算出一个数字"

```
我们管理的是 Market Risk（价格变动导致的损失）
    │
    ▼
风险 = 不确定性 = 我们不知道明天的 return 是多少
    │
    ▼
所以我们用 **分布** 来描述所有可能的 return
    │
    ├─→ 单个资产? → Univariate distribution (Ch1)
    │       │
    │       ├─→ Normal? 还是 t?  → MLE fit + AICc 选择 (Ch1/Ch2)
    │       └─→ 得到分布 → 直接算 VaR/ES (Ch4/Ch5)
    │
    └─→ 多个资产? → 需要知道它们如何"一起动"
            │
            ├─→ 线性共动 → Covariance / Correlation (Ch2)
            │       │
            │       └─→ 用来 simulate 多变量 → Cholesky / PCA (Ch3)
            │
            └─→ 非线性依赖 → Copula (Ch5)
                    │
                    └─→ Copula simulation → Portfolio VaR/ES
```

---

# Chain 1: Return → Moments → Distribution → Risk

**核心问题**：一列 return 数据，最终变成 VaR/ES 数字

```
Raw prices (Problem 4)
    │
    │  r_t = (P_t - P_{t-1}) / P_{t-1}   [discrete]
    │  r_t = ln(P_t / P_{t-1})            [log]
    ▼
Returns 序列（一列数字）
    │
    │  chapter1.first4_moments()
    ▼
4 Moments: Mean, Variance, Skewness, Excess Kurtosis
    │
    │  为什么要算？
    │  ├─ Mean: expected return（中心在哪）
    │  ├─ Variance: 波动有多大（risk 的粗略度量）
    │  ├─ Skewness: 左尾 vs 右尾是否对称
    │  │    └─ 负 skew → 左尾更厚 → 大跌比大涨更可能 → 对 risk 管理很重要
    │  └─ Excess Kurtosis: 尾巴比 Normal 胖多少
    │       └─ > 0 → heavy tails → Normal 会低估极端风险
    ▼
那怎么选分布？
    │
    │  chapter1.fit_normal_vs_t_aicc()
    │
    ├─→ Normal(μ, σ): 只有 2 参数，尾巴薄（指数衰减 e^{-x²/2}）
    │       └─ 优点：简单、解析公式多
    │       └─ 缺点：低估 tail risk
    │
    └─→ t(ν, μ, σ): 3 参数，多了 ν（degrees of freedom）
            └─ ν 控制尾巴厚度：ν 越小 → 尾巴越厚
            └─ ν → ∞ 时退化为 Normal
            └─ 优点：更真实地捕捉 fat tails
            └─ 用 AICc 选：penalize 多出的参数，选 AICc 更小的
    │
    ▼
选好分布后 → 算 VaR / ES
    │
    ├─→ Parametric VaR (Ch4):
    │       Normal: VaR = -(μ + σ · z_α)
    │       t:      VaR = -t_α(ν) · σ + μ  → 用 stats.t.ppf()
    │
    ├─→ Parametric ES (Ch5):
    │       Normal: ES = -μ + σ · φ(z_α)/(1-α)
    │       t:      ES = 数值积分 → chapter5.calculate_es_t()
    │
    └─→ Empirical (非参数):
            VaR = -quantile(x, α)       ← 不需要假设任何分布
            ES  = -mean(x[x ≤ quantile]) ← 直接从数据算
```

**为什么要三种方法？**
- Parametric Normal：快，但假设太强（thin tails）
- Parametric t：更准，但依赖 MLE fit 的质量
- Empirical：无假设，但需要大样本（尾部数据少时不稳定）
- 考试通常要你**三个都算，然后比较和解释差异**

---

# Chain 2: 单变量 → 多变量 — 为什么需要 Covariance

**核心问题**：为什么不能对每个资产单独算风险然后加起来？

```
单个资产的 risk → σ_i（标准差）
    │
    │  但 portfolio 有多个资产...
    │  Portfolio return = Σ w_i · r_i
    ▼
Portfolio variance = ?
    │
    │  如果 assets 独立: σ²_p = Σ w_i² σ_i²     ← 简单但不真实
    │  实际上 assets 有相关性!
    │
    │  所以需要知道 Cov(r_i, r_j) for ALL pairs
    ▼
Covariance Matrix Σ
    │
    │  Σ_{ij} = Cov(r_i, r_j)
    │  对角线 = 各自的 variance
    │  非对角线 = 两两之间的 covariance
    │
    │  Portfolio variance = w^T Σ w  ← 这就是为什么需要完整的 Σ
    ▼
怎么估计 Σ？
    │
    ├─→ Equal-weight sample cov: np.cov(X, rowvar=False)
    │       └─ 所有历史数据同等权重
    │       └─ 问题：10 年前的数据和昨天一样重要？
    │
    ├─→ EW covariance (Ch3): chapter3.ew_covariance()
    │       └─ 更 recent 的数据权重更大
    │       └─ λ 控制 memory：λ = 0.97 → ~33 天 window
    │       └─ 为什么？因为 volatility 和 correlation 会随时间变化
    │       └─ RiskMetrics 的 λ = 0.94 是经典选择
    │
    └─→ Mixed-lambda (进阶): eu.ew_cov_mixed_lambda()
            └─ Variance 用 λ_var（如 0.97, 反应快）
            └─ Correlation 用 λ_corr（如 0.94, 更稳定）
            └─ 为什么分开？因为 volatility 变化快，correlation 变化慢
            └─ 最后 combine: Σ = D · R · D
```

---

# Chain 3: Covariance → Correlation — 为什么需要互相转换

```
Covariance Σ
    │
    │  问题：Σ_{ij} 的大小取决于 scale（单位）
    │  Cov(r_AAPL, r_MSFT) = 0.0003 → 这算大还是小？看不出来
    ▼
Correlation ρ = Σ_{ij} / (σ_i · σ_j)    ← 标准化到 [-1, +1]
    │
    │  eu.cov_to_corr(Σ)
    │
    │  ρ = +1: 完全同向
    │  ρ = 0:  线性无关
    │  ρ = -1: 完全反向
    ▼
为什么需要反向转换 Corr → Cov？
    │
    │  因为 mixed-lambda 方法：
    │  Step 1: EW(λ_var) → 得到 variance vector σ²
    │  Step 2: EW(λ_corr) → 得到 cov → cov_to_corr → Correlation R
    │  Step 3: 用 σ 和 R 重建 → Σ = D · R · D
    │
    │  eu.corr_to_cov(R, σ)
    ▼
另一个原因：Copula fitting
    │
    │  Copula 需要的是 Correlation（不是 Cov）
    │  而且要 Spearman correlation（not Pearson）
    │  为什么？因为 Spearman 是 rank-based，不受 marginal 形状影响
```

### Pearson vs Spearman — 什么时候用哪个？

```
Pearson correlation
    │  计算方式：基于原始值的 covariance 除以 std
    │  衡量的是：线性关系的强度
    │  缺点：受 outlier 影响大、只能捕捉线性
    │  用途：一般 covariance 估计、portfolio variance
    │
Spearman correlation
    │  计算方式：先把数据 rank 化（排序），再算 rank 的 Pearson
    │  衡量的是：单调关系（不一定线性）
    │  优点：robust to outliers、不受 marginal 变换影响
    │  用途：Copula fitting（因为 copula 就是在 rank 空间工作的）
```

---

# Chain 4: 真实数据的"脏活" — Missing Data → Pairwise → Not PSD

**核心问题**：为什么好好的 covariance matrix 会"坏掉"？

```
理想情况：n 个 assets，T 个时间点，所有数据都有
    │  → np.cov(X, rowvar=False) → 保证 PSD ✓
    │
但现实中...
    │
    ├─→ 有些资产某些天没有数据（停牌、不同交易所、上市时间不同）
    │       └─ X 矩阵里有 NaN
    ▼
怎么算带 missing 的 cov？
    │
    ├─→ 方法 A: 删掉任何有 NaN 的行（listwise deletion）
    │       └─ 问题：如果每列都有少量 NaN，可能删掉大部分数据
    │
    └─→ 方法 B: Pairwise covariance（chapter3.missing_cov()）
            └─ 对每一对 (i,j)，只用两者都有数据的行来算 Cov(i,j)
            └─ 好处：保留更多数据
            └─ 坏处：不同 pair 用了不同的 observation set
    │
    ▼
Pairwise 的后果 → 矩阵可能 NOT PSD!
    │
    │  为什么？
    │  PSD 要求 w^T Σ w ≥ 0 对所有 w
    │  但 pairwise 估计的 Σ 各元素来自不同的数据子集
    │  → 它们之间不一致 → 可能存在某个 w 使得 w^T Σ w < 0
    │  → 即某个 portfolio 有"负 variance" → 数学上不可能
    │
    │  怎么检测？
    │  np.linalg.eigvalsh(Σ).min() < 0  →  NOT PSD
    ▼
必须修复！（否则 Cholesky 会报错，simulation 做不了）
    │
    ├─→ near_psd (Rebonato-Jäckel): chapter3.near_psd()
    │       └─ 算法：eigendecomposition → 把负 eigenvalue 设为 0（或 ε）→ 重建
    │       └─ 如果是 correlation matrix，还要 rescale 使对角线 = 1
    │       └─ 简单快速，但不是最优的
    │
    └─→ Higham alternating projection: chapter3.higham_nearest_psd()
            └─ 算法：交替投影到两个集合
            │   Set A = PSD matrices
            │   Set B = 对角线 = 1 的 matrices（如果是 correlation）
            │   反复投影直到收敛
            └─ 数学证明：找到的是 Frobenius 范数意义下的 **最近** PSD matrix
            └─ Frobenius 距离 ||Σ_fixed - Σ_raw||_F 一定 ≤ near_psd 的
    │
    ▼
修复后的矩阵 → 可以安全做 Cholesky / PCA / simulation
```

**Problem 3 就是这条 chain 的考题**：给你一个 non-PSD 矩阵，走这条路。

---

# Chain 5: Covariance → Simulation — 为什么需要模拟？怎么模拟？

**核心问题**：有了 Σ，为什么不直接用公式算 portfolio risk？

```
如果 portfolio 是线性的（只有股票，no options）：
    │  Portfolio variance = w^T Σ w → 可以直接算 VaR (Delta-Normal)
    │  chapter4.var_delta_normal()
    │
但如果 portfolio 有非线性（options, structured products）：
    │  或者你想要 ES（需要整个 tail 的分布，不只是一个 quantile）
    │  或者你想要 empirical distribution 而不是假设 Normal
    ▼
需要 Monte Carlo Simulation
    │
    │  目标：生成 n_sim 个 "多资产 return 的 scenario"
    │  每个 scenario = (r_1, r_2, ..., r_n) 的一组实现
    │  这些 scenarios 需要保持正确的 correlation 结构
    ▼
方法 1: Cholesky Decomposition (Ch3)
    │
    │  原理：Σ = L · L^T （L 是下三角）
    │  模拟：Z ~ N(0, I)  →  X = L·Z + μ
    │  结果：X 的 covariance = L·E[ZZ^T]·L^T = L·I·L^T = Σ ✓
    │
    │  chapter3.simulate_normal_cholesky(Σ, nsim)
    │
    │  优点：精确保留完整 covariance 结构
    │  缺点：需要 Σ 是 PSD（否则 Cholesky 分解失败）
    │  缺点：维度高时计算量大
    ▼
方法 2: PCA Simulation (Ch3)
    │
    │  原理：Σ = V Λ V^T （eigendecomposition）
    │        只保留 top k 个 eigenvalue/eigenvector
    │  模拟：Z ~ N(0, I_k)  →  X = V_k · √Λ_k · Z
    │
    │  chapter3.simulate_pca(Σ, nsim, pct_exp=0.75)
    │
    │  优点：降维 → 更快
    │  缺点：丢失了小 eigenvalue 对应的 variance → Frobenius error 更大
    │
    │  怎么选 k？看 cumulative explained variance
    │  chapter3.pca_cumulative_variance(Σ)
    │  → 选 k 使得 cumulative ≥ 目标（如 75%, 90%）
    ▼
Cholesky vs PCA 怎么选？
    │
    ├─ n 小（如 5 个资产）→ Cholesky（已经很快，没必要降维）
    └─ n 大（如 500 个资产）→ PCA（降维显著加速）
    │
    │  eu.pipeline_cholesky_vs_pca() 可以直接比较两者
    │
    ▼
模拟出来的 scenarios → 计算 portfolio PnL → 取 quantile = VaR, tail mean = ES
```

---

# Chain 6: 为什么 Normal 不够 → Copula 的动机

**核心问题**：Cholesky/PCA 假设 multivariate Normal，但现实不是

```
Multivariate Normal 的假设：
    │  1. 每个 marginal 是 Normal
    │  2. 依赖结构完全由 linear correlation 决定
    │  3. 没有 tail dependence（极端事件不会同时发生得更频繁）
    │
但金融数据的现实：
    │  1. Marginals 有 fat tails（excess kurtosis > 0）→ t 比 Normal 好
    │  2. 相关性在 crisis 时会增加（correlation breakdown）
    │  3. 存在 tail dependence（2008 年所有资产同时暴跌）
    ▼
解决方案：Copula（Ch5）
    │
    │  Sklar's Theorem:
    │  任何 joint distribution F(x₁,...,xₙ) 都可以分解为：
    │  F(x₁,...,xₙ) = C( F₁(x₁), ..., Fₙ(xₙ) )
    │
    │  C = Copula（纯依赖结构）
    │  Fᵢ = 各自的 marginal CDF（各自的分布形状）
    │
    │  → 把"每个资产长什么样"和"它们怎么一起动"分开建模
    ▼
Gaussian Copula 的工作流程：
    │
    │  Step 1: 对每个资产单独 fit marginal (Normal 或 t)
    │          chapter5.fit_copula_marginals()
    │          → 每个资产可以用不同的分布！
    │
    │  Step 2: 用 Spearman correlation 估计 copula 的依赖结构
    │          chapter5.fit_gaussian_copula_corr(method="spearman")
    │          → 为什么 Spearman? 因为 copula 在 rank 空间工作
    │            Spearman 就是 rank correlation，天然匹配
    │
    │  Step 3: 模拟
    │          a) 从 MVN(0, Σ_copula) 生成 correlated Normal variables
    │          b) 通过 Φ(z) 变成 correlated Uniform(0,1)
    │          c) 通过各自的 F_i^{-1}(u) 变回各资产的 return
    │          chapter5.simulate_copula_mixed()
    │
    │  Step 4: sim returns × holdings × prices = PnL
    │          eu.portfolio_pnl()
    │
    │  Step 5: 从 PnL 分布取 VaR / ES
    │          eu.var_es_from_pnl()
```

---

# Chain 7: VaR 的三大方法 — 它们的联系和区别

```
所有 VaR 方法的本质都一样：
    "找到 return 分布的 α-quantile，取负号"
    │
    │  区别在于：用什么分布？
    ▼
方法 1: Parametric Normal (Ch4)
    │  假设 R ~ N(μ, σ²)
    │  VaR = -(μ + σ · Φ⁻¹(α))
    │  = -(μ - 1.645σ)  当 α = 0.05
    │
    │  chapter4.var_normal()
    │
    │  优点：只需 μ, σ，公式简单
    │  缺点：假设 Normal → 低估 tail risk
    │
方法 2: Parametric t (Ch4/Ch5)
    │  假设 R ~ t(ν, μ, σ)
    │  VaR = -(μ + σ · t_ν⁻¹(α))
    │
    │  eu.var_t()
    │
    │  优点：捕捉 fat tails
    │  缺点：需要 MLE fit ν（额外参数）
    │
方法 3: Historical / Empirical (Ch4)
    │  不假设任何分布
    │  直接排序历史 returns，取第 α×n 个
    │
    │  chapter4.var_historical()
    │
    │  优点：model-free
    │  缺点：受限于历史样本量，尾部数据少
    │
方法 4: Monte Carlo (Ch4)
    │  模拟 → 得到 PnL 分布 → 取 quantile
    │
    │  chapter4.var_monte_carlo()
    │
    │  优点：可以处理任何分布 + 非线性 portfolio
    │  缺点：计算量大，有 sampling noise
    │
方法 5: Delta-Normal (Ch4)
    │  对 portfolio 做线性近似
    │  portfolio PnL ≈ Σ δ_i · ΔS_i
    │  → portfolio 的 PnL ~ Normal → 直接算 VaR
    │
    │  chapter4.var_delta_normal()
    │
    │  优点：多资产 + closed-form
    │  缺点：只对 linear portfolio 准确

对比链:
    Parametric Normal < Parametric t < Monte Carlo (in tail accuracy)
    Monte Carlo < Parametric Normal (in speed)
    Monte Carlo ≈ Copula Simulation (but copula handles non-Normal marginals)
```

---

# Chain 8: VaR → ES — 为什么 VaR 不够？

```
VaR 的问题：
    │
    ├─ 问题 1: VaR 只看"门槛"，不看"门槛后面有多深"
    │       95% VaR = $100 → 5% 的日子亏 > $100
    │       但是平均亏 $110 还是 $500？VaR 不告诉你
    │
    ├─ 问题 2: VaR 不是 coherent risk measure
    │       具体来说，VaR 违反 sub-additivity:
    │       VaR(A + B) 可能 > VaR(A) + VaR(B)
    │       → 合并 portfolio 后风险反而变大？不合理
    │       → 这会惩罚 diversification
    │
    └─ 问题 3: 监管已经从 VaR 转向 ES
            Basel III → FRTB → 要求用 ES
    ▼
Expected Shortfall 解决这些问题：
    │
    │  ES = E[Loss | Loss > VaR] = tail 的平均深度
    │
    ├─ 解决问题 1: ES 看完整的 tail shape，不只是入口
    ├─ 解决问题 2: ES 是 coherent (满足 sub-additivity)
    │       ES(A + B) ≤ ES(A) + ES(B)  永远成立
    │       → diversification 在 ES 下永远有好处
    └─ 解决问题 3: 符合新监管要求
    ▼
Coherent Risk Measure 的 4 个 Axioms:
    │
    │  (1) Monotonicity: X 总是比 Y 亏得少 → ρ(X) ≤ ρ(Y)
    │  (2) Sub-additivity: ρ(X+Y) ≤ ρ(X) + ρ(Y)   ← VaR 在这里 fail
    │  (3) Positive Homogeneity: ρ(λX) = λ·ρ(X)
    │  (4) Translation Invariance: ρ(X + cash) = ρ(X) - cash
    │
    │  VaR:  ✓ ✗ ✓ ✓  → NOT coherent
    │  ES:   ✓ ✓ ✓ ✓  → Coherent ✓
```

---

# Chain 9: Time Series → GARCH — 为什么 volatility 会变？

```
金融 returns 的一个 stylized fact:
    │  今天波动大 → 明天很可能也波动大
    │  这叫 "volatility clustering"（波动率聚集）
    │
    │  但 returns 本身通常不可预测（接近 white noise）
    │  可预测的不是 return 的方向，而是 return 的波动幅度
    ▼
怎么建模？先从简单的开始：
    │
    ├─→ AR(p) model: 今天的 return 依赖于过去 p 天
    │       X_t = φ₁X_{t-1} + ... + φ_pX_{t-p} + ε_t
    │       识别方法：PACF 在 lag p 后截断
    │       chapter2.fit_ar_model(), chapter2.select_best_ar_order_aicc()
    │
    ├─→ MA(q) model: 今天的 return 依赖于过去 q 天的 shock
    │       X_t = ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}
    │       识别方法：ACF 在 lag q 后截断
    │       chapter2.fit_ma_model()
    │
    └─→ 问题：AR/MA 模型的 variance 是常数！
            它们不能解释 volatility clustering
    ▼
GARCH(1,1) — 让 variance 也有 dynamics
    │
    │  σ²_t = ω + α · r²_{t-1} + β · σ²_{t-1}
    │
    │  ω: baseline variance
    │  α: 昨天的 shock 有多大影响（shock 越大 → 今天 variance 越大）
    │  β: 昨天的 variance 持续到今天多少（persistence）
    │  α + β < 1: 保证 stationary
    │
    │  Unconditional (long-run) variance:
    │  σ² = ω / (1 - α - β)
    ▼
GARCH 和 EW Covariance 的关系？
    │
    │  其实 EWMA variance (RiskMetrics) 就是 GARCH 的特殊情况：
    │  令 ω = 0, α = 1-λ, β = λ → σ²_t = (1-λ)r²_{t-1} + λσ²_{t-1}
    │  所以 EW covariance 本质上是一个 "integrated GARCH"
    │  区别：GARCH 有 mean reversion (unconditional variance)
    │        EWMA 没有 mean reversion (ω = 0 → no anchor)
```

---

# Chain 10: Regression — OLS → t-Regression → 为什么用 t？

```
线性回归 y = Xβ + ε 的本质：
    │  给定 X，预测 y 的 conditional mean
    │
    │  E[y | X] = Xβ → β tells us the "sensitivity"
    ▼
OLS (Ordinary Least Squares):
    │  假设 ε ~ N(0, σ²)  → 即 error 是 Normal
    │  minimize Σ(y_i - x_i^T β)² → β̂ = (X^T X)^{-1} X^T y
    │
    │  chapter2.mle_regression()
    │  （OLS = Normal MLE，两者给出相同的 β̂）
    ▼
但如果 errors 有 fat tails?
    │  ε ~ t(ν, 0, σ) instead of Normal
    │  → OLS 仍然 unbiased，但不再是 efficient
    │  → t-regression via MLE 更好
    │
    │  eu.mle_regression_t()
    │  → 同时估计 β, σ, ν
    │  → 用 AICc 比较 OLS(Normal) vs t-regression
    ▼
为什么 care？
    │  如果 residuals 有 fat tails，Normal 假设下的
    │  confidence intervals 太窄（低估不确定性）
    │  t-regression 给出更 honest 的 uncertainty quantification
```

---

# Chain 11: Conditional Distribution — 为什么需要？

```
已知一个资产的 return → 对另一个资产的预期会改变吗？
    │
    │  如果 X₁, X₂ 独立 → 不会改变 → E[X₂|X₁=x] = E[X₂]
    │  如果 correlated → 会改变!
    ▼
Bivariate Normal Conditional Distribution:
    │
    │  (X₁, X₂) ~ N(μ, Σ)
    │  已知 X₁ = x₁
    │
    │  E[X₂ | X₁ = x₁] = μ₂ + ρ(σ₂/σ₁)(x₁ - μ₁)
    │      → 条件均值线性依赖于 x₁
    │      → ρ > 0: X₁ 高 → X₂ 的预期也高
    │
    │  Var(X₂ | X₁ = x₁) = σ₂²(1 - ρ²)
    │      → 条件 variance 不依赖于 x₁ 的值！
    │      → 只取决于 ρ：|ρ| 越大 → 条件 variance 越小 → 知道 X₁ 后不确定性减少
    │
    │  chapter3.conditional_bivariate_stats()
    ▼
Risk management 应用：
    │  "如果 SPY 今天跌了 3%，AAPL 的预期 return 是多少？"
    │  → 用 conditional mean
    │  "知道 SPY 的 return 后，AAPL 的 uncertainty 减少了多少？"
    │  → 看 conditional variance vs unconditional variance
    │  → 减少比例 = ρ²（即 R-squared in regression!）
```

---

# Chain 12: 所有 Chain 汇合 — Copula Pipeline（考试大题）

```
                    Prices (Problem 4)
                         │
                    return_calculate()
                         │
                    Returns Matrix R
                   ╱     │     ╲
                  ╱      │      ╲
    Per-asset     Per-asset    Spearman
    moments       fit (N/t)   correlation
    (Chain 1)     (Chain 1)    (Chain 3)
         │            │            │
         │     fit_copula_     fit_gaussian_
         │     marginals()     copula_corr()
         │            │            │
         │            ╰─────┬──────╯
         │                  │
         │     simulate_copula_mixed()
         │                  │
         │           Simulated Returns
         │                  │
         │          ×  holdings × prices
         │                  │
         │            Portfolio PnL
         │                  │
         │         ┌────────┴────────┐
         │    var_es_from_pnl()  per-asset VaR/ES
         │         │                 │
    Moments   Portfolio VaR/ES  Diversification
    Report    Report             Benefit Report
```

---

# Chain 13: 所有 Frobenius 距离出现的地方 — 它衡量什么？

```
Frobenius norm ||A - B||_F = sqrt( Σ_{ij} (a_{ij} - b_{ij})² )
    │
    │  就像向量的欧几里得距离，但用在矩阵上
    │  越小 → 两个矩阵越"像"
    ▼
出现在 3 个场景：
    │
    ├─→ PSD fix quality:  ||Σ_fixed - Σ_raw||_F
    │       → Higham ≤ near_psd（Higham 是 optimal）
    │       → 越小越好（修改越少）
    │
    ├─→ Simulation recovery: ||Σ_simulated - Σ_input||_F
    │       → 衡量 simulation 是否正确恢复了 covariance
    │       → Cholesky ≈ 0（exact），PCA > 0（approximate）
    │
    └─→ Cholesky vs PCA comparison:
            → 两者的 Frobenius 谁更小？
            → Cholesky 一定更小（因为它用了全部 eigenvalues）
            → PCA 的 Frobenius 取决于 pct_exp（保留的 variance 比例）
```

---

# 速记口诀

```
"一列算 moments，fit 两个选 AICc"          → Chain 1 (单变量)
"多列要 cov，cov 要 PSD"                   → Chain 2 + 4
"missing 数据用 pairwise，pairwise 可能坏"  → Chain 4
"坏了用 Higham，Higham 最优"               → Chain 4
"cov 有了做 Cholesky，大矩阵用 PCA"        → Chain 5
"Normal 不够用 Copula，Copula 分 marginal 和 dependence" → Chain 6
"VaR 看门槛，ES 看深度"                    → Chain 8
"VaR 不 coherent，ES 才 coherent"          → Chain 8
"volatility cluster → GARCH, EWMA 是特例"  → Chain 9
"Spearman for copula, Pearson for 其他"     → Chain 3
```
