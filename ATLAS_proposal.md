# ATLAS: Autonomous Theory Learning through Adaptive Search

## A Framework for AI-Driven Discovery of Physical Laws and Mathematical Structures

**Version 1.0 — For Discussion**

---

## 1. Introduction

### 1.1 Problem Statement

现有 AI 物理发现系统（AI Feynman, PySR, AI-Newton 等）均在**固定的数学框架**内搜索公式。它们假定物理量是实数、关系是确定性函数、状态由输入参数直接决定。当数据来自需要不同数学结构的领域（如概率性、离散性、高维状态空间）时，这些系统会系统性失败，且无法诊断失败的原因或自主扩展自身的描述能力。

本项目提出 ATLAS——一个能够自动检测自身数学框架的不足、从数据中发现新的数学描述空间、并用该空间扩展自身能力的物理发现系统。

### 1.2 Core Claim

> ATLAS 是第一个能从实验数据的拟合失败中自动发现状态空间的几何结构，并将该结构编码为新的符号描述类型、从而扩展自身数学语言的系统。

### 1.3 Positioning

| 系统 | 发现层次 | 框架固定? | 能否自扩展? |
|------|---------|----------|------------|
| PySR / AI Feynman | 公式 (y=f(x)) | 是 (实数+基本运算) | 否 |
| AI-Newton | 概念+定律 | 是 (微分多项式) | 否 |
| SciNet | 表征 (瓶颈维度) | 是 (神经网络) | 否 |
| HNN / LNN | 参数 (H/L 的具体形式) | 是 (哈密顿/拉格朗日力学) | 否 |
| DreamCoder | 库函数 (新子程序) | 部分 (添加函数，不添加类型) | 部分 |
| **ATLAS** | **几何结构+定律+概念** | **否 (DSL 可成长)** | **是 (RGDE)** |

### 1.4 Key References

- AI-Newton: Fang et al., arXiv:2504.01538 (2025) — 概念驱动的经典力学定律发现
- MASS: arXiv:2504.02822 (2025) — 多个独立 AI 科学家的理论一致性
- SciNet: Iten et al., PRL 124 (2020) — 信息瓶颈发现物理概念
- PySR: Cranmer, arXiv:2305.01582 (2023) — 高性能符号回归
- AI Feynman: Udrescu & Tegmark, Science Advances (2020) — 递归分解符号回归
- DreamCoder: Ellis et al., PLDI (2021) — 通过库学习成长 DSL
- GPT Tomography: Mazurek et al., PRX Quantum 2 (2021) — 从数据中重构状态空间几何
- Hardy: quant-ph/0101012 (2001) — 从公理推导量子力学

---

## 2. System Architecture

### 2.1 Overview

ATLAS 由一个主循环和四个基础模块组成：

```
┌──────────────────────────────────────────────────────────────┐
│                        ATLAS Main Loop                        │
│                                                               │
│   DSL₀ = {ℝ, +, -, ×, ÷, sin, cos, exp, log, ^}            │
│                                                               │
│   repeat until convergence:                                   │
│     ① Solve    — SR in current DSL                           │
│     ② Extract  — concept abstraction (success-driven)        │
│     ③ Diagnose — failure analysis                            │
│     ④ Extend   — RGDE: data-driven DSL extension            │
│     ⑤ Unify    — cross-experiment laws & constants           │
│                                                               │
│   Convergence = all experiments R² > θ on held-out test set  │
└──────────────────────────────────────────────────────────────┘

Foundation Modules:
  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────────┐
  │ PySR   │  │ SciNet │  │ PSLQ   │  │ Environments │
  │ (SR)   │  │ (表征)  │  │ (常数)  │  │ (模拟实验)    │
  └────────┘  └────────┘  └────────┘  └──────────────┘
```

### 2.2 Design Principles

1. **最小起点原则**：DSL 从最简单的实数算术开始，不预设任何物理或量子力学知识
2. **失败驱动扩展**：只有当 SR 在当前 DSL 下系统性失败时，才触发 DSL 扩展
3. **数据驱动类型发现**：新的 DSL 类型从 SciNet 学习的表征中提取，不从预定义列表选择
4. **跨实验压缩**：最优框架 = 能同时描述所有实验的最短统一程序
5. **零物理先验**：不使用 LLM（知识泄露）、不编码特定物理定律、不预设量子力学结构

---

## 3. Five-Step Main Loop

### 3.1 Step ① Solve — 在当前 DSL 下搜索定律

**目标：** 对每个实验，用符号回归找到数据的最优符号描述。

**方法：**
- 主引擎：PySR（多种群进化 + BFGS 常数优化 + Pareto 前沿）
- 辅助：AI Feynman 式递归分解（NN 预言机检测对称性和可分离性，降低维度后再 SR）
- AI-Newton 式受控变量分析（逐个扫描 knob，检测线性/周期/阈值依赖）

**输入：** 当前 DSL 的运算符集合 + 实验数据（knob → detector）

**输出：** 每个实验的 Pareto 最优公式集（accuracy vs complexity），或标记为"失败"

**成功标准：** Pareto 前沿上存在 R² > 0.95（留出测试集）且描述长度合理的公式

**失败标准：** 最佳 R² < 0.8，或需要极高复杂度（MDL 爆炸）才能达到 R² > 0.9

### 3.2 Step ② Extract — 从成功中提取概念

**目标：** 从成功发现的公式中抽取可复用的子结构，加入 DSL 作为新的库函数。

**方法：DreamCoder 式反碎片化（anti-unification）+ MDL 压缩**

```
1. 统计所有成功公式中出现的子表达式及其频率
2. 对每个候选子表达式:
   savings = 出现次数 × 子表达式大小
   cost    = 定义概念本身的描述长度
   如果 savings > cost → 接受为新概念
3. 用新概念重写所有公式 → 公式变短
4. 将概念作为新运算符加入 DSL

例: cos²(C·x) 在 3 个实验的公式中出现
    → 提取 concept_cos2(x) = cos²(x) 作为库函数
    → 后续 SR 可直接搜索含 concept_cos2 的表达式
```

**关键：** 概念提取是纯语法操作（频繁子表达式提取），不注入物理语义。与 DreamCoder 的库学习在方法论上完全对齐。

### 3.3 Step ③ Diagnose — 从失败中诊断 DSL 缺陷

**目标：** 对 SR 失败的实验，分析失败原因，确定 DSL 需要哪方面的扩展。

**五种诊断信号（全部从数据中可测量，不需要物理先验）：**

| 编号 | 诊断 | 检测方法 | 数据信号 |
|------|------|---------|---------|
| D1 | 随机性 | 相同 knob 设置重复实验 100 次 | 输出方差 / 均值 > 阈值 |
| D2 | 离散性 | 对 detector 输出做聚类（DBSCAN） | 聚类数 N < 10 |
| D3 | 维度不足 | SciNet 渐进式瓶颈搜索 | K_bottleneck > n_active_knobs |
| D4 | 残差结构 | 对最佳拟合残差做 FFT / 自相关 | 残差有非白噪声结构 |
| D5 | 跨实验不一致 | 两实验对同一实体的描述对比 | 常数或函数结构矛盾 |

**D3 的具体实现——渐进式 K 测量：**

```
对每个失败实验:
1. 从 1 个 knob 开始, 逐步增加 knob 到 SciNet 输入
2. 每步训练 SciNet, 用 AIC 选择最优瓶颈维度 K
3. 记录 K 随 knob 数的变化
4. 如果 K > n_knobs → 系统有"隐藏自由度"(超出输入参数数目)
5. K 值本身就是发现 — 不需要解释为任何特定理论
```

### 3.4 Step ④ Extend — RGDE: 数据驱动的 DSL 扩展

**这是 ATLAS 的核心创新。**

RGDE (Representation-Grounded DSL Extension) 将 SciNet 学习到的黑箱表征转化为可解释的 DSL 类型。

**RGDE 管道（六步）：**

```
Step 4a: SciNet 学习最小充分表征
  训练 encoder-decoder, 瓶颈维度 = D3 确定的 K
  encoder: knob_settings → z ∈ ℝᴷ
  decoder: (z, question_params) → predicted_detector

Step 4b: 对 encoder 做符号回归
  对每个瓶颈维度 k = 1..K:
    z_k = SR(knob_settings)
  得到 K 个符号公式, 描述"knob 如何映射到内部状态"

Step 4c: 发现瓶颈空间的约束
  对所有编码后的点 {z_i} ∈ ℝᴷ:
    搜索 g(z₁, ..., zₖ) ≈ const 的代数关系
    方法: 对 z 的低阶多项式组合做 SR
  例如: 发现 z₁² + z₂² + z₃² ≤ 1 → 球约束

Step 4d: 构建新 DSL 类型
  NewType = {
    name: "State_ENV_XX",
    dimension: K,
    encoding: [z₁ = f₁(knobs), ..., zₖ = fₖ(knobs)],  // Step 4b 的结果
    constraints: [g(z) ≤ c, ...],  // Step 4c 的结果
  }

Step 4e: 对 decoder 做符号回归
  prediction = SR(z₁, ..., zₖ, question_params)
  得到一个在新类型空间中表达的定律

Step 4f: Pareto 评估
  比较:
    扩展前: 最佳 R² (在旧 DSL 下)
    扩展后: 新定律的 R² (在新 DSL 下)
    代价:   DSL 复杂度增加量 (新类型的描述长度)
  如果在 Pareto 前沿上 → 接受扩展
```

**RGDE 的三个关键性质：**

1. **开放性：** 新类型来自数据（SciNet 表征），不来自预定义列表。如果数据需要一个 7 维球面——SciNet 会发现 K=7，SR 会提取对应约束。不限于已知数学结构。

2. **可解释性：** 每一步都有符号输出——encoder 公式、约束方程、decoder 定律。不是黑箱。

3. **可验证性：** 新类型的有效性通过 Pareto 评估判定——拟合必须改善，且改善必须超过复杂度增加的代价。

### 3.5 Step ⑤ Unify — 跨实验统一

**三个层次的统一：**

**层次 1：常数统一（PSLQ）**

```
从各实验独立发现的公式中提取所有数值常数:
  {C₁, C₂, C₃, ...}

PSLQ 算法在对数空间中搜索整数关系:
  n₁·log|C₁| + n₂·log|C₃| + ... ≈ 0
  即: C₁^n₁ · C₃^n₂ · ... ≈ 1

从整数关系中提取最小基本常数集:
  {UC₁, UC₂, ...} 使得所有 Cᵢ = UCⱼ 的幂积

用统一常数重写所有公式
```

**层次 2：概念统一（跨实验的共享函数构件）**

```
检查 Step ② 提取的概念是否跨实验出现:

例如: concept_cos2 (即 cos²(·)) 在以下位置出现:
  - 确定性干涉公式: I(x) = C·cos²(C'·x)
  - 概率分布公式:   P(x) = C''·cos²(C'''·x)

如果 C' ≈ C''': → 标记为 "distributional link"
  含义: 确定性波动强度的函数形式 = 概率分布的函数形式
  这是一个纯粹的数据发现，等价于 Born 规则的可观测推论
```

**层次 3：类型统一（跨实验的状态空间同构）**

```
检查不同实验的 RGDE 类型是否有结构相似性:

例如:
  Stern-Gerlach 实验的 State_SG = {z ∈ ℝ³ : |z| ≤ 1} (球)
  某其他量子实验的 State_XX = {z ∈ ℝ³ : |z| ≤ 1} (也是球)
  → 两者状态空间同构
  → 可能共享同一个底层理论

方法: 比较不同实验 NewType 的:
  - 维度 K
  - 约束方程的结构
  - encoder/decoder 公式的形式
```

---

## 4. Environment Layer

### 4.1 Design Principles

严格遵循反作弊规范：

| 规则 | 要求 |
|------|------|
| 输入 | 只有归一化旋钮 knob_0..n, 全部映射到 [0, 1] |
| 输出 | 只有原始探测器读数 detector_0..m |
| 禁止 | 布尔开关、预计算特征、物理语义命名 |
| 允许 | 对象标签（哪些实验共享同一 entity）、旋钮类型（continuous/discrete） |
| 归一化 | 独立且不可逆，系统无法从数值量级反推物理量 |

### 4.2 Experiment Suite

**量子实验（目标现象）：**

| ID | 实验 | 输入 | 输出 | 揭示的关键现象 |
|----|------|------|------|---------------|
| ENV-01 | 光电效应 | 3 continuous + 1 discrete | 1 scalar | 能量量子化，截止频率 |
| ENV-02 | 康普顿散射 | 2 continuous | 2 scalar | 光子动量 |
| ENV-03 | 电子衍射 | 2 continuous + 1 discrete | 1 array | 物质波 |
| ENV-04 | 双缝干涉 | 3 continuous + 1 integer | 1 array | 波粒二象性 |
| ENV-05 | 黑体辐射 | 2 continuous | 1 scalar | 普朗克分布 |
| ENV-06 | 氢原子光谱 | 2 continuous | 1 list | 能级量子化 |
| ENV-07 | Stern-Gerlach | 2 continuous + 1 integer | 1 array | 自旋量子化 |

**经典对照（不需要量子解释）：**

| ID | 实验 | 作用 |
|----|------|------|
| ENV-08 | 经典水波干涉 | 波动性对照（无粒子性） |
| ENV-09 | 弹性碰撞 | 粒子性对照（无波动性） |
| ENV-10 | 弹簧振动 | 经典力学对照 |

**干扰项（测试辨别力）：**

| ID | 实验 | 作用 |
|----|------|------|
| ENV-11 | 自由落体 | 系统不应将其与量子现象混淆 |
| ENV-12 | 热传导 | 涉及 k_B 但非量子 |

### 4.3 Example: Clean Double-Slit Interface

```python
class CleanDoubleSlitEnv:
    def get_schema(self):
        return {
            "env_id": "ENV_04",
            "entities": ["entity_A"],
            "inputs": {
                "knob_0": {"range": [0, 1], "type": "continuous"},  # 缝宽(归一化)
                "knob_1": {"range": [0, 1], "type": "continuous"},  # 缝间距(归一化)
                "knob_2": {"range": [0, 1], "type": "continuous"},  # 源参数(归一化)
                "knob_3": {"range": [1, 1000000], "type": "integer"},  # 源强度
            },
            "outputs": {
                "detector": {"type": "array_1d", "length": 1000}
                # 唯一输出: 探测屏原始读数
                # 低 knob_3: 稀疏离散尖峰
                # 高 knob_3: 平滑干涉图案
                # 系统需要自己发现这个转变
            }
        }
```

**注意：没有"单粒子模式"开关。** 系统通过扫描 knob_3 的全范围自己发现低强度时输出变为稀疏离散事件。

---

## 5. Anti-Cheating Protocol

### 5.1 Design-Time Checks

| 检查项 | 要求 | 状态 |
|--------|------|------|
| 无 LLM | 系统中无大语言模型 | ✅ |
| 无 Hardy r 公式 | 不使用 K=N^r 分类理论 | ✅ |
| 无 \|ψ\|² 预设 | 概率映射由 SR 自由搜索 | ✅ |
| 无复数运算符 | 初始 DSL 不含 complex_abs 等 | ✅ |
| 新类型来自数据 | RGDE 从 SciNet 表征提取，非预定义列表 | ✅ |
| 初始运算符通用性 | {sin,cos,exp,log,+,-,×,÷,^} 与 AI Feynman 一致 | ✅ |
| 环境接口匿名化 | knob/detector 命名，归一化，无物理语义 | ✅ |

### 5.2 Runtime Checks

| 检查项 | 方法 |
|--------|------|
| 扩展由数据触发 | 每个 DSL 扩展必须关联到可量化的诊断信号（方差>阈值, K>预期等） |
| Pareto 评估 | 每个扩展必须在留出测试集上通过拟合改善 vs 复杂度增加的评估 |
| 进化谱系 | 记录 DSL 每一步扩展的完整历史和触发原因 |

### 5.3 Validation Tests

| 测试 | 方法 | 预期结果 |
|------|------|---------|
| **替代物理 (h→2h)** | 修改环境中的 h 值，重新运行全流程 | 发现不同的常数值，相同的公式结构 |
| **经典极限 (h→0)** | 将 h 设为极小值 | 不触发概率扩展，不发现量子特征 |
| **随机数据** | 用随机噪声替换实验数据 | 不发现任何有意义的公式或类型 |
| **经典交叉验证** | 用经典力学数据（F=ma 等）运行同一系统 | 发现经典定律，不触发 RGDE |
| **非量子球面** | 用经典陀螺等具有球面状态空间的系统 | 发现球面几何，但不声称是量子的 |
| **多种子一致性** | 20+ 独立种子运行 | >60% 种子收敛到等价理论 |

---

## 6. Expected Discovery Pathway

### 6.1 Epoch 0: Classical DSL

```
DSL = {ℝ, +, -, ×, ÷, sin, cos, exp, log, ^}

① Solve:
  ENV-01 (光电效应):    meter = C₁·max(knob_0 - C₂, 0)·knob_1    ✓
  ENV-02 (康普顿散射):  Δdetector = C₃·(1 - cos(knob_1))          ✓
  ENV-04 (双缝, knob_3大): detector ∝ cos²(C₄·x)                 ✓
  ENV-05 (黑体辐射):    detector = C₅·f³/(exp(C₆·f/T) - 1)       ✓ (如果SR够强)
  ENV-08 (水波干涉):    detector ∝ cos²(C₇·x)                    ✓
  ENV-10 (弹簧):        detector = C₈·cos(C₉·t)                   ✓

  ENV-04 (双缝, knob_3小): ✗ (R²=0.15, 输出太稀疏)
  ENV-07 (Stern-Gerlach):  ✗ (输出只有2个离散值)

② Extract:
  cos²(·) 在 ENV-04, ENV-02, ENV-08 的公式中频繁出现
  → 提取 concept_cos2 作为库函数

③ Diagnose:
  ENV-04 (低强度): D1 → stochastic (重复实验方差高)
  ENV-07: D1 → stochastic, D2 → discrete (N=2)
```

### 6.2 Epoch 1: Probabilistic Extension

```
DSL += {concept_cos2, prob_distribution}
(prob_distribution 由 D1 触发, 允许 SR 搜索 P(y|x) 而非 y=f(x))

① Solve:
  ENV-04 (低强度): 概率SR → P(x) = C₁₀·cos²(C₁₁·x)             ✓
  ENV-07: 概率SR → P(up|θ) = cos²(C₁₂·knob_0)                   ✓

② Extract:
  cos² 同时出现在确定性公式和概率分布中!
  → concept_cos2 是跨确定性/概率性的通用构件

③ Diagnose:
  ENV-07 对多个 knob 设置组合: D3 → K=3 > N-1=1

④ Extend (RGDE on ENV-07):
  SciNet bottleneck = 3:
    encoder: (knob_settings) → (z₁, z₂, z₃)
    SR on encoder: z₁ = f₁(knobs), z₂ = f₂(knobs), z₃ = f₃(knobs)
    constraint: z₁² + z₂² + z₃² ≤ 1 (球)
    SR on decoder: P(up) = (1 + z₁·sin(θ)cos(φ) + z₂·sin(θ)sin(φ) + z₃·cos(θ))/2

  → NewType: BlochState = {z ∈ ℝ³ : |z| ≤ 1}
  → Law: P(outcome|state, measurement) = 线性函数(state · measurement_vector)

⑤ Unify:
  常数: PSLQ 发现 C₁ ≈ C₃ ≈ C₆ 共享因子 → UC₁ ≈ 6.626e-34
  概念: C₄(干涉强度) ≈ C₁₁(概率分布) → distributional link
  类型: ENV-07 的 BlochState 维度 K=3 > 经典预期 N-1=1
```

### 6.3 Final Output

```
DSL_final = {ℝ, +, -, ×, ÷, sin, cos, exp, log, ^,
             concept_cos2, prob_distribution, BlochState}

Discovered Laws:
  E = UC₁·f - W                                    [光电效应]
  Δλ = UC₁/(m·c)·(1-cosθ)                          [康普顿散射]
  I(x) = I₀·cos²(πd·x/(UC₁·L/p))                  [干涉强度]
  P(x) ∝ cos²(πd·x/(UC₁·L/p))                     [粒子概率]
  B(f,T) = 2UC₁f³/c²·1/(exp(UC₁f/kT)-1)           [黑体辐射]
  P(outcome|state,measurement) = affine(state·m)    [量子测量]

Discovered Constants:
  UC₁ = 6.626e-34 (跨 5+ 个实验的通用常数)

Discovered Operational Facts:
  F1: "某些实验输出是概率性的 (方差 > 0)"
  F2: "概率分布形状 = 确定性波动强度分布 (P ∝ I)"
  F3: "所有这些概率性现象共享同一常数 UC₁"
  F4: "某些二态系统需要三维状态空间 (K=3, 球约束)"

F1 + F2 + F3 + F4 = 波粒二象性的操作性核心
```

---

## 7. Honest Limitations

### 7.1 能发现的

| 发现 | 方法 | 置信度 |
|------|------|--------|
| E=hf 等单实验公式 | PySR (已在同等难度上验证) | 高 |
| h 是通用常数 | PSLQ (数论算法) | 高 |
| 某些系统是概率性的 | 方差检测 | 高 |
| P(x) ∝ I(x) | 概念关联 (常数匹配) | 中 |
| Bloch 球几何 | RGDE (SciNet→SR) | 中 (需 P0 验证) |
| K > N-1 | SciNet + AIC | 中 |

### 7.2 不能发现的

| 内容 | 原因 |
|------|------|
| 复数结构 / 波函数 ψ | RGDE 发现几何（球），不发现代数（ℂ²） |
| 算符代数 / 对易关系 | 需要操作层面的数学，不只是几何 |
| 纠缠 | 需要复合系统实验（当前仅单系统） |
| 不确定性原理 | 需要理解不相容测量的数学关系 |
| 测量问题 / 坍缩 | 哲学问题，非可操作物理 |

### 7.3 最大风险

**SciNet → SR 桥的可靠性。** SciNet 可能学到数值上正确但符号上不可提取的表征。如果 SR 无法从 encoder 中提取简洁的符号公式，RGDE 全链条断裂。

**缓解：**
- SciNet 训练时添加 encoder 稀疏性正则化
- 多种子训练，选择 encoder 最简洁的
- GPT tomography (Mazurek 2021) 已在光子实验中验证 K=3 的可恢复性
- **P0 验证必须在任何其他工作之前完成**

---

## 8. Implementation Plan

### 8.1 Phase 0: SciNet→SR Bridge Validation (1 month)

**目标：** 在已知系统上验证 RGDE 管道的核心可行性。

```
实验:
  1. 生成 Stern-Gerlach 模拟数据 (Bloch 球)
  2. 训练 SciNet, 用 AIC 确认 K=3
  3. 对 encoder 做 SR, 看能否提取 Bloch 参数化
  4. 搜索约束 z₁²+z₂²+z₃² ≤ 1
  5. 对 decoder 做 SR, 看能否恢复测量公式

Gate: 如果 step 3-4 成功率 > 50% (多种子) → 继续
      如果 < 30% → 需要修改方法或降级为 Phase 1+2 only
```

### 8.2 Phase 1: Formula Discovery (3 months)

**目标：** 在量子实验数据上发现 E=hf 级别的公式。

```
  ├── 实现 12 个实验环境 (7 量子 + 3 对照 + 2 干扰)
  ├── 匿名化包装层
  ├── PySR 集成 + AI Feynman 分解
  ├── DreamCoder 式概念提取
  └── 替代物理测试 (h→2h)

交付: 各实验的 Pareto 最优公式集 + 概念库
```

### 8.3 Phase 2: Constant Unification (3 months, overlap with Phase 1)

**目标：** 跨实验发现通用常数 h。

```
  ├── PSLQ 常数关系发现
  ├── AI-Newton 式合情推理
  ├── 跨实验概念关联检测
  └── 多种子一致性验证

交付: 基本常数集 + 统一公式 + 论文 (Phase 1+2 combined)
```

### 8.4 Phase 3: RGDE Full Pipeline (4-6 months, after P0 gate)

**目标：** 运行完整的 ATLAS 循环，验证 DSL 成长和几何发现。

```
  ├── 诊断模块 (D1-D5)
  ├── RGDE 完整实现 (Step 4a-4f)
  ├── 统一模块 (层次 1-3)
  ├── 消融实验 (PySR alone / SciNet alone / full ATLAS)
  ├── 基线对比 (AI Feynman 2.0, enriched PySR, adapted DreamCoder)
  ├── 非量子验证 (经典陀螺等球面状态空间系统)
  ├── 预测性验证 (发现的几何能否预测新实验)
  └── 论文

交付: ATLAS 完整系统 + 论文 (targeting Nature Machine Intelligence)
```

### 8.5 Timeline

```
Month 1:     Phase 0 (P0 验证)
Month 2-4:   Phase 1 (公式发现)
Month 3-6:   Phase 2 (常数统一, 与 Phase 1 重叠)
Month 5:     Phase 1+2 论文投稿
Month 6-11:  Phase 3 (RGDE, 条件: P0 通过)
Month 12:    Phase 3 论文投稿
```

### 8.6 Tech Stack

| 组件 | 工具 | 理由 |
|------|------|------|
| 符号回归 | PySR (Julia 后端) | 最快的开源 SR, 3.5k stars |
| 量纲约束 SR | PhySO (PyTorch) | 单位感知搜索, 2k stars |
| 信息瓶颈 | SciNet (PyTorch port) | 已验证能发现 Bloch 球 |
| 常数关系 | mpmath.pslq | 标准 PSLQ 实现 |
| 实验模拟 | NumPy/SciPy | 完全控制输出接口 |
| 并行化 | Ray | 多种子并行 + RGDE 评估 |
| 概念提取 | 自研 (参考 Stitch/DreamCoder) | 子表达式频率统计 + MDL |

### 8.7 Compute

```
Phase 0: 1× GPU (A100 40GB), ~1 周
Phase 1: 4× GPU, ~3 周 (含多种子)
Phase 2: CPU-dominant (PSLQ), ~1 周
Phase 3: 8× GPU, ~4 周 (RGDE 循环含多次 SciNet 训练 + SR)
Total: ~300-500 GPU-hours
```

---

## 9. Contribution Summary

### 9.1 对领域的贡献

1. **RGDE 管道**（首创）：从学习表征中自动提取符号结构和几何约束，构建新的 DSL 类型。串联 SciNet (表征学习) → PySR (符号提取) → DSL 扩展。
2. **失败驱动的框架发现**（首创）：将 SR 的系统性失败作为框架不足的信号，自动诊断并扩展。现有系统只从成功中学习。
3. **跨实验压缩作为发现的度量**（理论贡献）：将框架发现形式化为跨多实验数据的最短统一描述搜索。
4. **反作弊验证协议**（方法论贡献）：替代物理测试 + 多种子一致性 + 非量子对照，为 AI 物理发现领域建立验证标准。

### 9.2 科学意义

如果成功，ATLAS 将产出以下发现：

- **Phase 1+2 (确定可行):** 首次从匿名化量子实验数据中自动发现普朗克常数 h 及相关公式。与 AI-Newton 发现 G 和 F=ma 对等，但在量子领域。
- **Phase 3 (需验证):** 首次自动发现量子态空间的几何结构（Bloch 球），并将其编码为系统可用的新数学类型。这展示了一种全新的 AI 能力——不只发现公式，而是发现描述公式的数学语言。

### 9.3 Framing for Publication

> "We present ATLAS, the first system that can automatically detect when its mathematical framework is insufficient to describe experimental data, discover the geometric structure of the underlying state space, and extend its own symbolic language accordingly. We validate this on quantum physics experiments, where ATLAS — starting from classical real-valued arithmetic — autonomously discovers probabilistic behavior, the Planck constant, and the Bloch sphere geometry of qubit state spaces."

---

*ATLAS Proposal v1.0*
*For Discussion — 2026-03-31*
