# Brainstorm v2: ATLAS — Autonomous Theory Learning through Adaptive Search

---

## 0. 命名

**ATLAS** = **A**utonomous **T**heory **L**earning through **A**daptive **S**earch

隐喻：Atlas 是支撑天穹的巨人——系统从数据中支撑起理论框架。

---

## 1. 核心洞察：三层发现对应三种不同的搜索

文献综述揭示了一个清晰的层次结构：

```
层次 3: 框架发现  "物理量是什么类型的数学对象？"
        ↓ 确定了搜索空间
层次 2: 定律发现  "这些对象满足什么方程？"
        ↓ 确定了常数
层次 1: 参数发现  "方程中的常数是多少？"
```

每一层的搜索性质完全不同：

| | 层次 1: 参数 | 层次 2: 定律 | 层次 3: 框架 |
|--|-----------|----------|----------|
| 搜索空间 | 连续 (ℝⁿ) | 离散+连续 (表达式树) | 离散 (极小) |
| 搜索方法 | 梯度下降/BFGS | GP/SR/RL | 枚举/MCTS |
| 空间大小 | 低维连续 | 极大 (~无穷) | 极小 (~百级) |
| 每步评估成本 | 低 (函数求值) | 中 (拟合+MDL) | 高 (需要重跑层次2) |
| 现有工具 | scipy.optimize | PySR, AI Feynman | **无 (需要发明)** |

**ATLAS 的设计原则：为每一层使用最合适的搜索方法，而非用同一种方法处理所有层次。**

---

## 2. 层次 3 的搜索空间：操作性理论 (Operational Theories)

### 2.1 放弃 GPT，使用更基础的"操作性理论"形式化

GPT 直接用会构成作弊（它是从量子力学反思中产生的）。但 GPT 背后的**操作性思想**是通用的：

```
操作性理论 = 回答以下问题的规则集合:
  Q1: 系统有多少个可区分状态？ → N
  Q2: 描述系统需要多少个实数参数？ → K
  Q3: 相同输入重复实验，输出是否每次相同？ → 确定性/概率性
  Q4: 两个系统组合后，描述需要多少参数？ → K_composite(K_A, K_B)
  Q5: 什么变换保持理论结构不变？ → 变换群 G

这些问题不涉及任何特定物理理论
它们是任何想理解实验数据的智能体都需要回答的基本问题
```

### 2.2 可从数据中直接测量的量

**关键发现：上述 Q1-Q5 中的大部分可以从实验数据中直接测量，不需要先验物理知识。**

```
Q1 (可区分状态数 N):
  方法: 对环境的 knob 做离散扫描
        观测 detector 输出的"类别数"
        例如: Stern-Gerlach 实验中, 粒子落在2个位置 → N=2

Q2 (状态参数数 K):
  方法: SciNet 式信息瓶颈!
        训练编码器-解码器, 逐步缩小瓶颈维度
        最小使预测仍准确的维度 = K
        这是 SciNet 已经验证可行的方法

Q3 (确定性 vs 概率性):
  方法: 相同 knob 设置重复实验多次
        检测 detector 输出的方差
        方差 = 0 → 确定性; 方差 > 0 → 概率性

Q4 (组合规则):
  方法: 比较单系统实验和双系统联合实验
        K_single vs K_joint 的关系
        经典: K_joint = K_A + K_B (独立)
        量子: K_joint = K_A * K_B + more (纠缠态需要更多参数)

Q5 (变换群):
  方法: Lie对称性检测 (文献中已有方法)
        或: 测量状态空间的维度和连通性
```

### 2.3 这些测量量如何决定理论

Hardy (2001) 的结果告诉我们：

```
如果满足几个"合理"假设 (概率性、简单性、子空间、组合、连续性):

  K = N^r - 1 (某个正整数 r)

  r = 1 → K = N-1   → 经典概率论
  r = 2 → K = N²-1  → 复数量子力学

  没有其他选项!

所以: 系统只需要从数据中测量 K 和 N,
      计算 r = log(K+1) / log(N),
      如果 r ≈ 1 → 经典
      如果 r ≈ 2 → 量子
```

**这个结果的深刻含义：系统不需要"发明"量子力学。它只需要测量一个数字 (r)，这个数字告诉它"数据来自哪种理论"。**

---

## 3. ATLAS 完整架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        ATLAS Architecture                        │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Level 3: Theory Identifier (理论识别器)                      │ │
│  │                                                              │ │
│  │  从数据中测量操作性参数:                                      │ │
│  │    N (可区分状态数) ← 离散扫描                                │ │
│  │    K (状态维度)     ← SciNet信息瓶颈                         │ │
│  │    r = log(K+1)/log(N)                                       │ │
│  │    deterministic?   ← 重复实验方差检测                        │ │
│  │    K_joint          ← 联合实验瓶颈                            │ │
│  │                                                              │ │
│  │  输出: 操作性理论参数 → 决定 Level 2 的搜索空间              │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                              │                                    │
│  ┌──────────────────────────▼──────────────────────────────────┐ │
│  │ Level 2: Law Discoverer (定律发现器)                         │ │
│  │                                                              │ │
│  │  在 Level 3 确定的理论框架内做符号回归:                      │ │
│  │                                                              │ │
│  │  if r ≈ 1 (经典):                                           │ │
│  │    标准 PySR — 搜索 y = f(x), f: ℝ → ℝ                     │ │
│  │                                                              │ │
│  │  if r ≈ 2 (量子) AND deterministic = False:                  │ │
│  │    概率 SR — 搜索 P(y|x) = f(x),                            │ │
│  │    其中 f 被约束为合法的概率分布                               │ │
│  │    (来自Level 3确定的状态空间几何)                             │ │
│  │                                                              │ │
│  │  AI-Newton 式概念提取 + 合情推理泛化                          │ │
│  │  输出: 各实验的定律 (含待定常数)                              │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                              │                                    │
│  ┌──────────────────────────▼──────────────────────────────────┐ │
│  │ Level 1: Constant Finder (常数发现器)                        │ │
│  │                                                              │ │
│  │  BFGS 常数优化 (PySR内置)                                    │ │
│  │  PSLQ 跨实验常数关系                                         │ │
│  │  输出: 基本常数集 + 统一的定律集                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Cross-cutting: Environment Layer + Anti-cheating             │ │
│  │  清洁接口 / 替代物理测试 / 多种子一致性                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Level 3 详细设计: Theory Identifier

### 4.1 测量 N (可区分状态数)

```python
class DistinguishabilityMeasurer:
    """
    N = 系统在一次测量中能给出多少种完全可区分的结果

    方法: 对每个实验, 扫描 knob 设置,
          统计 detector 输出的"聚类数"

    例如:
      自由落体实验: detector 输出是连续值 → N 概念上 = ∞ (但有限精度下可量化)
      Stern-Gerlach: detector 显示2个离散峰 → N = 2
      氢光谱: detector 显示离散线 → N = 线的数量

    具体算法:
      1. 在多种 knob 设置下收集大量 detector 数据
      2. 对 detector 输出做聚类 (DBSCAN / GMM)
      3. 最优聚类数 = N
      4. 对于连续输出, N 定义为在给定精度下的有效类别数
    """

    def measure_N(self, env, n_samples=10000):
        # 收集数据
        outputs = []
        for _ in range(n_samples):
            knobs = env.sample_random_knobs()
            result = env.run(knobs)
            outputs.append(result["detector_0"])

        # 聚类
        if is_discrete(outputs):
            return count_distinct_values(outputs)
        else:
            # 对连续输出, 用有效维度
            optimal_k = find_optimal_clusters(outputs, method="BIC")
            return optimal_k
```

### 4.2 测量 K (状态空间维度) — SciNet 方法

```python
class StateDimensionMeasurer:
    """
    K = 完全描述系统状态需要多少个独立实参数

    方法: SciNet 信息瓶颈
      训练 encoder-decoder, 瓶颈维度从 1 开始递增
      最小使预测准确的瓶颈维度 = K

    为什么这有效 (来自 Iten et al. 2020):
      SciNet 已经证明可以从数据中发现:
      - 日心说的2个自由度 (K=2)
      - 碰撞中的守恒量维度
      - 量子态的 Bloch 球参数 (K=3 for qubit)
    """

    def measure_K(self, env, max_K=20):
        results = {}

        for k in range(1, max_K + 1):
            # 训练瓶颈维度为 k 的 SciNet
            model = SciNet(
                encoder_layers=[128, 64],
                bottleneck_dim=k,
                decoder_layers=[64, 128],
                activation="tanh"
            )

            # 训练
            data = collect_prediction_data(env)
            model.train(data, epochs=200)

            # 评估预测精度
            val_loss = model.evaluate(data.validation)
            results[k] = val_loss

        # 找到 "elbow point" — K 增加但 loss 不再显著下降
        K = find_elbow(results)
        return K
```

### 4.3 测量确定性/概率性

```python
class StochasticityDetector:
    """
    检测: 相同输入 → 输出是否每次相同?

    方法: 对同一组 knob 设置多次重复实验
    """

    def detect(self, env, n_repeats=100, n_settings=50):
        for _ in range(n_settings):
            knobs = env.sample_random_knobs()
            outputs = [env.run(knobs)["detector_0"] for _ in range(n_repeats)]

            variance = np.var(outputs)
            if variance > self.threshold:
                return StochasticResult(
                    is_stochastic=True,
                    variance_by_setting=self._variance_map(env, n_repeats)
                )

        return StochasticResult(is_stochastic=False)
```

### 4.4 计算 r — 理论类型指标

```python
class TheoryTypeCalculator:
    """
    Hardy 的结果: K + 1 = N^r

    r = 1: 经典
    r = 2: 量子 (复数)
    其他: 异常理论

    只需要 N 和 K, 两者都可以从数据中测量
    """

    def calculate_r(self, N, K):
        if N <= 1:
            return None  # 不适用

        r = np.log(K + 1) / np.log(N)

        theory_type = "unknown"
        if abs(r - 1) < 0.1:
            theory_type = "classical"
        elif abs(r - 2) < 0.1:
            theory_type = "quantum_complex"
        elif abs(r - 1.5) < 0.1:
            theory_type = "quantum_real"  # K = N(N+1)/2 - 1

        return TheoryIdentification(r=r, type=theory_type, N=N, K=K)
```

### 4.5 r 如何影响 Level 2 的搜索

```
if r ≈ 1 (经典):
  Level 2 搜索空间 = 标准实数域符号回归
  运算符: {+, -, ×, ÷, sin, cos, exp, log, sqrt}
  拟合目标: y = f(x)  (确定性函数)

if r ≈ 2 (量子) AND stochastic:
  Level 2 搜索空间需要扩展:
  (a) 拟合目标变为概率分布: P(y|x) = f(x)
  (b) 状态空间维度是 K = N²-1, 不是 N-1
      → 可能需要更多隐变量来描述系统
  (c) 具体搜索策略:
      - 先对 P(y|x) 做分布拟合 (KDE → 参数估计)
      - 对分布参数做 SR
      - 检查: 分布参数是否可以写成某个 K 维向量的函数?
```

---

## 5. Level 2 详细设计: Law Discoverer

### 5.1 经典分支 (r ≈ 1)

直接复用已验证的方法：

```
PySR + AI Feynman 递归分解 + AI-Newton 概念提取
(与 Phase 1 plan 相同, 不再赘述)
```

### 5.2 概率分支 (r ≈ 2, stochastic)

这是新内容。当 Level 3 判定系统是概率性的：

```python
class ProbabilisticLawDiscoverer:
    """
    当系统输出是概率性的, 拟合目标不再是 y = f(x),
    而是 P(y|x) 的某种参数化描述

    方法: 两阶段
      Stage 1: 对每组 knob 设置, 估计输出的经验分布
      Stage 2: 对分布参数做 SR
    """

    def discover(self, env, data):
        # Stage 1: 分布估计
        distributions = {}
        for knob_setting in data.unique_settings():
            outputs = data.get_outputs(knob_setting)
            # 非参数估计
            dist = KernelDensityEstimate(outputs)
            # 参数化描述 (位置、宽度、偏度、峰度、峰数...)
            params = extract_distribution_features(dist)
            distributions[knob_setting] = params

        # Stage 2: 对分布参数做 SR
        # 例如: 双缝实验中, 分布的"峰位置"可能满足 SR 可发现的规律
        for param_name, values in zip_params(distributions):
            law = pysr.fit(
                X=np.array(list(distributions.keys())),
                y=np.array(values),
            )
            if law.best_r2 > 0.99:
                self.laws.append((param_name, law))

        # 关键检查: 概率分布本身是否等于某个确定性函数?
        # 即: P(y|x) ∝ f(x, y)?
        # 如果是: 这就是 Born 规则的雏形!
        # 具体: 对 (x, y, P) 三元组做 SR, 搜索 P = g(x, y)
        # 然后检查 g 是否可以分解为 |h(x,y)|²
        # (但不预设这种分解 — 让 SR 自己发现)
```

### 5.3 关键时刻：概率分布 vs 波动公式的关联

```
假设系统到达了这个状态:

Level 3 发现:
  ENV_04 (双缝干涉, 高强度): r ≈ 1 (行为经典)
    → 确定性公式: I(x) = C * cos²(C' * x)

  ENV_04 (双缝干涉, 低强度): r ≈ 2 (行为概率性)
    → 概率分布: P(x) ∝ ??? (待 Level 2 发现)

Level 2 在概率分支中发现:
  P(x) = C'' * cos²(C''' * x)

Level 1 的常数统一发现:
  C ≈ C'' 且 C' ≈ C'''

→ 系统自主得出结论:
  "高强度下的确定性强度分布 = 低强度下的概率分布"
  即: P(x) ∝ I(x)

这不需要任何物理先验!
这是从数据中测量到的事实!

它等价于: "概率 ∝ 经典波动强度"
即 Born 规则 P = |ψ|² 的可观测推论!
```

**这是整个框架最令人兴奋的部分。系统不需要发现 Born 规则的数学形式（P = |ψ|²），它只需要发现一个经验事实：P(x) ∝ I(x)。**

---

## 6. 反作弊审查

### 6.1 Level 3 的反作弊

```
Q: SciNet 信息瓶颈是否构成作弊?
A: 不. SciNet 的唯一假设是"物理可以被低维表示压缩"
   这是关于宇宙的哲学命题, 不是特定物理理论
   已在经典系统(日心说)和量子系统(Bloch球)中验证通用性

Q: 计算 r = log(K+1)/log(N) 是否构成作弊?
A: ⚠️ 灰色地带.
   r 的计算公式来自 Hardy (2001), 他是在反思量子力学后得出的.
   但: r 的值是从数据中测量到的, 不是假设的.
   如果数据给出 r=1 → 系统判定"经典"
   如果数据给出 r=1.5 → 系统判定"未知理论"
   如果数据给出 r=2 → 系统判定"需要扩展描述"

   更安全的替代: 不计算 r, 直接用 K 和 N 的值
   如果 K > N-1 → "经典描述不够, 需要更多参数"
   这不涉及任何理论推断, 只是数据事实

Q: 概率分支的 SR 是否预设了 Born 规则?
A: 不. 系统搜索的是 P(y|x) 的任意函数形式
   不预设 P = |f|² 或任何特定映射
   如果 SR 发现 P(x) = C * cos²(x/λ), 且这恰好等于
   确定性实验中的 I(x) = C' * cos²(x/λ)
   → 这是从数据中发现的关联, 不是预设的

Q: 替代物理测试能验证吗?
A: 是.
   h → 2h: 干涉条纹间距变化 → K 和 N 的测量值变化相应
   r 应保持不变 (理论类型不随常数值变化)
   但具体定律和常数值应变化
```

### 6.2 Anti-Cheating Checklist 更新

```
✅ Level 3 只使用从数据中可测量的量 (N, K, 方差)
✅ SciNet 是已验证的通用方法 (非量子特有)
✅ 概率分布拟合不预设特定映射形式
✅ P(x) ∝ I(x) 是从数据关联中发现的, 不是预设的
⚠️ r = log(K+1)/log(N) 的计算可以简化为 "K > N-1?" 的判断
❌ 需要确保实验中"低强度"和"高强度"不是人为分组
    → 系统应通过连续扫描 knob_3 (源强度) 自己发现这个转变
```

---

## 7. 与其他方法的关系图

```
        SciNet (概念发现, 信息瓶颈)
            ↓ 提供: K 的测量
            ↓
    AI-Newton (概念驱动定律发现)
            ↓ 提供: 单实验公式 + 概念提取
            ↓
        PySR (符号回归引擎)
            ↓ 提供: 符号公式
            ↓
      AI Feynman (递归分解)
            ↓ 提供: 高维问题的分解策略
            ↓
    MASS (多种子一致性)
            ↓ 提供: 发现的可靠性验证
            ↓
        PSLQ (常数关系)
            ↓ 提供: 通用常数
            ↓
   Hardy/GPT (操作性理论)
            ↓ 提供: 理论类型的判定标准
            ↓
    ═══════════════════
     ATLAS (本框架)
    ═══════════════════
```

**ATLAS 不是从零发明——它是对所有已验证方法的系统性整合，加上一个新的 Level 3 (Theory Identifier)。**

---

## 8. 预期的发现路径

```
Step 1: Level 3 对所有实验做基本测量
  ENV_01 (光电效应): N不适用(连续输出), deterministic=True
  ENV_02 (康普顿散射): 同上
  ENV_04 (双缝, 高强度): N不适用, deterministic=True
  ENV_04 (双缝, 低强度): N不适用, deterministic=False!
  ENV_07 (Stern-Gerlach): N=2!, deterministic=False!

  → 发现: 某些实验是概率性的, 某些不是

Step 2: SciNet 测量 K
  ENV_07 (Stern-Gerlach, N=2): K=3 (Bloch球)
  → r = log(4)/log(2) = 2!
  → 判定: 这个系统需要 N²-1 参数, 不是 N-1

Step 3: Level 2 分支
  确定性实验: 标准 SR → 发现 E=hf 等
  概率性实验: 概率 SR → 发现 P(x) 的规律

Step 4: 跨实验关联
  发现: ENV_04 低强度的 P(x) ∝ ENV_04 高强度的 I(x)
  → "概率分布 = 波动强度" — Born 规则的经验等价物

Step 5: 常数统一
  PSLQ 发现 h 跨多个实验

Step 6: 综合
  "某些实验是概率性的 + K=N²-1 + P∝I + 常数h"
  = 波粒二象性的操作性描述
```

---

## 9. 开放问题

```
1. SciNet 能否在量子实验数据上准确测量 K?
   → 需要实验验证 (Phase 1 的前置工作)

2. K > N-1 的判断是否足以触发概率分支?
   → 可能需要其他信号 (如方差检测) 作为辅助

3. 概率 SR 能否发现 P(x) 的解析形式?
   → 取决于 KDE 的质量和数据量

4. 跨实验 P∝I 的发现是否需要对象标签?
   → 可能需要知道两个实验研究的是同一 entity

5. 对于没有离散 N 的实验 (如光电效应),
   Level 3 如何工作?
   → 可能需要不同的策略 (直接 SR, 跳过 Level 3)

6. 如果真实数据中 r 不是整数怎么办?
   → 噪声、有限数据、近似效应都会导致 r 不精确
   → 需要鲁棒的判定阈值
```

---

## 10. 下一步

```
1. 等待剩余调研结果, 整合 DreamCoder + MCTS + GitHub 信息
2. 细化 Level 3 的算法, 特别是 K 的测量方法
3. 设计最小可行实验: 只用 Stern-Gerlach + 双缝干涉
   → 验证 SciNet 能否测量 K=3 (qubit)
   → 验证概率 SR 能否发现 P(x) ∝ cos²
4. 启动 Critic Agent 评估框架
```
