# Brainstorm v4: 整合 DreamCoder 式库学习

---

## 从 DreamCoder 借鉴的核心思想

DreamCoder (Ellis et al., NeurIPS 2021) 是一个"学会编程"的系统：

```
DreamCoder 循环:
  1. 从一个最小 DSL 开始 (基本操作: +, *, if, fold, ...)
  2. 用程序合成解决一批任务
  3. 从成功的程序中提取共同子程序 → "库函数"
  4. 将库函数添加到 DSL 中
  5. 新 DSL 使未来任务更容易解决
  6. 回到 step 2

关键: DSL 在学习过程中成长
     系统发明了新的"概念"(库函数)
     这些概念不是预设的，而是从数据中发现的
```

**DreamCoder 和我们问题的精确类比：**

| DreamCoder | ATLAS |
|------------|-------|
| 任务 (task) | 实验 (experiment) |
| 程序 (program) | 物理定律 (law) |
| DSL | 物理框架 (framework) |
| 库函数 (library function) | 物理概念 (physical concept) |
| DSL 成长 | 框架发现 |

**但有一个关键区别：**

```
DreamCoder: 从成功的程序中抽象库函数 (学习什么有用)
ATLAS:      需要从失败中发现 DSL 的不足 (学习什么缺失)

两者互补:
  从成功中学: "cos²(x) 在多个公式中出现 → 提取为概念"
  从失败中学: "确定性公式无法拟合 → DSL 需要概率构造"
```

---

## ATLAS v4: 双向库学习

### 核心架构

```
┌──────────────────────────────────────────────────────┐
│                ATLAS v4 Main Loop                     │
│                                                       │
│   DSL_0 = {ℝ, +, -, ×, ÷, sin, cos, exp, log}      │
│                                                       │
│   for epoch in range(MAX_EPOCHS):                    │
│                                                       │
│     ┌─────────────────────────────────────┐          │
│     │ Step A: Solve (在当前 DSL 下做 SR)   │          │
│     │   对每个实验, 用 PySR 搜索定律        │          │
│     │   记录: 成功/失败 + 发现的公式        │          │
│     └─────────────┬───────────────────────┘          │
│                   │                                   │
│     ┌─────────────▼───────────────────────┐          │
│     │ Step B: Abstract (从成功中抽象概念)   │          │
│     │   DreamCoder 式: 提取公共子表达式     │          │
│     │   AI-Newton 式: 从定律中提取概念      │          │
│     │   → 添加新库函数到 DSL                │          │
│     └─────────────┬───────────────────────┘          │
│                   │                                   │
│     ┌─────────────▼───────────────────────┐          │
│     │ Step C: Diagnose (从失败中发现缺陷)   │          │
│     │   分析 SR 为什么失败:                  │          │
│     │   - 随机性? → DSL 需要概率构造        │          │
│     │   - 高维? → DSL 需要新类型            │          │
│     │   - 离散? → DSL 需要整数/离散构造     │          │
│     │   → 提议 DSL 扩展                     │          │
│     └─────────────┬───────────────────────┘          │
│                   │                                   │
│     ┌─────────────▼───────────────────────┐          │
│     │ Step D: Extend (扩展 DSL)            │          │
│     │   评估每个提议的扩展:                  │          │
│     │   - 在扩展后的 DSL 下重新 SR          │          │
│     │   - 拟合改善 vs 复杂度增加            │          │
│     │   - Pareto 最优的扩展被接受           │          │
│     └─────────────┬───────────────────────┘          │
│                   │                                   │
│     if all experiments solved:                       │
│       break                                          │
│     DSL = DSL + accepted_extensions + new_concepts   │
│                                                       │
│   return DSL_final, laws, concepts, constants        │
└──────────────────────────────────────────────────────┘
```

### Step A: Solve — 在当前 DSL 下做 SR

```
不需要任何新内容 — 直接用 PySR
唯一变化: PySR 的运算符集合 = 当前 DSL 的操作集
随着 DSL 成长, PySR 可用的运算符越来越丰富

epoch 0: operators = {+, -, ×, ÷, sin, cos, exp, log}
epoch 1: operators = {+, -, ×, ÷, sin, cos, exp, log, concept_1}
         其中 concept_1 可能是 "cos²" (从 Step B 提取)
epoch 2: operators = {..., concept_1, Distribution}
         其中 Distribution 是从 Step C 诊断出需要的概率构造
```

### Step B: Abstract — DreamCoder 式概念提取

```python
class ConceptAbstractor:
    """
    输入: 当前 epoch 所有成功发现的公式
    输出: 新的库函数 (概念)

    方法: 反碎片化 (anti-unification) + 压缩

    DreamCoder 的原始方法:
      1. 对所有成功程序, 找最长公共子程序
      2. 如果某个子表达式在 ≥3 个公式中出现 → 提取为库函数
      3. 用库函数重写所有公式 → 公式变短
      4. 如果总描述长度减少 → 接受库函数

    适配到物理:
      1. 对所有实验的成功公式做子表达式统计
      2. 高频子表达式 → 可能是物理概念
      3. 用 MDL 评估: 添加概念后总描述长度是否减少
    """

    def extract(self, formulas):
        # 统计所有子表达式的出现频率
        subtree_counts = count_all_subtrees(formulas)

        # 按 (频率 × 大小) 排序 — 大的、频繁的子表达式最有价值
        candidates = sorted(
            subtree_counts.items(),
            key=lambda x: x[1] * tree_size(x[0]),
            reverse=True
        )

        # 用 MDL 评估每个候选
        new_concepts = []
        for subtree, count in candidates:
            # 计算: 如果把 subtree 定义为新概念, 总描述长度变化多少
            savings = count * tree_size(subtree)  # 所有出现处的节省
            cost = tree_size(subtree) + 1  # 定义概念本身的成本
            if savings > cost:
                new_concepts.append(Concept(
                    name=f"concept_{len(self.all_concepts)}",
                    expression=subtree,
                    frequency=count
                ))

        return new_concepts
```

### Step C: Diagnose — 从失败中诊断 DSL 缺陷

```python
class DSLDiagnoser:
    """
    输入: SR 失败的实验列表 + 失败的 SR 结果
    输出: DSL 扩展提议列表

    每种失败模式对应一种 DSL 扩展:
    """

    # DSL 扩展操作的完整列表 (通用, 不特指量子力学)
    EXTENSIONS = {
        "probabilistic": {
            "trigger": "相同输入多次实验输出不同 (方差 > 0)",
            "extension": "添加概率分布构造到 DSL",
            "具体": "允许 SR 搜索 P(y|x) 而非 y=f(x)",
            "通用性": "概率论是通用数学, 不特指任何物理理论"
        },
        "discrete": {
            "trigger": "输出只取有限个离散值",
            "extension": "添加整数/离散构造到 DSL",
            "具体": "添加 floor, mod, indicator 运算符",
            "通用性": "离散数学是通用的"
        },
        "higher_dim": {
            "trigger": "SciNet 测量的瓶颈维度 K 大于预期",
            "extension": "添加向量/矩阵类型到 DSL",
            "具体": "允许 SR 搜索向量值函数",
            "通用性": "线性代数是通用数学"
        },
        "periodic": {
            "trigger": "SR 残差有周期性模式",
            "extension": "增加周期函数的搜索优先级",
            "具体": "调整 PySR 的搜索参数",
            "通用性": "Fourier 分析是通用的"
        },
        "threshold": {
            "trigger": "SR 在某个参数值附近突然失败",
            "extension": "添加分段函数构造",
            "具体": "添加 max(0,x), heaviside, abs 运算符",
            "通用性": "分段函数是通用数学"
        }
    }

    def diagnose(self, failed_experiments):
        proposals = []

        for env, sr_result in failed_experiments:
            # 检测每种失败模式
            if self.is_stochastic(env):
                proposals.append("probabilistic")

            if self.has_discrete_output(env):
                proposals.append("discrete")

            K = self.measure_K(env)
            N = self.measure_N(env)
            if K > N - 1 + tolerance:
                proposals.append("higher_dim")

            if self.residual_is_periodic(sr_result):
                proposals.append("periodic")

            if self.has_sharp_transition(env, sr_result):
                proposals.append("threshold")

        return list(set(proposals))  # 去重
```

### Step D: Extend — 评估并接受 DSL 扩展

```python
class DSLExtender:
    """
    输入: DSL 扩展提议列表
    输出: 被接受的扩展集

    评估标准: 扩展后 SR 的拟合改善 vs DSL 复杂度增加
    使用 Pareto 前沿选择
    """

    def evaluate_and_extend(self, proposals, dsl, experiments):
        results = []

        for proposal in proposals:
            new_dsl = dsl.apply_extension(proposal)

            # 在扩展后的 DSL 下重新 SR (只对之前失败的实验)
            new_sr = rerun_sr(new_dsl, experiments.failed)

            improvement = sum(
                new_sr[e].best_r2 - experiments.sr_results[e].best_r2
                for e in experiments.failed
            )
            cost = dsl_complexity(new_dsl) - dsl_complexity(dsl)

            results.append((proposal, improvement, cost))

        # Pareto 选择
        accepted = pareto_filter(results)
        return accepted
```

---

## 为什么 v4 比 v3 更好

```
v3 的问题:
  Level 3 (Theory Identifier) 是一次性的 — 先分类, 再做 SR
  如果分类错误, 整个流程走偏
  没有迭代/自纠正机制

v4 的改进:
  没有一次性分类
  DSL 从最小开始, 每个 epoch 只扩展一点
  每次扩展都有 Pareto 评估
  可以回退 (如果扩展没用, 下个 epoch 不会再提议)
  多次迭代保证最终 DSL 足够但不过度

类比:
  v3 像医生先做诊断再开药 (一次性)
  v4 像免疫系统: 持续监测, 增量适应, 无需总诊断
```

---

## DSL 成长的预期路径

```
Epoch 0: DSL = {ℝ, +, -, ×, ÷, sin, cos, exp, log}
  Solve: 光电效应→成功(E=hf-W), 康普顿→成功, 双缝高强度→成功
         双缝低强度→失败!, Stern-Gerlach→失败!
  Abstract: 从成功公式中提取 concept_1 = cos²(·)
  Diagnose: 双缝低强度 → stochastic, SG → discrete + stochastic

Epoch 1: DSL += {concept_1(cos²), probabilistic, discrete}
  Solve: 双缝低强度 → 概率SR → P(x) = C·cos²(Cx)  → 成功!
         SG → 概率SR → P(up)=cos²(θ/2), P(down)=sin²(θ/2) → 成功!
  Abstract: P(x)中也有cos² → concept_1 是跨确定性/概率性的通用概念!
            这个发现等价于 Born 规则的经验内容
  Diagnose: 无新失败

Epoch 2: DSL 不变 (所有实验都已成功)
  跨实验分析:
    常数统一 (PSLQ): 发现 h
    概念关联: 确定性cos²和概率cos²共享参数

输出:
  DSL_final = {ℝ, +, -, ×, ÷, sin, cos, exp, log, cos², prob_dist}
  Laws = {E=hf-W, Δλ=h/(mc)(1-cosθ), P(x)∝cos²(πdx/hL), ...}
  Concepts = {cos²(·), h}
  Key finding: "确定性强度分布 = 概率分布形状"
```

---

## 反作弊终审

```
DSL 扩展操作是否作弊?

  "probabilistic" 扩展:
    触发条件: 方差 > 0 (纯统计检测)
    扩展内容: 允许搜索 P(y|x) (概率论, 非物理特有)
    → ✅ 通用

  "discrete" 扩展:
    触发条件: 输出只有有限个值 (数据事实)
    扩展内容: 整数运算符 (离散数学)
    → ✅ 通用

  "higher_dim" 扩展:
    触发条件: K > N-1 (SciNet 测量)
    扩展内容: 向量/矩阵类型 (线性代数)
    → ✅ 通用

  所有扩展都是:
    1. 由数据中的可观测信号触发 (不是预设)
    2. 对应通用数学构造 (不是物理特有)
    3. 通过 Pareto 评估验证有用性 (不是盲目添加)

  关键区别 (vs v2 的"文法搜索"):
    v2: 预定义了一个包含所有可能修改的列表 (含 ℝ→ℂ, |f|² 等)
    v4: 修改由数据触发, 只添加被证明有用的扩展

  注意: "概率性"扩展不含 |ψ|²
    系统搜索的是 P(y|x) 的任意函数形式
    如果发现 P(x) = cos²(...)
    这是 SR 的结果, 不是预设的
```

---

## 与 DreamCoder 的关键差异

```
DreamCoder:
  - 任务是预定义的 (有明确的输入输出对)
  - 库学习只从成功中提取
  - DSL 只添加新函数, 不改变类型系统

ATLAS v4:
  - "任务"是实验数据 (需要自己定义什么是"成功")
  - 从成功和失败中都学习
  - DSL 不仅添加函数, 还改变类型系统 (加概率、加向量)
  - 每次扩展都有物理验证 (替代物理测试)

ATLAS v4 可以说是 "DreamCoder for Physics":
  DreamCoder 学会了用越来越复杂的程序解决编程任务
  ATLAS 学会了用越来越丰富的数学框架描述物理实验
```
