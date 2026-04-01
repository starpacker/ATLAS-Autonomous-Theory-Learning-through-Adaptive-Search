# 核心问题：如何表示和发现一个"框架"？

---

## 1. 首先搞清楚："框架"到底是什么？

一个物理理论框架不是一个公式，而是一套**规则系统**。对比：

```
公式:    F = ma
         这是一个在已有框架内的具体表达式

框架:    经典力学
         - 状态由实数描述 (位置 x ∈ ℝ, 速度 v ∈ ℝ)
         - 状态的演化由微分方程控制 (ẍ = f(x,t))
         - 测量不影响状态
         - 力的叠加是线性的
         - 所有物理量同时有确定值
```

```
框架:    量子力学
         - 状态由复数向量描述 (|ψ⟩ ∈ ℂⁿ)
         - 演化由线性算符控制 (iℏ∂ψ/∂t = Ĥψ)
         - 测量改变状态 (波函数坍缩)
         - 可观测量由厄米算符表示，可以不对易
         - 物理量不同时有确定值 (不确定性原理)
```

**关键观察：框架之间的区别不在于方程不同，而在于"基本对象的种类"和"组合规则"不同。**

用计算机科学的语言说：

```
公式 = 在一个编程语言里写的一行代码
框架 = 编程语言本身 (类型系统 + 语法 + 语义)
```

发现新公式 = 在已有语言里写新程序
发现新框架 = 发明新编程语言

---

## 2. 框架的形式化表示

一个物理框架可以被精确地分解为四个层次：

```
┌─────────────────────────────────────────────────┐
│ Layer 4: 解释规则 (Interpretation)               │
│   "如何从数学对象得到可观测预言"                   │
│   经典: x(t) 直接就是位置                         │
│   量子: P(x) = |⟨x|ψ⟩|² 才是可观测量             │
├─────────────────────────────────────────────────┤
│ Layer 3: 动力学规则 (Dynamics)                    │
│   "对象如何随时间变化"                            │
│   经典: ẍ = F/m (二阶ODE)                        │
│   量子: iℏ∂ψ/∂t = Ĥψ (一阶线性PDE on ℂ)        │
├─────────────────────────────────────────────────┤
│ Layer 2: 组合规则 (Algebra)                      │
│   "对象之间如何运算"                              │
│   经典: 实数乘法 (交换律 ab=ba)                   │
│   量子: 算符乘法 (不对易 ÂB̂≠B̂Â)                  │
├─────────────────────────────────────────────────┤
│ Layer 1: 基本类型 (Type System)                  │
│   "物理量是什么类型的数学对象"                     │
│   经典: 实数 ℝ, 实向量 ℝ³                        │
│   量子: 复向量 ℂⁿ, 厄米矩阵 Herm(n)             │
└─────────────────────────────────────────────────┘
```

**这四层中每一层都可以被形式化，且不需要自然语言或大模型。**

---

## 3. 不用 LLM 的框架表示方案

### 方案: 类型化文法 (Typed Grammar)

一个框架 = 一个文法。文法定义了"什么类型的表达式是合法的"。

```python
# 经典力学的文法 (简化)
classical_grammar = {
    "types": {
        "Scalar": "ℝ",           # 标量是实数
        "Vector": "ℝ³",          # 向量是三维实向量
        "State":  "(Vector, Vector)",  # 状态 = (位置, 速度)
    },
    "operations": {
        "+": ("Scalar", "Scalar") -> "Scalar",   # 实数加法
        "*": ("Scalar", "Scalar") -> "Scalar",   # 实数乘法
        "dot": ("Vector", "Vector") -> "Scalar", # 内积
        "d/dt": ("Scalar",) -> "Scalar",         # 时间导数
    },
    "axioms": [
        "commutative(+)",    # a+b = b+a
        "commutative(*)",    # a*b = b*a
        "associative(+)",
        "associative(*)",
    ],
    "dynamics": "d²x/dt² = f(x, dx/dt, t)",  # 二阶ODE
    "interpretation": "x(t) = observable_position",  # 直接可观测
}
```

```python
# 量子力学的文法 (简化)
quantum_grammar = {
    "types": {
        "Scalar":   "ℂ",                    # 标量是复数!
        "State":    "ℂⁿ",                   # 状态是复向量!
        "Operator": "Mat(n,n,ℂ)",            # 算符是复矩阵!
    },
    "operations": {
        "+": ("Scalar", "Scalar") -> "Scalar",
        "*": ("Scalar", "Scalar") -> "Scalar",
        "mat_mul": ("Operator", "Operator") -> "Operator",  # 矩阵乘法
        "apply": ("Operator", "State") -> "State",           # 算符作用
        "inner": ("State", "State") -> "Scalar",            # 内积
        "dagger": ("Operator",) -> "Operator",              # 共轭转置
        "norm_sq": ("State",) -> "ℝ",                       # |ψ|²
        "d/dt": ("State",) -> "State",
    },
    "axioms": [
        "commutative(+)",
        "NOT commutative(mat_mul)",   # 关键区别!
        "hermitian(Observable)",      # 可观测量是厄米的
        "unitary(Evolution)",         # 演化是幺正的
    ],
    "dynamics": "i * h_bar * d|ψ⟩/dt = H |ψ⟩",  # 一阶线性
    "interpretation": "P(x) = norm_sq(inner(|x⟩, |ψ⟩))",  # Born规则
}
```

**核心洞察：两个文法之间的差异是精确的、可枚举的。**

```
经典 → 量子 的文法变化:

Type change:   ℝ → ℂ (实数 → 复数)
Type addition: +Operator (新增算符类型)
Op change:     commutative(*) → NOT commutative(mat_mul)
Op addition:   +dagger, +norm_sq, +apply
Dynamics:      二阶ODE → 一阶线性PDE on ℂ
Interpretation: 直接值 → |amplitude|²
```

---

## 4. 框架发现 = 文法搜索

如果框架可以表示为文法，那么"发现新框架"就变成了一个**文法搜索问题**：

```
初始状态: 经典文法 (实数类型 + 基本运算 + 交换律)
目标状态: 某个能更好拟合所有数据的新文法
搜索空间: 所有可能的文法修改操作
```

### 文法修改的基本操作（有限且可枚举）

```
1. TYPE_EXTEND:  ℝ → ℂ (将实数类型扩展为复数)
2. TYPE_ADD:     添加新类型 (如: Matrix, Operator)
3. OP_ADD:       添加新运算 (如: conjugate, transpose)
4. AXIOM_REMOVE: 移除公理 (如: 移除交换律)
5. AXIOM_ADD:    添加公理 (如: 添加厄米性约束)
6. DYN_MODIFY:   修改动力学方程的阶数或结构
7. INTERP_MODIFY:修改解释规则 (如: 值→概率)
```

**每一步操作都是离散的、可枚举的。不需要自然语言来描述。**

### 搜索算法

```python
class FrameworkSearchAgent:
    """
    在文法空间中搜索最优框架

    不需要LLM。这是一个离散搜索问题。
    """

    def search(self, current_grammar, experiment_data, sr_results):
        """
        输入:
          current_grammar: 当前的类型化文法 (初始=经典力学文法)
          experiment_data: 所有实验数据
          sr_results: Phase 1/2 在当前文法下的SR结果

        过程:
          1. 识别当前文法的"失败点" — SR无法拟合的数据区域
          2. 对每个可能的文法修改操作:
             a. 应用修改, 得到新文法
             b. 在新文法下重新运行SR
             c. 评估: 拟合改善了多少? 文法复杂了多少?
          3. 选择 Pareto 最优的文法修改
          4. 递归: 在修改后的文法上继续搜索
        """

        failures = identify_sr_failures(sr_results)
        if not failures:
            return current_grammar  # 当前文法已够用

        candidates = []
        for modification in GRAMMAR_MODIFICATIONS:
            new_grammar = apply_modification(current_grammar, modification)

            # 在新文法下重新做SR
            new_sr_results = run_sr(new_grammar, experiment_data)

            # 评估
            fit_improvement = compare_fitness(new_sr_results, sr_results)
            complexity_cost = grammar_complexity(new_grammar) - grammar_complexity(current_grammar)

            candidates.append((new_grammar, fit_improvement, complexity_cost))

        # Pareto最优: 最大化拟合改善, 最小化复杂度增加
        pareto_front = compute_pareto(candidates)
        return pareto_front
```

### 具体的搜索实例

```
假设系统在 Phase 1 中发现:

  实验1(光电效应): meter = C₁ * max(knob_0 - C₂, 0) * knob_1
                   → R² = 0.999, SR成功 ✓

  实验4(双缝干涉, 高强度): detector = C₃ * cos²(C₄ * x)
                           → R² = 0.998, SR成功 ✓

  实验4(双缝干涉, 低强度): detector = 稀疏离散尖峰
                           → R² = 0.12, SR失败 ✗

系统检测到: 失败发生在低强度区域, 输出不是确定性函数

文法搜索启动:

  修改1: INTERP_MODIFY — 从"detector = f(knob)"到"P(detector) = f(knob)"
         即: 输出不是确定值, 而是概率分布
         → 在新文法下: P(x) 用KDE估计, 对P(x)做SR
         → 发现: P(x) = C₅ * cos²(C₆ * x)
         → R² = 0.997 ✓
         → 拟合改善: 0.12 → 0.997 (巨大!)
         → 复杂度增加: 1 (增加了概率解释)

  修改2: TYPE_EXTEND — ℝ → ℂ
         → 在新文法下: 搜索复数值函数 f(x) ∈ ℂ, |f|² = P(x)
         → 发现: f(x) = C₇ * (exp(iC₈x) + exp(iC₉x))
         → |f|² = cos²((C₈-C₉)x/2) 恰好 = Phase 1 的波动公式!
         → 拟合改善: 同上
         → 复杂度增加: 2 (扩展到复数 + 添加 norm_sq 运算)

  Pareto 分析:
    修改1: 改善0.997/复杂度+1 — Pareto最优
    修改2: 改善0.997/复杂度+2 — 被修改1支配(同样的改善但更复杂)

  → 系统首先选择修改1 (概率解释), 而非修改2 (复数)
  → 这恰好是历史上物理学家的实际认知路径!
     (先接受概率性, 后来才理解为什么需要复数)
```

---

## 5. 关于 LLM 的分析

### 5.1 一般大模型 (GPT-4, Claude, LLaMA-70B 等)

```
参数量: 7B - 400B+
训练数据: 包含大量物理教科书和论文
知识泄露风险: 极高 — 模型直接"知道"量子力学

用于框架发现: ❌ 不合理
理由:
  1. 模型见过量子力学 → "发现"量子框架 = 背诵
  2. 无法验证发现是来自数据还是来自训练记忆
  3. 替代物理测试也不够:
     模型可能对 h'=2h 的世界仍然输出量子力学框架
     (因为框架结构不依赖h的具体值)
```

### 5.2 小型从零训练的语言模型

```
参数量: 1M - 100M
训练数据: ???

问题: 训练数据是什么?
  选项A: 在数学文本上训练 → 泄露了数学框架知识
  选项B: 在代码上训练 → 泄露了编程语言结构知识
  选项C: 在实验数据上训练 → 语言模型对纯数值数据没什么用
  选项D: 在SR进化过程中生成的表达式上训练 → 这就是 NeSymReS 的做法

结论: 小型语言模型的训练数据问题无法回避
  如果训练数据含物理/数学知识 → 泄露
  如果训练数据不含 → 模型没有任何有用的先验, 不如不用
```

### 5.3 "语言模型"在这个问题中真正有用的地方

```
问题重新审视:

"框架发现"的搜索空间是什么?
  → 文法修改操作的序列
  → 这是一个离散的组合搜索空间
  → 空间大小 = O(|操作集|^深度)

这个搜索空间需要"语言能力"吗?
  → 不需要! 这是一个结构化搜索, 不是自然语言生成

什么工具适合这种搜索?
  → 遗传编程 (GP) — 在文法空间上进化
  → 强化学习 (RL) — 学习选择文法修改的策略
  → 蒙特卡洛树搜索 (MCTS) — 在文法修改树上搜索
  → 贝叶斯优化 — 在文法空间上做高斯过程代理模型

这些都不需要语言模型。
```

### 5.4 最终判断

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  大模型 (>7B):   ❌ 知识泄露不可接受                         │
│                                                              │
│  小模型 (1M-100M): ⚠️ 理论上可行但实际上无优势               │
│    - 如果训练数据干净 → 模型太弱, 不如直接搜索               │
│    - 如果训练数据有力 → 数据本身就含先验知识                  │
│                                                              │
│  不用语言模型:  ✅ 框架搜索是离散组合优化,                    │
│                    GP/RL/MCTS 更合适                         │
│                                                              │
│  唯一合理的LM使用场景:                                       │
│    用 NeSymReS 风格的小模型加速公式搜索 (Phase 1)            │
│    训练数据 = 随机生成的方程 (不含物理知识)                   │
│    这不是框架发现, 只是SR加速                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 框架发现 Agent 的具体设计

### 6.1 不需要 LLM 的架构

```
┌────────────────────────────────────────────────────────┐
│              Framework Discovery Agent                  │
│                                                         │
│  ┌──────────────┐                                      │
│  │ Failure      │ "当前文法下, 哪些数据拟合不了?"        │
│  │ Detector     │ 输入: SR结果 + 实验数据               │
│  │              │ 输出: 失败模式列表                     │
│  └──────┬───────┘                                      │
│         │                                               │
│  ┌──────▼───────┐                                      │
│  │ Grammar      │ "有哪些可能的文法修改?"                │
│  │ Mutator      │ 输入: 当前文法 + 失败模式              │
│  │              │ 输出: 候选文法修改列表                  │
│  └──────┬───────┘                                      │
│         │                                               │
│  ┌──────▼───────┐                                      │
│  │ Grammar      │ "修改后的文法更好吗?"                  │
│  │ Evaluator    │ 输入: 新文法 + 实验数据                │
│  │              │ 输出: (拟合改善, 复杂度增加) 分数      │
│  └──────┬───────┘                                      │
│         │                                               │
│  ┌──────▼───────┐                                      │
│  │ Search       │ "选择哪个修改?"                        │
│  │ Strategy     │ MCTS / GP / RL / Bayesian             │
│  │              │ 输出: 最优文法修改序列                  │
│  └──────────────┘                                      │
└────────────────────────────────────────────────────────┘
```

### 6.2 Failure Detector — 什么构成"失败"？

```python
class FailureDetector:
    """
    检测当前框架的不足, 不需要知道"正确"框架是什么

    失败类型 (从数据中可检测, 不需要先验):
    """

    def detect(self, sr_results, experiment_data):
        failures = []

        # 类型 1: 拟合精度差
        # "在这些数据上, 最好的公式也只有 R²=0.3"
        for exp, result in sr_results.items():
            if result.best_r2 < 0.9:
                failures.append(FitFailure(exp, result))

        # 类型 2: 随机性
        # "相同输入, 多次实验输出不同"
        for exp in experiment_data:
            repeated = exp.get_repeated_measurements()
            if repeated and variance(repeated) > threshold:
                failures.append(StochasticFailure(exp, variance=variance(repeated)))

        # 类型 3: 过拟合
        # "训练集R²=0.99但测试集R²=0.5"
        for exp, result in sr_results.items():
            if result.train_r2 - result.test_r2 > 0.3:
                failures.append(OverfitFailure(exp, gap=result.train_r2 - result.test_r2))

        # 类型 4: 复杂度爆炸
        # "需要极长的公式才能拟合, 违反简洁性"
        for exp, result in sr_results.items():
            if result.best_complexity > COMPLEXITY_THRESHOLD and result.best_r2 > 0.95:
                failures.append(ComplexityFailure(exp, complexity=result.best_complexity))

        # 类型 5: 跨实验不一致
        # "两个实验对同一实体给出矛盾的描述"
        for exp_i, exp_j in pairs_with_shared_entity(experiment_data):
            if contradicts(sr_results[exp_i], sr_results[exp_j]):
                failures.append(ContradictionFailure(exp_i, exp_j))

        return failures
```

### 6.3 Grammar Mutator — 文法修改操作的完整列表

```python
class GrammarMutator:
    """
    所有可能的文法修改操作 — 有限集合, 可枚举
    """

    MUTATIONS = [
        # ═══ Layer 1: 类型修改 ═══

        # 扩展数域: ℝ → ℂ
        # 动机: 如果两个实数函数 f,g 总是成对出现
        #       (如 cos 和 sin), 可能暗示需要 e^(ix) = cos(x)+i·sin(x)
        TypeExtend("Scalar", "ℝ", "ℂ"),

        # 添加向量/矩阵类型
        # 动机: 如果数据有多个输出之间有耦合关系
        TypeAdd("Vector", dimension="auto"),
        TypeAdd("Matrix", dimension="auto"),

        # 添加函数类型 (高阶)
        # 动机: 如果不同实验需要"不同的函数但结构相似"
        TypeAdd("Function", from_type="Scalar", to_type="Scalar"),

        # ═══ Layer 2: 运算修改 ═══

        # 添加新运算
        OpAdd("conjugate", ("ℂ",), "ℂ"),        # 复共轭
        OpAdd("norm_sq", ("ℂ",), "ℝ"),           # |z|²
        OpAdd("mat_mul", ("Matrix", "Matrix"), "Matrix"),
        OpAdd("apply", ("Matrix", "Vector"), "Vector"),
        OpAdd("inner", ("Vector", "Vector"), "Scalar"),

        # 移除公理 (放宽约束)
        AxiomRemove("commutative", "*"),  # 允许非交换乘法

        # ═══ Layer 3: 动力学修改 ═══

        # 修改方程阶数
        DynModify("order", 2, 1),  # 二阶ODE → 一阶
        DynModify("order", 1, 2),  # 一阶 → 二阶

        # 修改方程类型
        DynModify("linearity", "nonlinear", "linear"),  # 线性化

        # ═══ Layer 4: 解释规则修改 ═══

        # 从确定值到概率分布
        InterpModify("deterministic", "probabilistic"),
        # 即: output = f(input) → P(output) = g(input)

        # 从直接值到某函数的值
        InterpModify("direct", "squared"),
        # 即: observable = f → observable = f²

        InterpModify("direct", "norm_squared"),
        # 即: observable = f → observable = |f|²
        # 注意: 这个修改只有在 Scalar=ℂ 的文法中才合法
    ]
```

### 6.4 Grammar Evaluator — 评估文法修改的质量

```python
class GrammarEvaluator:
    """
    评估: 文法修改后, 对实验数据的拟合是否改善?

    方法: 在新文法下重新运行SR, 比较拟合质量
    """

    def evaluate(self, old_grammar, new_grammar, data, old_sr_results):
        # 1. 在新文法下, 重新定义SR的搜索空间
        new_operators = derive_operators(new_grammar)
        new_type_constraints = derive_type_constraints(new_grammar)

        # 2. 运行SR (使用新的运算符集和类型约束)
        new_sr_results = run_pysr(
            data,
            operators=new_operators,
            constraints=new_type_constraints,
        )

        # 3. 如果新文法包含概率解释
        if new_grammar.interpretation == "probabilistic":
            # 先估计经验分布, 再对分布做SR
            empirical_dist = estimate_distribution(data)
            new_sr_results = run_pysr_on_distribution(empirical_dist, ...)

        # 4. 计算改善
        fit_improvement = 0
        for exp in data.experiments:
            old_fit = old_sr_results[exp].best_r2
            new_fit = new_sr_results[exp].best_r2
            fit_improvement += max(0, new_fit - old_fit)

        # 5. 计算复杂度增加
        complexity_cost = (
            len(new_grammar.types) - len(old_grammar.types)
            + len(new_grammar.operations) - len(old_grammar.operations)
            + len(new_grammar.axioms) - len(old_grammar.axioms)
        )

        return fit_improvement, complexity_cost
```

### 6.5 Search Strategy — 在文法空间中的搜索

```
文法空间的结构:

  根节点: 经典力学文法 (ℝ, +, ×, commutative, ODE, deterministic)

  每个节点的子节点: 通过一步文法修改可达的新文法

  搜索树深度: 经典→量子 大约需要 3-5 步修改
    Step 1: deterministic → probabilistic (概率解释)
    Step 2: ℝ → ℂ (复数扩展)
    Step 3: +norm_sq (添加取模平方运算)
    Step 4: +Operator type (添加算符类型)
    Step 5: remove commutative(*) (移除交换律)

  每步的分支因子: ~20 (MUTATIONS列表大小)

  总搜索空间: ~20^5 = 320万 (深度5)
  → 这比SR的搜索空间小几个数量级!
  → MCTS 或 beam search 完全可以处理
```

**MCTS (蒙特卡洛树搜索) 实现：**

```python
class FrameworkMCTS:
    """
    在文法修改树上做 MCTS

    与 AlphaGo 的类比:
      棋盘状态 = 当前文法
      合法走子 = 文法修改操作
      胜负判定 = SR拟合质量 (在新文法下)

    区别:
      不需要神经网络做价值估计 (搜索空间小得多)
      每一步的评估需要运行SR (慢, 但可并行)
    """

    def search(self, root_grammar, data, budget):
        root = MCTSNode(grammar=root_grammar)

        for _ in range(budget):
            # Selection: UCB1 选择最有希望的路径
            node = self.select(root)

            # Expansion: 尝试一个新的文法修改
            child = self.expand(node)

            # Simulation: 在新文法下运行SR, 评估质量
            reward = self.simulate(child, data)

            # Backpropagation: 更新路径上所有节点的统计量
            self.backpropagate(child, reward)

        # 返回最优路径
        return self.best_path(root)
```

---

## 7. 关键的诚实性问题

### 这个方法本身是否构成作弊？

```
审查: MUTATIONS 列表中包含了
  TypeExtend("Scalar", "ℝ", "ℂ")
  InterpModify("direct", "norm_squared")
  AxiomRemove("commutative", "*")

这些修改加在一起就是量子力学!
把它们列在候选操作中, 是否等于告诉系统答案?
```

**分析：**

```
关键区别: 候选操作 vs 搜索路径

  候选操作是一个通用的、有限的文法修改集合
  它不只能到达量子力学 — 它能到达任何文法

  例如, 同样的操作集也能到达:
    - 非交换几何 (TypeExtend + AxiomRemove)
    - 概率论 (InterpModify alone)
    - 矩阵力学 (TypeAdd Matrix)
    - 张量场论 (TypeAdd Tensor)
    - 布尔逻辑 (TypeExtend ℝ→{0,1})

  总路径数 ~320万, 通往量子力学的路径只占一小部分

  类比: 符号回归的运算符集 {+,-,×,÷,sin,cos,exp,log}
       也"包含"了 F=ma 的组成元素
       但没人说这是作弊 — 因为运算符集是通用的

  同理: 文法修改集是通用的, 不特指量子力学
```

**但仍需验证：**

```
替代物理测试仍然是必要的:
  1. 在经典力学数据上运行 → 不应该修改文法 (经典文法已够用)
  2. 在虚构物理数据上运行 → 应该到达不同的文法
  3. 在随机数据上运行 → 不应该找到有意义的文法修改
```

### MUTATIONS 列表的完备性问题

```
问: 如果真正的物理需要一个不在 MUTATIONS 列表中的修改怎么办?
答: 这确实是一个限制.

  MUTATIONS 列表的构建原则应该是:
  "基于抽象代数和数学逻辑中的基本概念"
  而非 "基于物理学中出现过的结构"

  具体来说, 类型扩展/添加 来自代数 (数域扩张, 向量空间, 模)
  运算修改 来自普适代数 (公理系统的修改)
  解释规则修改 来自概率论和测度论

  这些是数学的基本语言, 不是物理特有的

  但: 如果需要的框架涉及拓扑、范畴论等更抽象的数学,
  当前的 MUTATIONS 列表可能不够.
  → 这是一个可以在 Phase 3 中逐步扩展的方面
```

---

## 8. 总结

```
核心回答:

Q: 什么样的 agent 可能发现新框架?
A: 在 "类型化文法空间" 上做搜索的 agent.
   框架 = 文法. 框架发现 = 文法搜索.
   搜索空间是有限的、离散的、可枚举的 (~百万级).

Q: 如何不用 LLM 表示框架?
A: 用类型化文法: (类型集, 运算集, 公理集, 动力学规则, 解释规则).
   完全形式化, 不需要自然语言.

Q: LLM 是否合理?
A: 大模型: 不合理 (知识泄露).
   小模型: 无优势 (训练数据问题, 不如直接搜索).
   框架搜索是离散组合优化, MCTS/GP/RL 更合适.

Q: 这种方法是否构成作弊?
A: 文法修改集是通用的数学操作 (类型扩张、公理修改等),
   不特指量子力学. 与SR的运算符集同级.
   但仍需通过替代物理测试验证.
```
