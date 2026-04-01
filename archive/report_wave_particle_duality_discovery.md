# Multi-Agent System for Autonomous Discovery of Wave-Particle Duality

## Technical Report v1.0

---

## 1. Executive Summary

本报告提出一种 **无知识泄露 (Zero Knowledge Leakage)** 的多智能体系统架构，用于从原始实验数据中自主发现波粒二象性。系统完全不使用大语言模型（LLM），而是依赖符号回归、遗传编程、信息论度量和强化学习等纯数学/算法方法，确保发现过程的科学严谨性。

**核心目标：** 给定一组可交互的物理实验环境，系统能够自主地：
1. 观察到光/电子同时展现粒子性和波动性
2. 提出统一的数学描述（如 E=hf, λ=h/p）
3. 发现这两种看似矛盾的性质是同一实体的不同表现

---

## 2. 问题分析：为什么波粒二象性比牛顿第二定律更难？

### 2.1 牛顿第二定律 vs 波粒二象性

| 维度 | 牛顿第二定律 (F=ma) | 波粒二象性 |
|------|---------------------|-----------|
| 数据类型 | 单一类型（力、质量、加速度） | 多模态（干涉条纹、光电流、散射角...） |
| 数学结构 | 简单代数关系 | 需要概率性描述 + 波函数 |
| 认知跳跃 | 量之间的比例关系 | 需要接受同一实体有两种互补描述 |
| 核心常数 | 无需发现新常数 | 需要发现普朗克常数 h |
| 范式冲突 | 无 | 粒子模型和波模型在经典物理中互斥 |

### 2.2 发现波粒二象性的最小充分条件

系统至少需要从数据中提取以下认知：

```
[认知1] 光在传播中表现出干涉和衍射 → 波动性
[认知2] 光与物质相互作用时表现为离散量子 → 粒子性
[认知3] 两种行为由同一物理常数 h 统一联系
[认知4] E = hf (能量-频率关系)
[认知5] λ = h/p (德布罗意关系)
[认知6] 电子等"粒子"也展现波动性 → 普适性
```

---

## 3. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator (编排层)                  │
│         调度实验、协调智能体、检测收敛                       │
└────────┬──────────────┬──────────────┬──────────────────┘
         │              │              │
    ┌────▼────┐   ┌─────▼─────┐  ┌────▼─────┐
    │ Explorer │   │ Modeler   │  │ Critic   │
    │ 探索者   │   │ 建模者    │  │ 审查者   │
    │(实验设计) │   │(公式发现) │  │(模型验证) │
    └────┬────┘   └─────┬─────┘  └────┬─────┘
         │              │              │
    ┌────▼──────────────▼──────────────▼──────────────────┐
    │              Shared Knowledge Base                    │
    │         (观测数据、已发现公式、异常记录)                 │
    └────────────────────┬────────────────────────────────┘
                         │
    ┌────────────────────▼────────────────────────────────┐
    │              Environment Layer (环境层)               │
    │                                                      │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
    │  │ 双缝实验  │ │ 光电效应  │ │康普顿散射 │ │电子衍射│ │
    │  │Simulator │ │Simulator │ │Simulator │ │Simulator│ │
    │  └──────────┘ └──────────┘ └──────────┘ └────────┘ │
    └─────────────────────────────────────────────────────┘
```

---

## 4. 环境层 (Environment Layer) 详细设计

### 4.1 设计原则

环境层是整个系统的"自然界"。它必须：
- **物理忠实**：基于已知物理方程精确模拟，但只输出"实验观测数据"，不暴露底层方程
- **可交互**：智能体可以调节实验参数（缝宽、光频率、入射角等）
- **有噪声**：加入合理的实验噪声，模拟真实实验条件
- **API化**：统一接口，智能体不需要知道底层实现

### 4.2 四个关键实验环境

#### 实验 A：双缝干涉实验 (Double-Slit Experiment)

```python
class DoubleSlitEnv:
    """
    输入参数:
        - particle_type: "photon" | "electron"  (智能体选择)
        - slit_width: float (缝宽, meters)
        - slit_separation: float (缝间距, meters)
        - wavelength_or_energy: float (光波长 或 电子能量)
        - detector_resolution: float (探测器分辨率)
        - num_particles: int (发射粒子数)
        - single_particle_mode: bool (是否逐个发射)

    输出数据:
        - detector_hits: List[(x, y, t)]  # 每个粒子的到达位置和时间
        - intensity_pattern: array  # 探测屏上的强度分布

    关键现象 (需被智能体发现):
        1. 大量粒子 → 干涉条纹 (波动性)
        2. 单个粒子 → 离散点击 (粒子性)
        3. 逐个发射仍产生干涉 → 自干涉
        4. 遮住一条缝 → 干涉消失
    """

    def step(self, params: dict) -> ExperimentResult:
        # 内部用量子力学计算，但只返回观测数据
        # 使用 Fraunhofer/Fresnel 衍射公式
        # P(x) = |ψ_1(x) + ψ_2(x)|^2
        # 单粒子模式：按概率分布采样离散点
        pass
```

#### 实验 B：光电效应 (Photoelectric Effect)

```python
class PhotoelectricEnv:
    """
    输入参数:
        - metal_type: str (不同逸出功的金属)
        - light_frequency: float (入射光频率)
        - light_intensity: float (光强)
        - voltage: float (外加电压，用于测截止电压)

    输出数据:
        - photocurrent: float (光电流)
        - max_kinetic_energy: float (最大动能，通过截止电压测量)
        - emission_delay: float (发射延迟时间)

    关键现象:
        1. 存在截止频率 (经典波理论无法解释)
        2. 最大动能与频率线性相关，与强度无关
        3. 光电流与强度成正比
        4. 几乎无发射延迟
    """

    # 内部: E_k = hf - W (爱因斯坦方程)
```

#### 实验 C：康普顿散射 (Compton Scattering)

```python
class ComptonScatteringEnv:
    """
    输入参数:
        - incident_wavelength: float (入射X射线波长)
        - scattering_angle: float (散射角)
        - target_material: str

    输出数据:
        - scattered_wavelength: float
        - electron_recoil_energy: float
        - electron_recoil_angle: float

    关键现象:
        1. 散射后波长变长 (经典波理论: 波长不变)
        2. Δλ 与散射角的关系
        3. 能量-动量守恒符合粒子碰撞模型
    """

    # 内部: Δλ = (h/mc)(1 - cosθ)
```

#### 实验 D：电子衍射 (Davisson-Germer Experiment)

```python
class ElectronDiffractionEnv:
    """
    输入参数:
        - electron_energy: float (加速电压)
        - crystal_type: str (不同晶格常数)
        - detector_angle: float

    输出数据:
        - diffraction_intensity: float (各角度的衍射强度)
        - intensity_pattern: array (完整衍射图样)

    关键现象:
        1. 电子（公认的粒子）产生衍射图案
        2. 衍射峰位置满足 Bragg 条件
        3. 等效波长与电子动量关系: λ = h/p
    """
```

### 4.3 环境层的反知识泄露设计

```
┌─────────────────────────────────────────────┐
│            Knowledge Firewall                │
│                                              │
│  ✗ 不暴露内部方程                              │
│  ✗ 不提供物理常数名称 (如 "普朗克常数")          │
│  ✗ 不标注物理量的名称 (用 param_1, obs_1 等)     │
│  ✓ 只返回数值数据                              │
│  ✓ 参数用无语义标签 (channel_A, channel_B)      │
│  ✓ 不同实验的参数不暗示它们之间的联系             │
└─────────────────────────────────────────────┘
```

**更严格的匿名化方案（推荐）：**

```python
# 环境对外接口 - 完全去语义化
class AnonymizedExperiment:
    """
    实验被编号为 Env_001, Env_002, ...
    参数被编号为 input_0, input_1, ...
    输出被编号为 obs_0, obs_1, ...

    智能体只知道:
    - 有哪些输入旋钮可以调
    - 每个旋钮的取值范围
    - 输出是什么格式的数值
    """

    def get_config(self) -> dict:
        return {
            "env_id": "ENV_003",
            "inputs": {
                "input_0": {"range": [1e-10, 1e-7], "type": "continuous"},
                "input_1": {"range": [0, 3.14], "type": "continuous"},
                "input_2": {"options": ["MAT_A", "MAT_B"], "type": "discrete"}
            },
            "outputs": {
                "obs_0": {"type": "continuous"},
                "obs_1": {"type": "array_1d"}
            }
        }

    def run(self, inputs: dict) -> dict:
        # 返回纯数值，无任何物理语义
        return {"obs_0": 2.43e-12, "obs_1": [0.01, 0.23, ...]}
```

---

## 5. 智能体层 (Agent Layer) 详细设计

### 5.1 核心约束：零知识泄露

```
┌─────────────────────────────────────────────────────┐
│              FORBIDDEN (禁止使用)                     │
│                                                      │
│  ✗ LLM (GPT, Claude, LLaMA, ...) — 训练数据含物理   │
│  ✗ 预训练的科学基础模型                               │
│  ✗ 任何包含物理先验知识的组件                          │
│  ✗ 硬编码的物理常数或公式模板                          │
│  ✗ 带有物理语义的特征工程                             │
│                                                      │
│              ALLOWED (允许使用)                        │
│                                                      │
│  ✓ 符号回归 (Symbolic Regression)                    │
│  ✓ 遗传编程 (Genetic Programming)                    │
│  ✓ 从零训练的神经网络 (不用预训练权重)                 │
│  ✓ 贝叶斯优化                                        │
│  ✓ 信息论方法 (互信息、KL散度)                        │
│  ✓ 聚类、降维等无监督学习                              │
│  ✓ 基本数学运算库 (numpy, scipy)                     │
│  ✓ 强化学习 (从零训练)                                │
│  ✓ 图搜索、约束求解                                   │
└─────────────────────────────────────────────────────┘
```

### 5.2 Agent 1: Explorer (探索者) — 实验设计智能体

**职责：** 决定下一步做什么实验，调什么参数

**算法核心：** 基于信息增益的主动学习 + 强化学习

```
Explorer 的决策循环:

1. 观察当前知识状态 K = {已知公式, 异常点, 未探索区域}
2. 对每个可选实验 e ∈ Experiments:
     对每组参数 θ ∈ ParamSpace(e):
       估计信息增益 IG(e, θ | K)
3. 选择 (e*, θ*) = argmax IG(e, θ | K)
4. 执行实验，获取数据
5. 将数据传入 Shared Knowledge Base
```

**信息增益的计算：**

```python
def estimate_information_gain(experiment, params, current_models):
    """
    信息增益 = 实验结果的预期"惊讶度"

    如果当前模型能很好地预测某实验结果 → 低信息增益
    如果当前模型对某区域预测分歧很大 → 高信息增益
    如果某参数区域从未探索过 → 高信息增益
    """
    predictions = [m.predict(experiment, params) for m in current_models]

    # 模型间分歧
    disagreement = variance(predictions)

    # 探索奖励 (该参数区域的数据密度)
    exploration_bonus = 1.0 / (data_density(experiment, params) + eps)

    # 异常区域奖励 (之前模型预测失败的区域附近)
    anomaly_bonus = proximity_to_anomalies(experiment, params)

    return w1 * disagreement + w2 * exploration_bonus + w3 * anomaly_bonus
```

**Explorer 的 RL 训练（可选增强）：**

```
State:  当前知识库状态的嵌入向量
Action: (实验选择, 参数设置)
Reward: Modeler 在新数据后发现的模型质量提升
        + Critic 发现的新异常数量
Policy: 从零训练的 PPO / SAC agent
```

### 5.3 Agent 2: Modeler (建模者) — 公式发现智能体

**职责：** 从数据中发现数学关系

**核心算法：多层级符号回归**

```
┌─────────────────────────────────────────────┐
│          Modeler 的三层发现架构               │
│                                              │
│  Layer 3: Cross-Experiment Unifier           │
│  (跨实验统一 — 寻找共享常数和统一框架)         │
│           ▲                                  │
│  Layer 2: Single-Experiment Modeler          │
│  (单实验建模 — 对每个实验找最佳公式)           │
│           ▲                                  │
│  Layer 1: Pattern Detector                   │
│  (模式探测 — 相关性、周期性、离散性检测)       │
│           ▲                                  │
│       Raw Data                               │
└─────────────────────────────────────────────┘
```

#### Layer 1: Pattern Detector (模式探测器)

```python
class PatternDetector:
    """纯统计方法，检测数据中的基本模式"""

    def detect_patterns(self, data):
        results = {}

        # 1. 相关性分析: 哪些输入影响哪些输出
        results["correlations"] = mutual_information_matrix(
            data.inputs, data.outputs
        )

        # 2. 线性/非线性判定
        results["linearity"] = linearity_test(data)
        # Ramsey RESET test

        # 3. 离散性检测: 输出是否呈离散分布
        results["discreteness"] = discreteness_score(data.outputs)
        # 用 KDE + 峰值检测

        # 4. 周期性检测: 输出是否有周期性模式
        results["periodicity"] = fft_peak_analysis(data.outputs)

        # 5. 阈值效应: 是否存在截止点
        results["thresholds"] = threshold_detection(
            data.inputs, data.outputs
        )
        # 分段回归 + 断点检测

        # 6. 分布形状: 输出的统计分布
        results["distribution"] = distribution_fitting(data.outputs)

        return results
```

#### Layer 2: Single-Experiment Symbolic Regression

**使用遗传编程 (GP) 进行符号回归：**

```python
class SymbolicRegressor:
    """
    基于 gplearn / PySR 风格的遗传编程符号回归

    基因型: 表达式树
    适应度: 拟合精度 + 复杂度惩罚 (Pareto前沿)
    """

    # 允许的基本运算 (无物理先验)
    OPERATORS = {
        "unary":  [sin, cos, exp, log, sqrt, abs, neg, inv],
        "binary": [add, sub, mul, div, pow],
    }

    # 终端符号
    TERMINALS = ["input_0", "input_1", ..., "C"]
    # C = 可学习常数，通过梯度下降优化

    def evolve(self, data, generations=5000, pop_size=2000):
        """
        1. 初始化随机表达式种群
        2. 每一代:
           a. 评估适应度 = accuracy(expr, data) - λ * complexity(expr)
           b. 选择 (锦标赛选择)
           c. 交叉 (子树交换)
           d. 变异 (点变异、子树变异、常数微调)
           e. 精英保留
        3. 返回 Pareto 前沿: {(accuracy, complexity, expression)}
        """

    def fit_constants(self, expression, data):
        """
        对表达式中的常数 C 进行梯度下降优化
        使用 BFGS 或 L-BFGS
        """
```

**关键增强：概率模型发现**

波粒二象性的核心在于概率性。标准符号回归找确定性公式 `y = f(x)` 是不够的。

```python
class ProbabilisticSymbolicRegressor:
    """
    不仅找 y = f(x)
    还能找 P(y|x) = g(y; f(x), σ(x))
    即输出的概率分布模型

    这对发现量子现象至关重要:
    - 双缝实验中，单粒子到达位置是概率性的
    - 需要发现 P(x) = |ψ(x)|^2 这种概率分布
    """

    def evolve(self, data):
        # 拟合目标变为最大化似然，而非最小化MSE
        # fitness = log_likelihood(data | expression) - λ * complexity
        pass
```

#### Layer 3: Cross-Experiment Unifier (跨实验统一器)

**这是发现波粒二象性的关键层。**

```python
class CrossExperimentUnifier:
    """
    输入: 各实验的最优公式集合
    输出: 跨实验的统一关系

    核心思想: 在不同实验的公式中寻找共享结构
    """

    def find_shared_constants(self, formulas: Dict[str, Expression]):
        """
        步骤1: 提取所有公式中的数值常数

        例如:
          Env_001 (光电效应): obs_0 = C1 * input_0 - C2
          Env_002 (康普顿散射): obs_0 = C3 / (input_1) * (1 - cos(input_0))
          Env_003 (双缝干涉): P(x) 的周期 ~ C4 / input_0

        步骤2: 比较常数值
          如果 C1 ≈ C3 ≈ C4 (在误差范围内)
          → 标记为 "Universal Constant UC_1" (即普朗克常数 h)

        步骤3: 用统一常数重写所有公式
        """

        all_constants = {}
        for env_id, formula in formulas.items():
            all_constants[env_id] = extract_constants(formula)

        # 聚类: 找数值接近的常数组
        all_values = flatten(all_constants.values())
        clusters = hierarchical_clustering(
            all_values,
            criterion="relative_tolerance",
            tolerance=0.01  # 1% 相对误差
        )

        # 也检查常数的简单组合 (C1*C2, C1/C2, etc.)
        extended_values = generate_combinations(all_values)
        extended_clusters = hierarchical_clustering(extended_values, ...)

        return universal_constants, rewritten_formulas

    def find_structural_analogies(self, formulas):
        """
        在公式的"骨架"层面寻找类比

        例如:
          如果光电效应给出 E = UC_1 * f
          而电子衍射给出 λ = UC_1 / p

        → 发现 E * λ = UC_1^2 / p
        → 或者 E = UC_1 * c / λ  (如果能从数据中推出 c)
        → 统一: 能量和波长通过同一常数 UC_1 关联
        """
```

### 5.4 Agent 3: Critic (审查者) — 模型验证与质疑智能体

**职责：** 验证模型、发现异常、提出反例

```python
class CriticAgent:
    """
    Critic 的核心功能:

    1. 模型验证 — 在新数据上测试模型预测
    2. 异常检测 — 发现模型无法解释的数据点
    3. 反例搜索 — 主动寻找能推翻模型的参数区域
    4. 一致性检查 — 检测不同模型间的矛盾
    5. 简洁性评估 — Occam's razor, MDL原则
    """

    def validate_model(self, model, experiment, n_tests=100):
        """在随机参数下测试模型预测精度"""
        errors = []
        for _ in range(n_tests):
            params = sample_random_params(experiment)
            predicted = model.predict(params)
            observed = experiment.run(params)
            errors.append(prediction_error(predicted, observed))

        return {
            "mean_error": mean(errors),
            "max_error": max(errors),
            "error_distribution": errors,
            "worst_case_params": get_worst_params(errors),
            "pass": mean(errors) < threshold
        }

    def find_contradictions(self, models: List[Model]):
        """
        检查模型集合的内部一致性

        关键场景:
        - 模型A说光是波 (干涉实验)
        - 模型B说光是粒子 (光电效应)
        → 标记为 "PARADOX" → 触发 Unifier 寻找统一框架
        """
        contradictions = []
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i >= j:
                    continue
                # 检查是否存在参数区域，两个模型给出矛盾预测
                conflict = adversarial_search(m1, m2)
                if conflict:
                    contradictions.append({
                        "model_1": m1, "model_2": m2,
                        "conflict_region": conflict,
                        "severity": conflict.score
                    })
        return contradictions

    def compute_model_score(self, model, data):
        """
        综合评分 = 拟合质量 + 泛化能力 + 简洁性

        使用 MDL (Minimum Description Length) 原则:
        Score = -log P(data|model) + L(model)

        L(model) = 模型描述长度 (表达式树的节点数 + 常数的精度位数)
        """
        fit_score = negative_log_likelihood(data, model)
        complexity = description_length(model)
        return fit_score + complexity
```

### 5.5 Orchestrator (编排层)

```python
class Orchestrator:
    """
    编排层的核心状态机
    """

    # 系统发现阶段
    PHASES = {
        "EXPLORATION":   "广泛探索各实验，积累数据",
        "MODELING":      "对单个实验寻找数学关系",
        "ANOMALY":       "深入研究模型失败的区域",
        "UNIFICATION":   "跨实验寻找统一框架",
        "VERIFICATION":  "对统一理论进行严格验证",
    }

    def main_loop(self):
        phase = "EXPLORATION"

        while not self.convergence_reached():
            if phase == "EXPLORATION":
                # Explorer 在所有实验中广泛采样
                for env in self.environments:
                    params = self.explorer.suggest(env, mode="broad")
                    data = env.run(params)
                    self.knowledge_base.add(env.id, params, data)

                # 每个实验积累足够数据后 → 进入建模
                if all(self.knowledge_base.count(env) > MIN_DATA
                       for env in self.environments):
                    phase = "MODELING"

            elif phase == "MODELING":
                # Modeler 对每个实验独立建模
                for env in self.environments:
                    data = self.knowledge_base.get(env.id)
                    models = self.modeler.fit(data)  # 符号回归
                    self.knowledge_base.add_models(env.id, models)

                # Critic 验证模型
                anomalies = self.critic.validate_all()

                if anomalies:
                    phase = "ANOMALY"
                else:
                    phase = "UNIFICATION"

            elif phase == "ANOMALY":
                # Explorer 针对异常区域设计实验
                for anomaly in anomalies:
                    params = self.explorer.suggest(
                        anomaly.env,
                        mode="focused",
                        focus_region=anomaly.params
                    )
                    data = anomaly.env.run(params)
                    self.knowledge_base.add(anomaly.env.id, params, data)

                # 重新建模
                phase = "MODELING"

            elif phase == "UNIFICATION":
                # 跨实验统一
                all_models = self.knowledge_base.get_all_models()
                unified = self.modeler.unify(all_models)

                # Critic 检查统一模型
                if self.critic.validate_unified(unified):
                    phase = "VERIFICATION"
                else:
                    phase = "ANOMALY"

            elif phase == "VERIFICATION":
                # 最终验证: 统一理论能否预测新实验结果
                success = self.critic.comprehensive_test(
                    self.knowledge_base.unified_model,
                    n_tests=1000
                )
                if success:
                    self.report_discovery()
                    break
                else:
                    phase = "ANOMALY"
```

---

## 6. 发现路径的预期演化

以下是系统预期的发现过程（基于信息论分析，而非事先编排）：

```
Phase 1: 初始探索 (Iteration 0 ~ 500)
├── 发现 Env_001(双缝): 强度分布有周期性条纹模式
├── 发现 Env_002(光电): obs_0 与 input_0 线性关系, 存在截止点
├── 发现 Env_003(康普顿): 波长偏移 与 角度的余弦关系
└── 发现 Env_004(电子衍射): 离散的强度峰

Phase 2: 单实验建模 (Iteration 500 ~ 2000)
├── Env_001: P(x) = A * cos²(π * d * x / (C₁ * L))  [干涉公式]
├── Env_002: E_k = C₂ * f - W                        [光电方程]
├── Env_003: Δλ = C₃ * (1 - cos θ)                  [康普顿公式]
└── Env_004: 峰位满足 2d*sin θ = n * C₄/√(2mE)      [布拉格条件+德布罗意]

Phase 3: 异常发现 (Iteration 2000 ~ 3000)
├── 异常1: 双缝实验中单粒子到达是离散的，但积累后成波状
│   → Critic标记: "离散-连续矛盾"
├── 异常2: 光电效应中，低于截止频率的光无论多强都不行
│   → Critic标记: "强度-频率悖论" (经典波理论预测强光应该能)
└── 异常3: 电子本应是粒子，却产生类似光的干涉条纹
    → Critic标记: "粒子-波矛盾"

Phase 4: 常数统一 (Iteration 3000 ~ 4000)
├── 发现 C₂ ≈ C₃ ≈ C₁ * 某组合 ≈ 6.626e-34
├── 统一为 Universal Constant UC₁
├── 所有公式用 UC₁ 重写:
│   E = UC₁ * f
│   λ = UC₁ / p
│   Δλ = UC₁/(mc) * (1-cosθ)
└── 命名: UC₁ (即普朗克常数 h)

Phase 5: 概念统一 (Iteration 4000 ~ 6000)
├── Unifier 发现:
│   "能量"和"频率"通过 UC₁ 关联 (E = UC₁ * f)
│   "波长"和"动量"通过 UC₁ 关联 (λ = UC₁ / p)
│   → 任何具有能量E的实体都有关联频率 f = E/UC₁
│   → 任何具有动量p的实体都有关联波长 λ = UC₁/p
│   → 波动描述和粒子描述是同一实体的两面
├── 统一概率框架:
│   P(x) = |Ψ(x)|²  (发现强度分布=某个波函数模方)
└── 验证: 统一模型在所有4个实验中预测误差 < 1%

═══════════════════════════════════════════
   DISCOVERY: Wave-Particle Duality

   核心发现:
   1. Universal Constant UC₁ = 6.626e-34
   2. E = UC₁ * f  (能量量子化)
   3. λ = UC₁ / p  (物质波)
   4. 所有实体同时具有波和粒子性质
   5. 观测结果本质上是概率性的: P = |Ψ|²
═══════════════════════════════════════════
```

---

## 7. 知识泄露防线：完整审计体系

### 7.1 三道防线

```
┌─────────────────────────────────────────────────┐
│         Defense Line 1: 架构防线                  │
│                                                   │
│  • 所有智能体从零构建，无预训练                      │
│  • 环境接口完全匿名化                              │
│  • 代码中不含物理术语                               │
│  • 运算符集合是通用数学运算，非物理导向              │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│         Defense Line 2: 运行时防线                │
│                                                   │
│  • 符号回归的初始种群完全随机生成                    │
│  • 常数通过数据拟合得到，而非预设                   │
│  • 记录所有公式的完整进化谱系                       │
│  • 任何公式都能追溯到数据驱动的变异/交叉操作         │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│         Defense Line 3: 验证防线                  │
│                                                   │
│  • 消融实验: 移除某实验的数据，看是否仍能发现       │
│  • 随机基线: 用随机数据替换真实数据，确认不会        │
│    "发现"同样的公式                               │
│  • 替代物理: 用不同物理常数的环境，确认系统          │
│    发现的是不同的常数值                             │
│  • 进化轨迹审计: 证明公式是逐步进化而来             │
└─────────────────────────────────────────────────┘
```

### 7.2 Anti-Leakage Test Protocol

```python
class AntiLeakageValidator:
    """验证系统确实是从数据中学习，而非泄露知识"""

    def test_1_altered_physics(self):
        """
        创建一个 h' = 2h 的平行宇宙环境
        系统应该发现 UC₁ = 2 * 6.626e-34
        而非真实的 h 值
        """
        altered_env = create_environments(h=2 * PLANCK_CONSTANT)
        result = run_full_system(altered_env)
        assert abs(result.UC_1 - 2 * PLANCK_CONSTANT) / (2 * PLANCK_CONSTANT) < 0.01

    def test_2_random_data(self):
        """
        用随机数据替换真实物理数据
        系统不应该发现任何有意义的公式
        """
        random_env = create_random_environments()
        result = run_full_system(random_env)
        assert result.unified_model is None or result.confidence < THRESHOLD

    def test_3_classical_only(self):
        """
        只提供经典物理能解释的实验数据
        系统不应该发现量子常数
        """
        classical_env = create_classical_environments()
        result = run_full_system(classical_env)
        assert not result.has_universal_constant_like(PLANCK_CONSTANT)

    def test_4_evolution_trace(self):
        """
        验证每个发现的公式都有完整的进化记录
        不能是"突然"出现最终公式
        """
        trace = system.get_evolution_trace()
        for formula in trace.final_formulas:
            ancestry = trace.get_ancestry(formula)
            assert len(ancestry) > MIN_EVOLUTION_STEPS
            assert all_mutations_are_valid(ancestry)
```

---

## 8. 技术栈与实现建议

### 8.1 推荐技术栈

| 组件 | 推荐工具 | 理由 |
|------|---------|------|
| 符号回归 | **PySR** (Julia后端) | 速度快，支持自定义运算符，活跃维护 |
| 遗传编程 | **DEAP** | Python原生GP框架，灵活性高 |
| 实验设计(RL) | **Stable-Baselines3** (从零训练) | 成熟的RL框架 |
| 信息论计算 | **scipy.stats** + 自定义 | 互信息、KL散度等 |
| 物理模拟 | **NumPy/SciPy** 自建 | 完全控制输出接口 |
| 聚类 | **scikit-learn** | 常数聚类、模式检测 |
| 可视化 | **Matplotlib** | 进化过程可视化 |
| 并行计算 | **Ray** | 大规模并行符号回归 |

### 8.2 计算资源估计

```
单实验符号回归 (5000代, 种群2000): ~2-4 GPU小时 (if using neural-guided SR)
                                    ~4-8 CPU小时 (纯GP)
Explorer RL训练: ~10-20 GPU小时
跨实验统一搜索: ~2-4 CPU小时
完整流程 (含反复迭代): ~50-100 GPU小时 / ~200-400 CPU小时
```

### 8.3 项目结构建议

```
ai-scientist/
├── environments/               # 环境层
│   ├── base.py                # 实验环境基类
│   ├── double_slit.py         # 双缝干涉模拟
│   ├── photoelectric.py       # 光电效应模拟
│   ├── compton.py             # 康普顿散射模拟
│   ├── electron_diffraction.py # 电子衍射模拟
│   └── anonymizer.py          # 匿名化包装器
│
├── agents/                     # 智能体层
│   ├── explorer/
│   │   ├── info_gain.py       # 信息增益计算
│   │   ├── active_learning.py # 主动学习策略
│   │   └── rl_policy.py       # RL实验设计策略
│   ├── modeler/
│   │   ├── pattern_detector.py    # 模式探测
│   │   ├── symbolic_regressor.py  # 符号回归
│   │   ├── probabilistic_sr.py    # 概率符号回归
│   │   └── cross_experiment.py    # 跨实验统一
│   └── critic/
│       ├── validator.py       # 模型验证
│       ├── anomaly_detector.py # 异常检测
│       └── consistency.py     # 一致性检查
│
├── orchestrator/               # 编排层
│   ├── main_loop.py           # 主循环
│   ├── knowledge_base.py      # 共享知识库
│   └── convergence.py         # 收敛判定
│
├── anti_leakage/              # 反知识泄露
│   ├── validator.py           # 泄露检测
│   └── altered_physics.py     # 替代物理测试
│
├── evaluation/                 # 评估
│   ├── metrics.py             # 发现质量度量
│   └── visualization.py       # 可视化
│
└── configs/                    # 配置
    └── default.yaml
```

---

## 9. 核心挑战与缓解策略

### 9.1 挑战分析

| 挑战 | 严重程度 | 缓解策略 |
|------|---------|---------|
| 符号回归的搜索空间指数爆炸 | **高** | 分层搜索 + 模块化构建 + Pareto前沿剪枝 |
| 概率性公式难以发现 | **高** | 专用概率SR模块 + 分布拟合先行 |
| 跨实验统一需要"概念跳跃" | **极高** | 常数聚类 + 结构类比 + 多尺度搜索 |
| 计算成本 | **中** | Ray并行 + 增量式搜索 + 早停策略 |
| 如何判定"发现了波粒二象性" | **中** | 明确的成功标准（见下文） |

### 9.2 最大风险：概念跳跃

系统最困难的部分是从"光电效应说光是粒子"和"干涉实验说光是波"到"光既是粒子又是波"的认知跳跃。

**缓解方案：矛盾驱动的探索**

```python
class ContradictionResolver:
    """
    当 Critic 发现两个模型给出矛盾描述时:

    1. 不是选择一个、抛弃另一个
    2. 而是寻找一个更高层次的模型，同时包含两者

    数学实现:
    - 将矛盾的两个模型视为更一般模型在不同极限下的近似
    - 搜索形如 f(x; α) 的参数化族，
      使得 f(x; α→0) ≈ model_1  且 f(x; α→∞) ≈ model_2
    """

    def resolve(self, model_wave, model_particle, data):
        # 构建统一搜索空间
        # 约束: 新模型必须在波实验数据上≈model_wave
        #        且在粒子实验数据上≈model_particle

        combined_data = merge(
            data["wave_experiments"],
            data["particle_experiments"]
        )

        # 用约束GP搜索
        unified = constrained_symbolic_regression(
            data=combined_data,
            constraints=[
                ApproximatesIn(model_wave, wave_regime),
                ApproximatesIn(model_particle, particle_regime),
            ]
        )
        return unified
```

---

## 10. 成功标准

### 10.1 定量成功标准

```yaml
Level 1 - 基础发现 (最低要求):
  - 在光电效应数据中找到 E = C * f 关系
  - 在双缝实验中找到干涉条纹的数学描述
  - 精度: R² > 0.99

Level 2 - 常数统一:
  - 识别出不同实验中出现的同一常数 (h)
  - 常数值误差 < 2%

Level 3 - 概念统一 (核心目标):
  - 发现 E = hf 和 λ = h/p 的对偶关系
  - 发现"粒子"(电子) 也有波动性
  - 提出统一的数学描述

Level 4 - 概率框架 (理想目标):
  - 发现观测结果的概率性本质
  - 提出 P(x) = |Ψ(x)|² 形式的概率分布
  - 在所有实验中统一验证
```

### 10.2 反知识泄露验证标准

```yaml
必须通过:
  - 修改 h 值测试: 系统发现修改后的 h' (误差 < 5%)
  - 随机数据测试: 系统不报告有意义的发现
  - 进化轨迹测试: 所有公式可追溯至随机初始化

加分项:
  - 经典物理测试: 只给经典数据时不发现量子常数
  - 泛化测试: 发现的理论能预测未训练过的实验
```

---

## 11. 与已有工作的对比

| 项目 | 方法 | 发现 | 是否防泄露 |
|------|------|------|-----------|
| AI Feynman (Udrescu 2020) | 神经网络+符号回归 | 从数据恢复已知物理公式 | 部分 (给定正确变量) |
| SciNet (Iten 2020) | 自编码器 | 发现潜在物理概念 | 是 |
| Rediscovery of F=ma (各团队) | 符号回归 | 牛顿第二定律 | 是 |
| **本方案** | **多智能体+符号回归+矛盾驱动** | **波粒二象性** | **三重防线** |

**本方案的独特贡献：**
1. 首次尝试自主发现 **互补性原理** 级别的概念
2. **矛盾驱动** 的发现机制（利用模型间矛盾推动概念跳跃）
3. **概率符号回归** 用于发现量子现象的概率本质
4. **三重反知识泄露** 验证体系

---

## 12. 实施路线图

```
Month 1-2: 环境层
├── 实现4个物理实验模拟器
├── 实现匿名化包装层
├── 单元测试 + 物理精度验证
└── 反泄露接口审计

Month 3-4: 单智能体原型
├── 实现 Pattern Detector
├── 实现基础符号回归 (PySR集成)
├── 实现概率符号回归
└── 在单实验上验证 (先验证光电效应能找到 E=hf)

Month 5-6: 多智能体集成
├── 实现 Explorer (信息增益 + RL)
├── 实现 Critic (验证 + 异常检测)
├── 实现 Cross-Experiment Unifier
├── 实现 Orchestrator 主循环
└── 实现 Contradiction Resolver

Month 7-8: 端到端实验 + 优化
├── 完整流程运行
├── 超参数调优
├── 计算优化 (Ray并行化)
├── 反知识泄露完整测试
└── 消融实验

Month 9: 论文撰写 + 开源
```

---

## 13. 结论

本方案提出了一条从原始实验数据自主发现波粒二象性的可行路径。其核心创新在于：

1. **矛盾驱动的概念跳跃**：利用 Critic 检测到的模型矛盾作为发现新物理的信号
2. **概率符号回归**：超越确定性公式发现，能捕捉量子现象的概率本质
3. **跨实验常数统一**：通过数值聚类自动发现普朗克常数
4. **三重反泄露防线**：确保发现过程的科学可信度

最大的挑战在于"概念统一"这一步——从"波模型"和"粒子模型"的矛盾中发现互补性原理。这需要 ContradictionResolver 能够搜索到足够一般化的数学框架。这一挑战的难度不应被低估，但通过精心设计的搜索策略和足够的计算资源，我们认为这是可以实现的。

---

*Report generated: 2026-03-30*
*Project: AI Scientist — Autonomous Discovery of Wave-Particle Duality*
