# Brainstorm v5: K 测量的精确化 — 核心技术难点

---

## 问题：K 取决于"任务定义"

SciNet 的 K 不是系统的固有属性——它取决于你让 SciNet 预测什么。

```
例: Stern-Gerlach 实验

任务 A: "给定磁场方向 θ, 预测粒子到达位置的分布"
  → SciNet 输入 = θ, 输出 = P(up|θ) = cos²(θ/2)
  → 瓶颈 K = 1 就够了 (θ 本身就是充分统计量)

任务 B: "给定制备参数 (a,b,c) 和测量方向 θ,
         预测粒子到达位置的分布"
  → 现在需要 K = 3 来编码制备状态 (Bloch 球)
  → 因为不同的制备设置在不同的 θ 下给出不同的 P

关键: 同一个物理系统, 不同的任务定义, K 不同!
```

**这不是 bug，这是 feature。** K 增大意味着系统发现了"更多的可控自由度"。

---

## 解决方案：渐进式任务扩展

让系统自动决定任务定义，从最简单开始，逐步扩展：

```python
class AdaptiveKMeasurer:
    """
    渐进式测量 K:
    1. 从最简单的任务开始 (一个 knob → 一个 detector)
    2. 测量 K₁
    3. 逐步增加更多 knob 到输入中
    4. 测量新的 K
    5. 如果 K 增大 → 说明新 knob 引入了新自由度
    6. 如果 K 不变 → 新 knob 不引入新信息
    """

    def measure_progressive(self, env):
        knobs = env.get_all_knobs()
        K_history = []

        # 从空输入开始, 逐步添加 knob
        for i in range(len(knobs)):
            input_knobs = knobs[:i+1]

            # SciNet: 用 input_knobs 预测 detector 输出
            K_i = self.run_scinet(
                input_features=input_knobs,
                output_features=env.detectors,
                data=env.collect_data(n=10000)
            )

            K_history.append({
                "n_knobs": i + 1,
                "K": K_i,
                "knobs_used": input_knobs
            })

        return K_history

    # K_history 可能看起来像:
    # [
    #   {"n_knobs": 1, "K": 1, "knobs_used": [knob_0]},
    #   {"n_knobs": 2, "K": 1, "knobs_used": [knob_0, knob_1]},  # knob_1 没加信息
    #   {"n_knobs": 3, "K": 3, "knobs_used": [knob_0, knob_1, knob_2]},  # K 跳到 3!
    # ]
    # → knob_2 引入了新自由度!
    # → 用 3 个 knob 描述系统时需要 K=3 维内部表示
    # → 这已经是一个有意义的发现
```

---

## 更深的洞察：K 的"跳变"就是发现

```
对于经典系统:
  K 随 knob 数线性增长: K ≈ n_knobs
  每个 knob 是一个独立自由度

对于量子系统:
  K 可能超线性增长: 当某些 knob 组合揭示了纠缠或干涉效应
  K > n_knobs 意味着系统有"内部自由度"不对应任何单个 knob

系统不需要知道"量子力学"
它只需要发现: "这个系统的内部维度 > 我可控参数的数量"
这本身就是一个可操作的、可量化的发现
```

---

## SciNet 在 ATLAS 中的精确定位

```
SciNet 不是用来"发现物理概念"的 (虽然 Iten 2020 这样说)
在 ATLAS 中, SciNet 有一个更精确的角色:

  SciNet = 维度探测器 (Dimension Probe)

  给定: 一组 knob→detector 的实验数据
  输出: 描述 knob→detector 映射需要的最小内部维度 K

  K 的值告诉 DSL Diagnoser:
    K ≤ n_knobs → 不需要新类型, 标准 SR 应该足够
    K > n_knobs → 系统有隐藏维度, 可能需要扩展 DSL

  如果多个实验都显示 K > n_knobs:
    → 跨实验的共性 → 这可能是一个通用物理特征
    → 等价于"发现某些系统需要经典之外的描述维度"
```

---

## 整合到 ATLAS v4 的 Step C (Diagnose) 中

```python
# 更新后的 DSLDiagnoser
class DSLDiagnoser_v5:

    EXTENSIONS = {
        # ... (之前的所有扩展保持不变)

        "hidden_dimensions": {
            "trigger": "SciNet K > 关联 knob 数 (多个实验一致)",
            "extension": "允许 SR 搜索隐变量模型: y = f(x, z), z 是潜变量",
            "具体": "用 VAE/SciNet 学习潜变量, 然后对 (x, z) → y 做 SR",
            "通用性": "潜变量建模是通用统计方法",
            "与物理的关系": (
                "如果潜变量恰好是3维实向量且满足归一化 → Bloch球 = qubit态"
                "但系统不知道这一点 — 它只知道需要3个额外参数"
            )
        }
    }
```

---

## 这个方案最终能发现什么？

```
ATLAS v4+v5 的完整输出 (预期):

1. [Phase 1 输出] 各实验的符号公式
   "ENV_01: meter = C₁ · max(knob_0 - C₂, 0) · knob_1"
   "ENV_04: detector ∝ cos²(C₃ · position)"

2. [Phase 2 输出] 跨实验通用常数
   "C₁ 和 C₃ 和 ... 共享基本因子 UC₁ = 6.626e-34"

3. [Phase 3 输出] 框架发现
   a. "某些实验是概率性的" (方差检测)
   b. "某些系统有隐藏维度: K > n_knobs" (SciNet)
   c. "概率分布形状 = 确定性强度分布" (概念提取关联)
   d. "所有概率性现象共享同一个常数 UC₁" (PSLQ)

4. [综合] 以上四个发现组合 =
   "存在一个通用常数 UC₁,
    某些系统需要超经典的描述维度,
    这些系统的行为是概率性的,
    概率分布与确定性波动强度之间存在精确关联"

   → 这就是波粒二象性的操作性内容,
     不需要提及"波函数"、"希尔伯特空间"或"Born规则"

人类物理学家读到这个报告后可以说:
  "啊, 它发现了量子力学的经验基础"
```

---

## 当前框架的诚实评估

```
能发现的:
  ✓ E = hf (符号回归)
  ✓ h 是通用常数 (PSLQ)
  ✓ 某些系统是概率性的 (方差检测)
  ✓ 概率分布 = 波动强度 (概念关联)
  ✓ K > N-1 (SciNet)

不能发现的:
  ✗ 波函数 ψ 的概念 (需要复数, 系统没有复数 DSL)
  ✗ Born 规则的数学形式 P = |ψ|² (只能发现经验等价物 P ∝ I)
  ✗ 叠加原理 (需要理解态空间的线性结构)
  ✗ 不确定性原理 (需要理解对易关系)
  ✗ 纠缠 (需要复合系统实验)

诚实地说: ATLAS 发现的是波粒二象性的"操作性影子",
不是完整的量子力学框架. 但这已经远超所有现有工作.
```
