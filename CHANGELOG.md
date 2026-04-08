# 优化进展记录

> 新 session 启动时：读本文件 → 读 PERF_REPORT.md → 读 ROUND_PLAN.md → 继续优化

## 当前最佳成绩

| 版本 | Cycles | 加速比 | 通过测试 | Git Commit |
|------|--------|--------|---------|------------|
| 原始 baseline | 147,734 | 1.00x | 0/9 | origin/main |
| Phase 0: VLIW 贪心打包 | 98,583 | 1.50x | 2/9 | — |
| Phase 1: SIMD 向量化 | 9,936 | 14.87x | 3/9 | 1d8491e |
| Phase 2: 列表调度器 (Agent D) | 1,820 | 81.2x | 4/9 | e9c42a8 |
| Round 2 融合 (Agent F) | 1,811 | 81.6x | 4/9 | c6a63a5 |
| **Round 3 进行中 (Agent I)** | **1,714** | **86.2x** | **5/9** | 待 merge |

**当前目标：< 1,487 cycles（test_opus45_11hr）**
**下一个里程碑：< 1,579 (test_opus45_2hr)，当前差 135 cycles**

---

## Round 3: First-Pass Broadcast + ALU Shift（进行中 2026-04-08）

### 规划器发现
- **First-pass broadcast bug**: rounds 1/2 和 rounds 12/13 访问相同的 tree levels 1/2，但只有 12/13 做了 broadcast，1/2 仍在 scatter load
- **修复**: 用 effective_level 替代 round 编号判断（3 行改动）
- 修复后 LOAD floor 从 1,587 降至 ~1,331，**瓶颈从 LOAD 翻转为 VALU**

### 竞赛结果（I/J 已完成，K/L 待完成）

| Agent | 策略 | Cycles | vs F(1,811) | 状态 |
|-------|------|--------|-------------|------|
| **I** | Broadcast fix + ALU shift offload | **1,714** | **-97** | ✅ 完成 |
| J | Broadcast fix only（安全后备） | 1,787 | -24 | ✅ 完成 |
| K | I + Level 3 cache | ? | ? | 🔄 进行中 |
| L | I + Scheduler tuning | ? | ? | 🔄 进行中 |

### J 的数据点
- 只做 broadcast fix 省了 24 cycles（1,811 → 1,787）
- 规划器预期省 384 cycles，实际远少 → rounds 1/2 的 load 量比预期少（可能之前代码已部分处理）

### I 的数据点
- Broadcast fix + ALU shift 共省 97 cycles（1,811 → 1,714）
- ALU shift 单独贡献 ~73 cycles（1,787 → 1,714）
- 通过了 test_opus45_casual (< 1,790) ✅ 新里程碑

---

## Round 2: 融合竞赛 (2026-04-08)

### 竞赛结果

| Agent | 策略 | Cycles | Trace: LOAD% | VALU% | ALU% | 裁决 |
|-------|------|--------|-------------|-------|------|------|
| **F** | D + bug fixes + ALU addr | **1,811** | 87.7% (3,174) | 80.1% (8,702) | 14.6% (3,174) | **MERGED** |
| G | F + ALU hash shift | 1,816 | 未分析 | 未分析 | 未分析 | +5 cycles，微小差异 |
| E | 保守修 bug（过度重写） | 3,239 | 未分析 | 未分析 | 未分析 | **退步 +1,419** |
| H | F + level 3 cache（过度重写） | 2,669 | 未分析 | 未分析 | 未分析 | **退步 +849** |

### 关键教训
1. **targeted fix >> 全面重写**（E/H 退步的根本原因）
2. **ALU 地址计算有效**（F 的 3,174 ALU ops 替代 VALU）
3. **ALU hash shift 收益极小**（G 只比 F 快 5 cycles）
4. ⚠️ **流程违规**：E/G/H 未做 trace 分析就清理了 worktree

---

## Round 1: 列表调度器 4-Way 竞赛 (2026-04-08)

### 竞赛结果

| Agent | 策略 | Cycles | Trace: LOAD% | VALU% | ALU% | 独特贡献 |
|-------|------|--------|-------------|-------|------|---------|
| **D** | 列表调度器 + 树缓存 L0-2 + wrap消除 | **1,820** | 94.3% (3,429) | 82.3% (8,985) | 0.3% (71) | 树缓存、wrap消除 |
| C | 列表调度器 + 平衡调度 | 1,994 | 79.8% (3,180) | 80.0% (9,566) | 0.5% (124) | Load/VALU 平衡 |
| A | 列表调度器 + round-0 broadcast | 2,044 | 96.4% (3,937) | 79.5% (9,745) | 0.5% (128) | 跨round流水线 |
| B | 列表调度器 + ALU 地址计算 | 2,383 | 88.6% (4,221) | 64.6% (9,235) | 14.8% (4,224) | ALU 地址计算 |

### 关键教训
- 减少 scatter loads > 优化计算引擎（C 少 757 loads 就比 A 快）
- B 的 ALU 想法好但因没做 broadcast 导致多了 loads → 净退步

---

## Phase 1: SIMD 向量化 (2026-04-08)

9,936 cycles。32 组 × VLEN(8)，multiply_add hash，全 valu post-hash，散射加载。详见 git commit 1d8491e。

---

## Phase 0: VLIW 贪心打包

98,583 cycles。贪心 bin-packing in build()。RAW/WAW 冲突检测。详见初始代码。

---

## 累积失败教训（不要重复犯！）

1. **不要重写 build_kernel** → 做 targeted fix（Round 2 E/H 退步教训）
2. **减少 load > 优化 compute** → Load 是硬瓶颈（Round 1 B 教训）
3. **ALU hash shift 收益极小** → 不值得作为主要方向（Round 2 G 教训）
4. **Perf Agent 必须对所有 agent 做 trace 分析** → 不能只分析赢家（Round 2 流程违规教训）
5. **清理 worktree 前必须完成全量分析** → 否则丢失失败原因数据

## Harness 架构

```
Ralph Loop: while cycles > 1487
├── 规划器 → 读 PERF_REPORT.md + CHANGELOG.md → 写 ROUND_PLAN.md
├── 4 生成器 → 读 ROUND_PLAN.md → 各自 worktree → 写代码 + 跑测试
├── Perf Agent → 对全部 4 个做 trace 分析 → 写 PERF_REPORT.md
├── 融合 → 选最佳 merge + 提取所有 agent 的有价值技术
├── 🌙 做梦 → 每 3 轮清理 CHANGELOG + 更新 memory
└── Git → 每次 merge 一个 commit
```

## 验证命令

```bash
python3 tests/submission_tests.py    # 正式验证
git diff origin/main tests/          # 确认未作弊
```
