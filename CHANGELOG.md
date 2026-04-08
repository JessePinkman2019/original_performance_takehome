# 优化进展记录

## 当前最佳成绩

| 版本 | Cycles | 对比 baseline |
|------|--------|--------------|
| 原始 baseline | 147,734 | 1.00x |
| VLIW 贪心打包 | 98,583 | 1.50x |
| **SIMD 向量化 + multiply_add** | 9,936 | 14.87x |
| **列表调度器 + 树缓存 (Agent D)** | 1,820 | 81.2x |
| **Round 2 融合: F (ALU addr + bug fixes)** | **1,811** | **81.6x** |

通过测试：4/9（含 `test_opus4_many_hours` < 2,164）
下一个目标：`test_opus45_casual` < 1,790

---

## 迭代 1: SIMD 向量化 (2026-04-08)

### Worktree A: 完整 SIMD 重写 → 成功，MERGED
- **结果：9,936 cycles**（从 98,583 降低 89.9%）
- 技术手段：
  - 32 组 × VLEN(8) 向量化，idx/val 常驻 scratch（512 words）
  - Hash 阶段 0/2/4 用 multiply_add（4097/33/9 乘子）
  - Hash 阶段 1/3/5 用 3 个 valu（2 独立 + 1 依赖 = 2 cycles/stage）
  - Post-hash 全 valu（&1, +1, multiply_add, <, *），无 flow 引擎
  - 散射加载 8×load_offset + valu 地址计算
  - 17 个常量向量 vbroadcast 一次性初始化
  - 初始化：vload 64 次（32 组 × idx + val），结束：vstore 64 次
- Perf Agent 验证：正确性通过，tests/ 未修改

### Worktree B: 未启动（Worktree A 已达标）

### Perf Agent 诊断
- 估计 ~18 cycles/group：1(addr) + 4(scatter) + 1(xor) + 9(hash chain) + 3(post-hash)
- **瓶颈预判：散射加载（4 cycles/组，占 22%）+ hash 依赖链（9 cycles/组，占 50%）**
- 下一步建议：列表调度器实现跨组 ILP，让 load 和 valu 引擎重叠

---

## 迭代 2-3: 列表调度器 4-way 竞赛 (2026-04-08)

### 竞赛结果

| Agent | 策略 | Cycles | LOAD% | VALU% | ALU% | 裁决 |
|-------|------|--------|-------|-------|------|------|
| **D** | 列表调度器 + 树缓存 levels 0-2 + wrap 消除 | **1,820** | 94.3% | 82.3% | 0.3% | **MERGED** |
| C | 列表调度器 + 批量交错 + 平衡调度 | 1,994 | 79.8% | 80.0% | 0.5% | 有价值技术 |
| A | 列表调度器 + round-0 broadcast + 跨 round 流水线 | 2,044 | 96.4% | 79.5% | 0.5% | 有价值技术 |
| B | 列表调度器 + ALU 地址计算 | 2,383 | 88.6% | 64.6% | 14.8% | 有价值技术 |

### 各 Agent 独特贡献（供融合）

- **D（赢家）**：树层级缓存 levels 0-2（7 节点 scratch），6 轮免 scatter load；早期 round wrap check 消除
- **A**：跨 round 流水线（round N compute tail 重叠 round N+1 load start）；store-compute 重叠
- **B**：ALU 做 scatter 地址计算（4,096 ALU slots），释放 510 个 VALU slots
- **C**：Load/VALU 近乎完美平衡（79.8%/80.0%），说明减 load 比优化 VALU 更有效

### Perf Agent Trace 诊断

- **主瓶颈：Load 引擎 94.3%**，连续满载 1,624 cycles
- 理论下限：3,429 loads / 2 per cycle = 1,714 cycles（当前 overhead 仅 6.2%）
- **ALU 99.5% 空闲**——12 slots/cycle 未利用
- Scratch 余量：1,331/1,536（205 words 可用于扩展缓存）

### 失败/不足记录

- B 的 ALU 策略虽好但因没做 broadcast 导致多了 284 loads，净效果为负
- 教训：减少 load 总量比转移计算引擎更优先（Load 是硬瓶颈）

### 下一步融合方向

在 D 基础上融合 A/B/C 技术：
1. 扩展树缓存 levels 3-4（+24 节点 = 31 words）
2. 融合 A 的跨 round 流水线
3. 融合 B 的 ALU 地址计算
4. 融合 C 的平衡调度策略

---

## 迭代 4-5: Round 2 融合竞赛 (2026-04-08)

### 规划器发现的两个 Bug（在 Round 1 冠军 D 中）
1. **Second-pass cache miss**: rounds 11-13 访问 levels 0-2 但未命中缓存。修复：检查 effective level
2. **不必要 wrap check**: rounds 11-15 不需要 wrap。修复：只在 round == forest_height 时 wrap

### 竞赛结果

| Agent | 策略 | Cycles | vs D(1,820) | 裁决 |
|-------|------|--------|-------------|------|
| **F** | Bug fixes + ALU addr + rounds 0/11-13 broadcast | **1,811** | **-9** | **MERGED** |
| G | F + ALU hash shift | 1,816 | -4 | 有价值 |
| E | 保守修 bug（重写过多丢失 D 优化） | 3,239 | +1,419 | **退步** |
| H | F + 树缓存 level 3（重写过多丢失 D 优化） | 2,669 | +849 | **退步** |

### Trace 对比: F vs D

| 引擎 | D (1,820) | F (1,811) | 变化 |
|------|----------|----------|------|
| LOAD | 94.3% (3,429) | 87.7% (3,174) | **-255 loads** |
| VALU | 82.3% (8,985) | 80.1% (8,702) | -283 VALU ops |
| ALU | 0.3% (71) | **14.6% (3,174)** | **+3,103 ALU ops（地址计算移到 ALU）** |

### 失败分析（关键教训）
- **E 和 H 退步严重**：这两个 agent 在应用 bug fix 时实质上重写了大部分代码，丢失了 D 的列表调度器质量和跨 round 优化
- **教训：targeted fix >> 全面重写。** 应该在 D 的代码上做最小改动，不是重写 build_kernel
- **F 和 G 成功的原因**：它们保持了 D 的核心代码结构，只做增量修改

### 融合价值提取
- **F 的 ALU 地址计算已验证有效**：3,174 ALU ops 替代 VALU，释放了 283 VALU slots
- **G 的 ALU hash shift 效果不明显**：只比 F 多 5 cycles（1,816 vs 1,811），可能是调度 overhead 抵消了收益
- **下一轮方向**：需要更大幅度的 load 减少（当前 3,174 loads → floor 1,587 cycles），或突破性的算法改进

---

## 🌙 做梦记录 #1 (2026-04-08)

### Stage A: 失败模式扫描
- 无反复失败模式（首轮竞赛）
- B 的失败教训已记录：减 load > 转移引擎

### Stage B: 方向评估
- 优化路径正确：147K → 98K → 9.9K → 1,820
- Load 瓶颈明确（94.3%），ALU 空闲（99.5%）
- 下一轮应融合而非继续独立探索

### Memory 更新
- project_optimization_state.md: 更新至 1,820 cycles
- 新建 insight_engine_bottleneck_analysis.md

---

## 已完成优化：VLIW 贪心打包

### 做了什么

替换了 `KernelBuilder.build()` 方法。原来每个 slot 单独一条指令，现在用贪心算法把独立的 slot 打包进同一条 VLIW bundle。

### 冲突检测规则

一个 slot 与当前 bundle 冲突（触发 flush）的条件：
1. **RAW**：该 slot 的 reads 与 bundle 已有的 writes 有交集
2. **WAW**：该 slot 的 writes 与 bundle 已有的 writes 有交集
3. **slot 上限**：该 engine 在当前 bundle 里已达到 `SLOT_LIMITS[engine]`
4. **flow 引擎**：只允许 1 个 slot，已有则强制 flush

WAR（写后读）不构成冲突，因为 VLIW 语义是所有 slot 先读旧值、cycle 末尾统一写入。

### debug slot 的处理

`debug` 类型的 slot 不计入 engine slot 上限，也不计入读写依赖分析，直接附加到当前 bundle（`bundle.setdefault("debug", []).append(slot)`）。

### 关键教训

- `submission_tests.py` 里 `enable_debug=False`，debug compare 不执行，正确性只靠最终内存比对。
- 调试时用 `enable_debug=False` + 手动对比最终 `inp_values_p` 区域更可靠，不要依赖 debug compare 验证中间步骤。
- `reference_kernel2` 是 generator，必须用 `for ref_mem in reference_kernel2(mem, value_trace)` 驱动，不能单独调用 `machine.run()`。

---

## 下一步优化方向

### 1. SIMD 向量化（优先级：高）

当前每个 batch item 串行处理（256 个 item × 16 rounds）。`VLEN=8`，可以把 8 个 item 打包成一个向量一起处理：
- `vload` 一次加载 8 个 `inp_values`
- `valu` 向量化 hash 运算（注意 hash 里有 `%` 和条件分支，需要特殊处理）
- `vstore` 一次写回 8 个结果
- 理论上限：再降约 8x，目标 ~12,000 cycles

难点：`myhash` 里没有向量化的 `%`，需要用位运算替代（`val % 2` 等价于 `val & 1`）；条件分支（选左/右子节点）需要用 `vselect` 替代。

### 2. 更激进的指令重排（优先级：中）

当前贪心打包是线性扫描，不做跨 item 的重排。相邻两个 batch item 的部分操作互不依赖，可以交错排列进一步填满 ALU 槽位（alu 上限是 12，目前很少用满）。

### 3. scratch 缓存优化（优先级：低）

`forest_values`（树节点）被反复从 mem load，如果访问模式有局部性可以预取到 scratch 缓存。但 scratch 只有 1536 word，树有 2047 个节点，装不下全部，需要分析实际访问分布。

---

---

## Harness 启动记录 (2026-04-08)

### 当前状态
- **98,583 cycles**（1.50x speedup）
- 通过测试：`test_kernel_correctness` + `test_kernel_speedup`（2/9）
- `git diff origin/main tests/` = 空（未作弊）

### Harness 架构
- 主控循环（Ralph Loop）：`while cycle > 1487: optimize()`
- 四角色 Agent：1 规划器 + 2 生成器（worktree 并行）+ 1 Perf 评估器
- 文件总线通信：ROUND_PLAN.md → OPTIMIZATION_PROPOSAL.md → PERF_REPORT.md
- 做梦机制：每 3 轮迭代触发 6 阶段（战术 CHANGELOG 清理 + 战略 memory 整理）

### Phase 1 目标
- SIMD 向量化 + multiply_add hash 优化
- 预期：~10,000 cycles

---

## 验证命令

```bash
# 正式验证（用这个，不要用 perf_takehome.py 里的测试）
python3 tests/submission_tests.py

# 确认 tests/ 未被修改
git diff origin/main tests/
```
