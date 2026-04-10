# 优化进展记录

## 当前最佳成绩

| 版本 | Cycles | 对比 baseline |
|------|--------|--------------|
| 原始 baseline | 147,734 | 1.00x |
| VLIW 贪心打包 | 98,583 | 1.50x |
| **SIMD 向量化 + multiply_add** | **9,936** | **14.87x** |

通过测试：`test_kernel_speedup` + `test_kernel_updated_starting_point` + `test_kernel_correctness`（3/9）

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

## Round 2: 4-group 跨组流水线 (2026-04-10)

- 候选 002a（3-group 流水线）: 5,056 cycles（改进 17.1%，通过 5791 阈值）
- 候选 002b（4-group 流水线）: 4,176 cycles（改进 31.5%，35.4x over baseline）
- 获胜者：002b，merged（commit c65462c）
- 主要瓶颈：flow 引擎（每 cycle 只能 1 条分支指令）限制 loop 吞吐；散射 load 仍占主导；hash 依赖链尚未完全被 ILP 隐藏
- 下一步建议：
  1. 进一步展开到 6-group 或 8-group pipeline，让 valu 槽完全饱和（目前 valu 利用率 ~50%）
  2. 将 scatter load 改为预计算地址 + 连续 vload，减少 load 引擎压力
  3. hash 阶段 multiply_add 链的中间结果尝试乱序排列，掩盖 2-cycle 延迟
  4. 目标下一轮 < 2164 cycles（test_opus4_many_hours 阈值）

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
