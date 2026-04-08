# ROUND_PLAN.md — Round 3

## 规划器读取的输入文件
- PERF_REPORT.md: Round 2 全量评估（F 有 trace，E/G/H 仅 cycle 数）
- CHANGELOG.md: 完整优化历史
- perf_takehome.py: 当前 F 代码
- problem.py: VM 语义

## 当前状态
- 1,811 cycles (Agent F)
- LOAD: 87.7% (3,174 slots), floor 1,587
- VALU: 80.1% (8,702 slots), floor 1,451
- ALU: 14.6% (3,174 slots), 85% 空闲
- 目标: < 1,487

## 规划器发现的关键 Bug

### First-Pass Broadcast 缺失
Round 1 所有 idx 在 {1,2}（level 1），Round 2 所有 idx 在 {3,4,5,6}（level 2）。
这和 Round 12/13（wrap 后的 level 1/2）完全相同，但当前代码只对 Round 12/13 做了 broadcast，Round 1/2 仍在做 scatter load。

**修复**：用 effective_level 替代 round 编号判断：
```python
effective_level = rnd if rnd <= forest_height else rnd - (forest_height + 1)
all_idx_zero = (effective_level == 0)
idx_in_1_2 = (effective_level == 1)
idx_in_3_6 = (effective_level == 2)
```

预期节省：512 loads（2 轮 × 256），零 scratch 开销。

## 修复后瓶颈翻转
修复后 LOAD 从 3,174 降至 ~2,662，floor 从 1,587 降至 1,331。
VALU 成为新瓶颈（floor ~1,504）。需要 ALU shift offload 压低 VALU。

---

## 四个生成器策略

### Generator I: First-Pass Broadcast + ALU Shift（推荐，预期 1,430-1,530）
- 修复 first-pass broadcast（3 行改动）
- ALU shift offload：hash 阶段 1/3/5 的 shift 用 8 个 ALU 标量 ops 替代 1 个 VALU op
- 风险：低-中。两个独立改动。

### Generator J: First-Pass Broadcast Only（安全后备，预期 1,580-1,650）
- 只修复 first-pass broadcast
- 不做其他改动
- 风险：低。验证 broadcast fix 正确性。

### Generator K: I + Level 3 Cache（预期 1,350-1,480）
- I 的所有改动
- 缓存 tree level 3（8 节点，rounds 3/14）
- 3 级 MUX 选择树
- 风险：中。MUX 代码复杂。

### Generator L: I + 调度器优化（预期 1,400-1,530）
- I 的所有改动
- 优化列表调度器的 load 优先级策略
- 减少 224 cycles 调度 overhead
- 风险：低-中。

## 关键规则
1. **所有 4 个必须包含 first-pass broadcast fix**
2. **SURGICAL EDITS ONLY**（Round 2 教训：E/H 重写退步）
3. **不要减小 BATCH_SIZE_GROUPS**
4. **改动前后对比 cycle 数**
