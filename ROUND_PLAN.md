# ROUND_PLAN.md — Round 2 融合计划

## 当前状态
- 最佳：1,820 cycles (Agent D)
- 目标：< 1,487 cycles

## 规划器发现的两个关键 Bug

### Bug 1: 第二遍遍历未命中缓存
Round 10 wrap 后所有 item 回到 idx=0，rounds 11-13 访问 levels 0-2（已缓存），但代码 `use_cache=(rnd < cache_levels)` 只检查 round 编号 < 3，漏掉了 11/12/13。浪费 768 scatter loads (~384 cycles)。
修复：`use_cache = get_read_level(rnd) < cache_levels`，其中 `get_read_level(rnd) = rnd if rnd <= 10 else rnd - 11`

### Bug 2: 不必要的 wrap check  
Round 11-15 的 idx 远小于 n_nodes=2047，不需要 wrap。只有 round 10 需要。浪费 320 ops。
修复：`needs_wrap=(rnd == 10)`

## 修复后瓶颈翻转
修复两个 bug 后 LOAD 从 3,429 降至 ~2,661，floor 从 1,714 降至 1,330。VALU 成为新瓶颈（floor ~1,467）。

## 四个融合方向

### Generator E: 保守修 bug（预期 1,480-1,540）
- 修 Bug 1 + Bug 2
- BATCH_SIZE_GROUPS 从 4 增到 8
- 无其他改动

### Generator F: E + ALU 地址计算（预期 1,430-1,500）
- E 的所有修复
- scatter 地址用 8 个 ALU 标量 ops 替代 1 个 VALU op（融合 Agent B）
- BATCH_SIZE_GROUPS = 16

### Generator G: F + ALU hash shift（预期 1,350-1,430）
- F 的所有改动
- hash 阶段 1/3/5 的 shift 操作用 8 个 ALU 替代 1 个 VALU
- BATCH_SIZE_GROUPS = 32

### Generator H: F + 树缓存 level 3 + 平衡调度（预期 1,380-1,450）
- F 的所有改动
- 树缓存扩展到 level 3（8 额外节点）
- Agent C 启发的动态平衡调度
