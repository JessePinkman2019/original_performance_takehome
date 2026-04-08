# PERF_REPORT.md — Round 3 全量评估

## 评估完整性：✅ 全部 4 个 worktree 已完成 trace 分析

---

## Round 3 结果总览

| Agent | 策略 | Cycles | LOAD% (slots) | VALU% (slots) | ALU% (slots) | 通过 | 裁决 |
|-------|------|--------|---------------|---------------|--------------|------|------|
| **L** | I + 关键路径调度 + WAR放松 + 全阶段合并 + 并行idx | **1,391** | **95.5% (2,655)** | **89.0% (7,423)** | **89.6% (14,950)** | **8/9** | **MERGE** |
| K | I + 关键路径调度 + fused XOR | 1,503 | 88.6% (2,662) | 82.4% (7,422) | 82.9% (14,950) | 7/9 | 有价值 |
| I | Broadcast fix + ALU shift | 1,714 | 77.7% (2,662) | 72.8% (7,486) | 72.7% (14,950) | 5/9 | 基准 |
| J | Broadcast fix only | 1,787 | 74.5% (2,662) | 84.2% (9,022) | 12.4% (2,662) | 5/9 | 安全后备 |

---

## Agent L 详细分析（1,391 cycles — WINNER）

### 为什么 L 赢
1. **关键路径调度（reverse depth）**：与 K 相同的核心优化，减 ~200 cycles
2. **WAR bundle 放松**：VLIW 同 cycle 内 read-before-write 是安全的，去掉了不必要的 WAR 约束 → 每 cycle 塞更多 slot
3. **全阶段合并为单次 pack**：init broadcasts + vloads + body + stores 全部进同一个调度 DAG → 消除阶段间的空隙
4. **Setup 折叠进 body**：常量 load 利用 body 中 load 引擎空闲的 cycle
5. **并行 idx 更新**：`multiply_add(idx, two, one) || val&1` 并行执行，关键路径从 3 降到 2 VALU
6. **三引擎近乎满载**：LOAD 95.5%, VALU 89.0%, ALU 89.6% — 史上最均衡

### L 的理论下限分析
- LOAD floor: 2,655 / 2 = 1,328 cycles
- VALU floor: 7,423 / 6 = 1,237 cycles
- ALU floor: 14,950 / 12 = 1,246 cycles
- **硬瓶颈: LOAD (1,328)**
- 当前 overhead: 1,391 - 1,328 = **63 cycles (4.7%)** — 极低

---

## Agent K 分析（1,503 cycles）

### K 的独特贡献
- **Bottom-up height scheduling**（与 L 独立发现同一技术）
- **Fused XOR for level 0**：`val ^= fv0_vec` 直接合并（-5 cycles）
- K 的 overhead: 1,503 - 1,331 = 172 cycles (11.5%) — L 更紧凑

### K 比 L 差的原因
- 没做 WAR relaxation → 每 cycle 少塞 slot
- 没做全阶段合并 → init 和 body 之间有空隙
- 没做 setup 折叠 → load 引擎有空闲 cycle 未利用
- 没做并行 idx 更新 → 关键路径多 1 cycle/group/round

---

## Agent I 分析（1,714 cycles）

### I 的贡献（已被 K/L 吸收）
- First-pass broadcast fix（所有 agent 共享）
- ALU shift offload（所有 agent 共享）
- I 的 overhead: 1,714 - 1,331 = 383 cycles (22.3%) — 调度质量差

---

## Agent J 分析（1,787 cycles）

### J 的特殊数据点
- **没有 ALU shift offload**：ALU 只有 12.4%（2,662 slots），VALU 却有 84.2%（9,022 slots）
- 对比 I（有 ALU shift）：VALU 从 84.2% 降到 72.8%，ALU 从 12.4% 升到 72.7%
- **证实 ALU shift offload 显著降低 VALU 压力**

---

## 各 Agent 独特贡献（供融合）

| Agent | 独特技术 | 价值 | 是否已在 L 中 |
|-------|---------|------|-------------|
| L | WAR bundle 放松 | 高 | ✅ |
| L | 全阶段合并单次 pack | 高 | ✅ |
| L | Setup 折叠进 body | 中 | ✅ |
| L | 并行 idx 更新 | 中 | ✅ |
| K | Fused XOR for level 0 | 低（-5 cycles） | ❓ 需检查 L 是否有 |
| K/L | 关键路径调度（reverse depth） | 极高（-200） | ✅ |
| I/J | First-pass broadcast fix | 高（-24） | ✅ |
| I | ALU shift offload | 高（-73） | ✅ |

---

## 下一轮规划器输入

### 当前瓶颈
- **LOAD floor = 1,328 cycles**，当前 1,391，overhead 仅 63 cycles
- 三引擎都在 89%+ → 几乎没有空闲 slot 可利用
- 距下一个目标 test_opus45_improved_harness (< 1,363) 还差 **28 cycles**

### 优化方向
1. **进一步减少 loads**：缓存 tree level 3（rounds 3/14，省 512 loads）→ load floor 降至 ~1,072
2. **压缩 63 cycles overhead**：分析 trace 中 load 引擎空闲的具体 cycle 位置
3. **K 的 fused XOR 如果 L 没有**：可省 ~5 cycles
4. **并行度已接近极限**：三引擎 89%+ 意味着进一步改进必须减少总 op 数而非改善调度
