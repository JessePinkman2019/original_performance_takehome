# 优化进展记录

## 当前最佳成绩

| 版本 | Cycles | 对比 baseline |
|------|--------|--------------|
| 原始 baseline | 147,734 | 1.00x |
| VLIW 贪心打包 | 98,583 | 1.50x |
| SIMD 向量化 + multiply_add | 9,936 | 14.87x |
| 4-group 流水线 | 4,176 | 35.4x |
| 8-group octet 流水线 | 3,344 | 44.2x |
| 16-group hextet 流水线 + setup 打包 | 3,132 | 47.2x |
| 早期 round 特判 (mod 0/1) | 2,966 | 49.8x |
| 早期 round 特判 (mod 0/1/2) | 2,908 | 50.8x |
| hextet 跨边界 tail/head 重叠 | 2,668 | 55.4x |
| **addr 预取 + intra-hextet 流水线** | **2,638** | **56.0x** |

通过测试：所有 < 147734 及 < 18532 及 baseline_updated，但未通过 < 2164（需继续优化）

---

## Round 8: addr 预取 + intra-hextet 流水线 (2026-04-10)

- 候选 008（addr prefetch + intra-hextet pipelining）: **2,638 cycles** — ACCEPT，merged（commit 65d92ec）
- 改进：2,668 → 2,638 cycles（1.12% 降低，56.0x over baseline）

### 008 技术细节
- **addr 操作前移至 tail 开头**：在 emit_hextet_noload/arith4/normal 的 tail 函数中，将 next_s 的 addr ops（A/B/C/D 的 addr 计算）提前到 step 50 的 valu ops 之前。因为 addr 用的是 valu 引擎，但与当前 valu ops 在不同 group 上，等价于"预取"地址，让后续 load 能更早启动（约提前 12c）
- **intra_hextet_s（hextet0→hextet1 内联预取）**：在 hextet0 的 steps 39-49（load 引擎此时空闲，因为 O/P 在 hash 阶段），提前开始 hextet1 的 A/B/C loads（8+8+8=24 slots in 12 cycles）
- **intra_tail_next_s**：在 hextet0 的 tail 中进一步预取 hextet1 的 C[6:8] + D（使 hextet1 跳过 D 的前 6 个 loads）
- **d_preloaded=True**：hextet1 的 step 6 删掉 D 的重复 load，消除该步骤溢出

### Cycle 分解（实测，16 rounds）
| 类型 | rounds 数量 | 总 cycles |
|------|-------------|----------|
| Setup | — | 120c |
| mod0 rounds (0, 11) | 2 | ~254c（127c×2） |
| mod1 rounds (1, 12) | 2 | ~280c（140c×2） |
| mod2 rounds (2, 13) | 2 | ~320c（160c×2） |
| normal rounds (3-10, 14-15) | 10 | ~1570c（157c×10） |
| **合计** | | **2638c** |

### Round 8 引擎利用率（精确测量）
| 引擎 | 总 slots | 活跃 cycles | 利用率 | 平均 slots/active_cycle |
|------|---------|-------------|--------|------------------------|
| valu | 10,134 | 2,225 | 84.3% | 4.56/6 |
| load | 2,574 | 1,406 | 53.3% | 1.92/2 |

---

## Round 8 根本性方向分析：为什么 1,487c 目标在当前架构下无法实现

### 精确理论下界计算

**Valu 下界（真正的瓶颈）**：
- 主循环 valu ops 总数：10,134 slots
- valu 引擎上限：6 slots/cycle
- **Valu 理论最小：10,134 / 6 = 1,689c**
- 加上 setup（120c）：**总理论下界 = 1,809c**

**Load 下界（次要瓶颈）**：
- 10 normal rounds × 32 groups × 8 loads = 2,560 load_offset ops
- 加 special round loads（14 ops）= 2,574 total
- load 引擎上限：2 slots/cycle
- **Load 理论最小：1,287c**（已被 valu 超越）

**结论：在满载系统中 valu 是瓶颈（1,689c），而不是 load（1,287c）**

### 为什么无法在 1,487c 完成

1,487c 意味着主循环只有 1,367c。1,367c × 6 valu slots = 8,202 个 valu slot 预算。但我们有 10,134 个 valu ops 需要执行 → 差距 1,932 ops，数学上不可能。

**每组每轮的 valu ops 不可约简**：
- addr（idx + fvp_vec）：1 op
- xor（val ^= node）：1 op  
- hash（6 阶段 × 2 ops = 12 ops，已使用 multiply_add 优化）：12 ops
- idx_update（lsb, offset, new_idx, cmp, idx）：5 ops
- **最小值：19 ops/group/normal round**

任何低于 19 的数字都需要改变算法正确性（不允许）。

### 当前浪费的根源

**Special rounds 的 430 cycle 惩罚**：
- Rounds 0-2 (mod0/1/2) 占用 430 cycle，期间 load 引擎几乎空闲（<6 ops）
- 这 430c 内 load 引擎浪费了 424 × 2 = 848 个 load slots
- 但因为 valu 是瓶颈，即使填满这些 load slots 也无法超过 valu 的 1,689c 下界

**实际差距来源**（2,638c vs 1,809c 理论）：
- dep chain overhead：valu ops 实际需要 2,225c vs 理论 1,689c（+536c = +31.7% 开销）
- 297 个 valu-idle cycles（其中 225 个是 load-active）= 约 +200c 可恢复的开销

### Round 9 方向：跨 round 对角线流水线

**问题本质**：当前以 round 为单位串行（Round k 全部完成 → Round k+1 开始），导致 special rounds 的 valu 和 load 无法真正重叠。

**方案 A：完整 32×16 对角线展开（推荐）**
- 将所有 512 个 (group, round) 对按拓扑依赖顺序 emit
- 依赖：group g, round k+1 的 addr_compute 依赖 group g, round k 的 idx_update 写入
- 在稳态中：load 引擎处理 normal-round 组的 load_offset，valu 引擎处理 special-round 组的 hash+idx
- 理论上可将 special round 的 valu 隐藏在 normal round 的 load 时间内
- **预期结果：~1,850-2,100c（vs 2,638c，降幅 20-28%）**

**方案 B（保守）：跨 special-normal round 的局部重叠**
- 只在相邻 special→normal 转换处做跨 round overlap
- 实现复杂度低，但收益有限（~100-200c）

**推荐方案 A**，理由：
- Round 0-2（430c）和 Round 11-13（类似）各自在 load 引擎空闲的情况下完成
- 对角线展开后，这些 round 的 valu 可以填入 normal round 的 load 间隙
- 从 430c 的 special-round 惩罚降至近 0c（完全重叠）
- 目标：突破 2,164c 门槛（test_opus4_many_hours），向 1,790c 门槛（test_opus45_casual）进发

- 候选 007a（inter-hextet tail/head overlap）: **2,668 cycles** — ACCEPT，merged（commit cf65700）
- 改进：2,908 → 2,668 cycles（8.25% 降低，55.4x over 147,734 baseline）
- 通过门槛：< 2,763（= 2908 × 0.95）✓

### 007a 技术细节
- **hextet0 尾巴预取 hextet1 头部**：在 hextet0 的 steps 50-63（P 组 hash+idx 尾巴阶段，valu 引擎空闲 ~5 slots/cycle），同时启动 hextet1 的 A/B/C/D 组 addr+load（load 引擎此时空闲）
- **hextet1 → hextet0 跨 round 预取**：hextet1 尾巴同样预取下一 normal round 的 hextet0 头部（A/B/C/D.addr + A.loads[:8]）
- **mod2 特判 tail 预取**：mod2（arith4）hextet 的尾巴也预取下一 normal round 的头部
- 净效果：每轮从 2 个独立的 ~80c hextet 降至约 77c 的重叠结构，合 154c/round（理论）vs 160c 实测减为约 153c/2hextets（整体 160→？，精确见数据）

### Cycle 分解（实测，16 rounds）
| 类型 | rounds 数量 | 总 cycles |
|------|-------------|----------|
| Setup + round mod0(0) | — | 314c |
| mod0 round(11) | 1 | +160c |
| mod1 rounds | 2（round 1, 12）| +280c（140c×2） |
| mod2 rounds | 2（round 2, 13）| +314c（174+140） |
| normal rounds (mod3-10, mod3-4) | 10 | +1600c（160c×10） |
| **合计** | | **2668c** |

### Scratch 使用
- scratch_ptr = 1431 / 1536（**105 words 剩余**，≈ 13 VLEN=8 向量）

### Round 8 瓶颈分析

| 指标 | 数值 |
|------|------|
| Normal hextet 实际 | **80 cycles** |
| 理论最小（load bound） | **64 cycles**（128 load_offset / 2 slots） |
| 当前 overhead | 16 cycles/hextet |
| 正常 round 代价 | 160c / round（2 hextets） |
| 理论最小 round | 128c / round |
| 节省潜力（10 normal rounds） | ~320 cycles → projected ~2348 |

**到 1487 需要减少 1181 cycles**：
- 10 normal rounds 优化至理论下界 → -320c → 2348
- 2 mod2 rounds 优化 → 还有约 -60c → 2288
- 2 mod1 rounds → 已近优，-10c
- 仍需 -800c 以上：当前架构无法实现

---

## Round 6: 早期 round 特判 mod 0/1/2 (2026-04-10)

- 候选 006（mod2 special-casing + valu arith-select）: **2,908 cycles** — ACCEPT（small improvement），merged（commit a876a68）
- 改进：2,966 → 2,908 cycles（1.96% 降低，50.8x over baseline）
- 标记 `small_improvement`：改进 <5%，但方向正确，接受

### 006 技术细节
- **round_mod==2（rounds 2, 13）**：idx ∈ {3,4,5,6}（4 个节点），用 2 级算术 select 替代 8×load_offset：
  - `bit0 = idx & 1`：区分奇偶（低位对）
  - `bit1 = (idx - 3) >> 1`：区分高低对
  - `nv = node3 + bit0*(node4-node3)`：选 {3,4} 之一
  - `nv_hi = node5 + bit0*(node6-node5)`：选 {5,6} 之一
  - `nv = nv_lo + bit1*(nv_hi-nv_lo)`：最终选择
  - 8 valu ops/group vs 8 load_offset/group，节省 174 cycles（2 mod2 rounds）
- **round_mod==1 优化**：用 `multiply_add` 替代 `flow vselect`，消除 16c/hextet 串行化
  - `nv = node2 + lsb*(node1-node2) = multiply_add(lsb, diff, node2_bcast)`
  - 纯 valu，无 flow 引擎，可并行打包

### Cycle 分解（精确测量）
| 类型 | rounds 数量 | 总 cycles |
|------|-------------|----------|
| Setup（固定） | — | 188c |
| mod0 rounds | 2（round 0, 11）| 254c（127c×2） |
| mod1 rounds | 2（round 1, 12）| 280c（140c×2） |
| mod2 rounds | 2（round 2, 13）| 348c（174c×2） |
| normal rounds | 10（mod 3-10, 14-15）| 1840c（184c×10） |
| **合计** | | **2910c（测量 2908c）** |

### Round 7 瓶颈分析

Normal rounds 贡献 1840/2908 = **63.3%** 的 cycles，是最大目标。

| 指标 | 数值 |
|------|------|
| Normal hextet 实际 | 92 cycles |
| 理论最小（load bound） | 64 cycles（128 load_offset / 2 slots） |
| 当前 overhead | 28 cycles/hextet（43.8% 浪费） |
| 到达 1487 所需的 normal round 预算 | 42 cycles/round（不可能，低于 load 下限） |

关键约束：`1487` 目标要求每 normal round 仅 42 cycles，而 load 引擎物理最低 128 cycles/round（64×2 hextets）。**在不改变 load 架构的情况下，1487 无法仅靠 normal 轮优化达到。**

### Round 7 推荐路线

**Route B（最优）: 减少 hextet tail/head 浪费**
- 当前 overhead：28 cycles/hextet（P 组 hash+idx 尾巴约 5-9c 空洞 + head 启动浪费）
- 目标：92 → ~75 cycles/hextet（节省 17c × 2 hextets × 10 rounds = 340c）
- 预期结果：2908 - 340 = **~2570 cycles**

**Route A2（次优）: special-case mod3（rounds 3, 14，8 节点）**
- mod3 idx ∈ {7..14}，用 3 位分解：bit0, bit1, bit2 各一次 multiply_add
- 9 valu ops/group vs 8 load_offset/group，节省 ~90c（2 mod3 rounds × 45c）
- 预期结果：2908 - 90 = **~2820 cycles**

**Route B+A2（组合）**
- 预期：**~2480 cycles**
- 仍需 >1.6× 进一步降低以达 1487

**根本路线（非常困难）**：
- 不能预缓存所有 1023 个 forest 值（1023 + 791 min scratch = 1814 > 1536 limit）
- 需要某种根本性的 load 绕过机制，例如减少 batch_size 或者找到 idx 分布规律

---

## Round 5: 早期 round 特判 (mod 0/1) (2026-04-10)

- 候选 005（early-round special-casing mod 0+1）: **2,966 cycles** — ACCEPT，merged（commit 3e9d7d4）
- 改进：3,132 → 2,966 cycles（5.3% 降低，49.8x over baseline）

### 005 技术细节
- **round_mod==0（rounds 0, 11）**：所有 idx=0，单次 scalar load forest_values[0] + vbroadcast，跳过 32×8=256 次 load_offset，使用 `emit_hextet_noload` 
- **round_mod==1（rounds 1, 12）**：idx ∈ {1,2}，2 次 scalar load + vbroadcast，per-group flow vselect（使用 `addr_tmp_g[g]` 作为 per-group lsb_cond scratch，避免 WAW），使用 `emit_hextet_vselect2`
- **mod 2-10, 13-15**：沿用 16-group hextet pipeline
- Scratch 使用：1378/1536（158 words 剩余）

### Cycle 分解（精确测量）
| 类型 | rounds 数量 | 总 cycles |
|------|-------------|----------|
| Setup（固定） | — | 121c |
| mod0 rounds | 2（round 0, 11）| 319c（193+126） |
| mod1 rounds | 2（round 1, 12）| 318c（159×2） |
| normal rounds | 12（mod 2-10, 13-15）| 2208c（184c×12） |
| **合计** | | **2966c** |

### Round 6 瓶颈分析

正常 round（12 个）贡献 2208/2966 = **74.4%** 的 cycles，每 hextet 实际 92c vs 理论下限 64c（128 load_offset / 2 = 64c）。

关键发现：valu `multiply_add` 可以替代 `flow vselect`！
- `nv = node2 + (node1-node2)*lsb = multiply_add(node2_vec, diff_vec, lsb_vec)`
- 消除 16 个串行 flow vselect（每 hextet 16c 瓶颈）
- 同样原理可扩展到 mod2 的 4-node 选择（8 valu ops/group vs 8 load_offset/group，但 valu 吞吐 6x）

### Round 6 推荐路线

**首选：Option C + A（valu 算术 vselect + mod2 特判）**
1. **mod1 优化**：用 `multiply_add` 替代 `flow vselect`，消除 16c/hextet 串行化，节省 ~64c（2 mod1 rounds × 2 hextets）
2. **mod2 特判（rounds 2, 13）**：`bit0=idx&1`, `bit1=(idx-3)>>1`, 2 次 multiply_add 选 4 节点，8 valu ops/group vs 8 load_offset/group，节省 ~108c（每 hextet 从 92c 降到 ~63c）
3. 总预期节省 ~170c → **目标 ~2796 cycles**

**备选：Option B（mod0 hextet 调度优化）**
- 理论 49c/hextet，当前 80c/hextet（avg），gap = 31c/hextet
- 如能达到 55c/hextet：节省 50c/round × 2 rounds = 100c
- 风险高，且不及 valu-select 路线收益明显

---

## Round 4: 16-group hextet 流水线 + setup 打包 (2026-04-10)

- 候选 004a（16-group hextet pipeline）: **3,132 cycles** — ACCEPT，merged（commit 6be33f1）
- 候选 004b（early-round special-casing）: REJECT — correctness failure
- 改进：3,344 → 3,132 cycles（6.3% 降低，47.2x over baseline）

### 004a 技术细节
- 每 hextet 处理 16 个 group（32 groups / 2 hextets = 2 hextets/round）
- 16-group 流水：A-D 启动(load+xor) → E-H 跟进 → I-P 在 E-H hash 阶段重叠 load → 共 63 步/hextet
- Setup 打包：15 个 const load 每 2 个打包（8 cycles），17 个 vbroadcast 每 6 个打包（3 cycles）
- Scratch 使用：1351/1536（185 words 剩余）
- 每 hextet 实际：3,132/32 = 97.9 cycles；每 group 平均 6.1 cycles

### 004b Bug 分析
- 策略：early-round special-casing（round mod 0/1/2 分别用 0/2/4 个 scalar load + vbroadcast/vselect）
- Bug 根因：`vsel4_nv_hi` 和 `vsel4_b_in_lo` 是**共享** scratch（所有 group 共用同一地址）
- 代码中注释承认了 Bug（第 958-963 行）："d_vsx4[2] and e_vsx4[2] both write vsel4_nv_hi!"
- 在 8-group 并行处理时，group B 写 `vsel4_nv_hi` 覆盖了 group A 尚未读取的值
- 修复方向：为每个 group 分配独立的 `vsel4_nv_hi_g[g]` 和 `vsel4_b_in_lo_g[g]`（需要 32×2×8=512 words）
- 但 512 words 超过剩余 scratch（185），需要减少 per-group 分配或改用 4-group 特判粒度

### Round 5 路线分析
- 当前瓶颈：97.9 cycles/hextet vs 目标 46.5 cycles/hextet（需要 2.1x）
- Normal rounds（mod 3-10，共 9/16 rounds）仍用 8-octet pipeline（每 octet ~49c），drag 整体
- **最优路线 A**：正常轮也用 16-group hextet（004a 当前已实现）+ fix early-round
- **最优路线 B**：在 004a 基础上 fix early-round（per-group scratch，scratch 勉强可行）
  - mod0 round: 1 scalar load + vbroadcast（无需 per-group scratch），节省大量 load cycles
  - mod1 round: 2 scalars + per-group vselect2（addr_tmp_g[g] 可复用），安全
  - mod2 round: 4 scalars + 需要 per-group nv_hi 和 b_in_lo，需 scratch 64 words，185 够用
- **估算**：mod0 2 rounds × ~50c 节省 = ~100c；mod1 2 rounds × ~30c = ~60c → 总 ~2870 cycles

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

## Round 3: 8-group 跨组流水线 (octet pipeline) (2026-04-10)

- 候选 003（8-group octet pipeline）: 3,344 cycles（改进 19.9%，44.2x over baseline）
- ACCEPT，merged（commit 0056419）
- 实际 cycles: 3344（Generator 报告一致）
- 主要分析（Evaluator 深度诊断）：
  - Main body 实际 3,139 cycles，setup 141 cycles，共 3,344
  - 每 octet 实际 49 cycles（期望 35），效率 71.4%
  - **核心浪费：1091 cycles（34.7%）load 引擎完全空闲**
  - load 活跃期间 valu 只用 41%（2.47/6 slots）
  - load 理论下限：4096 ops / 2 = 2048 cycles；valu 理论下限：9728 / 6 = 1621 cycles
  - **目标 1487 < load 下限 2048 → 必须减少 load 次数**
- 关键发现：idx 值**周期性重置**
  - Round 0 和 11：所有 256 个元素 idx=0（一个树节点）
  - Round 1,12：idx 只有 {1,2}（两个节点）
  - Round 2,13：idx 只有 {3,4,5,6}（四个节点）
  - Round 3,14：idx 只有 8 个节点
  - 这个结构可以大幅减少早期 round 的 load 次数
- 下一步建议（Round 4 三轨策略）：
  1. **Setup 打包优化（低风险）**：用 build() 替代 add() 批量发射，const 2 个/cycle，vbroadcast 6 个/cycle，setup 从 141 降至 ~50 cycles
  2. **16-group 流水线（中等难度，~900 cycle 节省）**：比 8-group 更宽，需 8 对 t1/t2/lsb/cmp（128 words extra），scratch = 1471 < 1536，目标 70 cycles/16-group 批次
  3. **早期 round 特判（高难度，可能触及目标）**：Round 0/11 用 1 次 scalar load + broadcast；Round 1/12 用 2 次 load + vselect；可节省 ~1100 load ops。但 valu 随之成为瓶颈（需要 vselect 级联）

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
