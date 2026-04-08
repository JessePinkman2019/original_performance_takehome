# CLAUDE.md

## 项目目标

优化 `perf_takehome.py` 中 `KernelBuilder.build_kernel` 方法，使其在虚拟机上运行的 clock cycle 数尽可能少。

当前 baseline：**147,734 cycles**。目标是尽可能低，争取低于 1,487 cycles（超越 Claude Opus 4.5 发布时的最佳成绩）。

## 禁止修改的文件

**绝对不能修改 `tests/` 文件夹下的任何文件。** 这是提交有效性的硬性要求。

- 不能改 `tests/submission_tests.py`
- 不能改 `tests/frozen_problem.py`（这是冻结版模拟器，专门防止通过修改模拟器作弊）
- 不能修改 `N_CORES`、`VLEN` 等常量（多核被故意禁用，启用它属于作弊）

可以自由修改的只有 `perf_takehome.py`。

## 验证方式

```bash
# 验证 tests/ 目录未被修改（结果必须为空）
git diff origin/main tests/

# 运行提交测试，查看通过哪些 cycle 阈值
python3 tests/submission_tests.py
```

不要用 `perf_takehome.py` 里的 `test_kernel_cycles` 作为最终验证，要用 `submission_tests.py`。

## 虚拟机架构要点

- **VLIW**：每条指令是一个 bundle，各引擎可并行执行多个 slot
- **引擎及并行上限**：`alu`×12、`valu`×6、`load`×2、`store`×2、`flow`×1
- **SIMD 向量宽度**：`VLEN = 8`
- **Scratch 空间**：1536 个 32-bit word，相当于寄存器/缓存
- **所有写操作在 cycle 末尾生效**，同一 cycle 内读的是旧值

## 主要优化方向

1. **VLIW 指令打包**：把多个独立 slot 合并进同一条指令，减少总 cycle 数
2. **向量化（SIMD）**：用 `valu`/`vload`/`vstore`/`vbroadcast` 一次处理 8 个元素
3. **减少内存往返**：把频繁访问的数据缓存到 scratch，避免重复 load
4. **循环展开**：减少跳转和循环控制的开销

## 常见作弊模式（不要做）

- 修改 `tests/` 降低测试标准
- 启用多核（`N_CORES > 1`）
- 修改 `frozen_problem.py` 改变模拟器行为
- 修改 `VLEN` 或其他约束常量
