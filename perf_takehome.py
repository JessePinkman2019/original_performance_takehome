"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def _get_scratch_range(self, slot):
        """Return (set_of_read_addrs, set_of_write_addrs) for a slot on a given engine."""
        engine, args = slot
        if engine == "debug":
            return set(), set()

        if engine == "alu":
            op, dest, a1, a2 = args
            return {a1, a2}, {dest}

        if engine == "valu":
            if args[0] == "vbroadcast":
                _, dest, src = args
                return {src}, set(range(dest, dest + VLEN))
            elif args[0] == "multiply_add":
                _, dest, a, b, c = args
                reads = set(range(a, a + VLEN)) | set(range(b, b + VLEN)) | set(range(c, c + VLEN))
                writes = set(range(dest, dest + VLEN))
                return reads, writes
            else:
                op, dest, a1, a2 = args
                reads = set(range(a1, a1 + VLEN)) | set(range(a2, a2 + VLEN))
                writes = set(range(dest, dest + VLEN))
                return reads, writes

        if engine == "load":
            if args[0] == "load":
                _, dest, addr = args
                return {addr}, {dest}
            elif args[0] == "load_offset":
                _, dest, addr, offset = args
                return {addr + offset}, {dest + offset}
            elif args[0] == "vload":
                _, dest, addr = args
                return {addr}, set(range(dest, dest + VLEN))
            elif args[0] == "const":
                _, dest, val = args
                return set(), {dest}

        if engine == "store":
            if args[0] == "store":
                _, addr, src = args
                return {addr, src}, set()
            elif args[0] == "vstore":
                _, addr, src = args
                return {addr} | set(range(src, src + VLEN)), set()

        if engine == "flow":
            if args[0] == "select":
                _, dest, cond, a, b = args
                return {cond, a, b}, {dest}
            elif args[0] == "vselect":
                _, dest, cond, a, b = args
                reads = set(range(cond, cond + VLEN)) | set(range(a, a + VLEN)) | set(range(b, b + VLEN))
                writes = set(range(dest, dest + VLEN))
                return reads, writes
            elif args[0] in ("pause", "halt"):
                return set(), set()
            elif args[0] == "cond_jump":
                _, cond, addr = args
                return {cond}, set()
            elif args[0] == "jump":
                return set(), set()
            elif args[0] == "add_imm":
                _, dest, a, imm = args
                return {a}, {dest}

        return set(), set()

    def build(self, slots, vliw=False):
        """Greedy VLIW list scheduler: pack ops into cycles respecting data dependencies and slot limits.

        Key invariant: an op can only be scheduled once ALL earlier ops (by list position) that
        write to addresses this op reads have been scheduled in a PRIOR cycle.
        """
        if not slots:
            return []

        instrs = []
        n = len(slots)
        scheduled = [False] * n

        # For each op, compute reads and writes
        op_reads = []
        op_writes = []
        op_engines = []
        for i, (engine, args) in enumerate(slots):
            r, w = self._get_scratch_range((engine, args))
            op_reads.append(r)
            op_writes.append(w)
            op_engines.append(engine)

        # Build dependency graph
        # For each scratch address, track the last op that writes/reads it
        last_writer = {}  # addr -> op index
        last_reader = {}  # addr -> op index
        dep_count = [0] * n  # number of unresolved dependencies for each op
        dependents = [[] for _ in range(n)]  # ops that depend on op i

        for i in range(n):
            deps_for_i = set()
            # RAW: check if any earlier op writes to addresses we read
            for addr in op_reads[i]:
                if addr in last_writer:
                    deps_for_i.add(last_writer[addr])
            # WAW: check if any earlier op writes to addresses we write
            for addr in op_writes[i]:
                if addr in last_writer:
                    deps_for_i.add(last_writer[addr])
            # WAR: if we write to addr X and some earlier op reads X,
            # we must not execute before that reader completes.
            # This prevents reordering a write before a read of the same address.
            for addr in op_writes[i]:
                if addr in last_reader:
                    deps_for_i.add(last_reader[addr])

            dep_count[i] = len(deps_for_i)
            for dep_idx in deps_for_i:
                dependents[dep_idx].append(i)

            # Update last_writer and last_reader
            for addr in op_writes[i]:
                last_writer[addr] = i
            for addr in op_reads[i]:
                last_reader[addr] = i

        # Ready set: ops with no unresolved dependencies
        ready = set()
        for i in range(n):
            if dep_count[i] == 0 and op_engines[i] != "debug":
                ready.add(i)
            elif op_engines[i] == "debug":
                scheduled[i] = True  # debug ops don't count

        remaining = sum(1 for i in range(n) if op_engines[i] != "debug")

        while remaining > 0:
            cycle_bundle = {}
            cycle_engine_count = defaultdict(int)
            cycle_writes = set()
            just_scheduled = []

            # Try to pack ready ops into this cycle
            # Sort ready set by original order for determinism
            for i in sorted(ready):
                engine = op_engines[i]
                if cycle_engine_count[engine] >= SLOT_LIMITS[engine]:
                    continue

                writes_i = op_writes[i]
                reads_i = op_reads[i]

                # WAW within cycle: can't write same addr as another op this cycle
                if writes_i & cycle_writes:
                    continue

                # RAW within cycle: can't read addr written this cycle
                # (writes take effect at end of cycle, reads see old values)
                if reads_i & cycle_writes:
                    continue

                # Schedule this op
                if engine not in cycle_bundle:
                    cycle_bundle[engine] = []
                cycle_bundle[engine].append(slots[i][1])
                cycle_engine_count[engine] += 1
                cycle_writes |= writes_i
                scheduled[i] = True
                remaining -= 1
                just_scheduled.append(i)

            # Remove scheduled ops from ready set and update dependents
            for i in just_scheduled:
                ready.discard(i)
                for j in dependents[i]:
                    if op_engines[j] == "debug":
                        continue
                    dep_count[j] -= 1
                    if dep_count[j] == 0:
                        ready.add(j)

            if not just_scheduled:
                # Should not happen if dependency graph is correct
                # Force schedule first remaining op
                for i in range(n):
                    if not scheduled[i] and op_engines[i] != "debug":
                        engine = op_engines[i]
                        cycle_bundle = {engine: [slots[i][1]]}
                        scheduled[i] = True
                        remaining -= 1
                        for j in dependents[i]:
                            if op_engines[j] != "debug":
                                dep_count[j] -= 1
                                if dep_count[j] == 0:
                                    ready.add(j)
                        break

            instrs.append(cycle_bundle)

        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized VLIW SIMD kernel with software pipelining.
        Uses vectorized operations (VLEN=8), multiply_add for hash stages,
        and interleaved ops across groups for maximum VLIW utilization.
        """
        V = VLEN  # 8
        BATCH = 4  # groups processed together

        # --- Scalar init ---
        tmp1 = self.alloc_scratch("tmp1")
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        self.add("flow", ("pause",))

        # --- Allocate vector scratch ---
        n_groups = batch_size // V  # 32

        # Persistent idx and val vectors for each group (across rounds)
        idx_vecs = [self.alloc_scratch(f"idx_v{g}", V) for g in range(n_groups)]
        val_vecs = [self.alloc_scratch(f"val_v{g}", V) for g in range(n_groups)]

        # Per-batch-slot temporary vectors
        # Use 2 sets (ping-pong) for software pipelining: load set B while computing set A
        nv_tmp = [[self.alloc_scratch(f"nv_{s}_{j}", V) for j in range(BATCH)] for s in range(2)]
        addr_tmp = [[self.alloc_scratch(f"ad_{s}_{j}", V) for j in range(BATCH)] for s in range(2)]
        ht1_tmp = [[self.alloc_scratch(f"h1_{s}_{j}", V) for j in range(BATCH)] for s in range(2)]
        ht2_tmp = [[self.alloc_scratch(f"h2_{s}_{j}", V) for j in range(BATCH)] for s in range(2)]
        branch_tmp = [[self.alloc_scratch(f"br_{s}_{j}", V) for j in range(BATCH)] for s in range(2)]

        # Vector constants
        fvp_vec = self.alloc_scratch("fvp_vec", V)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", V)
        one_vec = self.alloc_scratch("one_vec", V)
        two_vec = self.alloc_scratch("two_vec", V)
        zero_vec = self.alloc_scratch("zero_vec", V)

        # Hash constant vectors (6 stages, 2 constants each)
        hc = []
        for hi in range(len(HASH_STAGES)):
            hc.append(self.alloc_scratch(f"hc{hi}_v1", V))
            hc.append(self.alloc_scratch(f"hc{hi}_v3", V))

        # multiply_add constants for stages 0, 2, 4
        mult_4097_vec = self.alloc_scratch("m4097v", V)
        mult_33_vec = self.alloc_scratch("m33v", V)
        mult_9_vec = self.alloc_scratch("m9v", V)
        # Shift constants for stages 1, 3, 5
        shift_19_vec = self.alloc_scratch("s19v", V)
        shift_9_vec = self.alloc_scratch("s9v", V)
        shift_16_vec = self.alloc_scratch("s16v", V)

        # --- Init phase: broadcast constants, load initial data ---
        init_body = []

        # Broadcast scalar memory values to vectors
        init_body.append(("valu", ("vbroadcast", fvp_vec, self.scratch["forest_values_p"])))
        init_body.append(("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"])))

        one_s = self.alloc_scratch("one_s")
        two_s = self.alloc_scratch("two_s")
        zero_s = self.alloc_scratch("zero_s")
        init_body.append(("load", ("const", zero_s, 0)))
        init_body.append(("load", ("const", one_s, 1)))
        init_body.append(("load", ("const", two_s, 2)))
        init_body.append(("valu", ("vbroadcast", one_vec, one_s)))
        init_body.append(("valu", ("vbroadcast", two_vec, two_s)))
        init_body.append(("valu", ("vbroadcast", zero_vec, zero_s)))

        # Hash constants
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            s1 = self.alloc_scratch(f"hcs{hi}_1")
            init_body.append(("load", ("const", s1, val1)))
            init_body.append(("valu", ("vbroadcast", hc[hi*2], s1)))
            s2 = self.alloc_scratch(f"hcs{hi}_2")
            init_body.append(("load", ("const", s2, val3)))
            init_body.append(("valu", ("vbroadcast", hc[hi*2+1], s2)))

        # Multiply and shift constants
        for val, vec in [(4097, mult_4097_vec), (33, mult_33_vec), (9, mult_9_vec),
                         (19, shift_19_vec), (9, shift_9_vec), (16, shift_16_vec)]:
            s = self.alloc_scratch()
            init_body.append(("load", ("const", s, val)))
            init_body.append(("valu", ("vbroadcast", vec, s)))

        # Load initial idx and val vectors from memory using vload
        idx_addr_s = self.alloc_scratch("idx_addr_s")
        val_addr_s = self.alloc_scratch("val_addr_s")
        eight_s_init = self.alloc_scratch("eight_s_init")

        init_body.append(("load", ("const", eight_s_init, V)))
        # Start addresses at the base pointers
        init_body.append(("flow", ("add_imm", idx_addr_s, self.scratch["inp_indices_p"], 0)))
        init_body.append(("flow", ("add_imm", val_addr_s, self.scratch["inp_values_p"], 0)))

        for g in range(n_groups):
            init_body.append(("load", ("vload", idx_vecs[g], idx_addr_s)))
            init_body.append(("load", ("vload", val_vecs[g], val_addr_s)))
            if g < n_groups - 1:
                init_body.append(("alu", ("+", idx_addr_s, idx_addr_s, eight_s_init)))
                init_body.append(("alu", ("+", val_addr_s, val_addr_s, eight_s_init)))

        # (init_body will be finalized after level allocations below)

        # --- Main computation ---
        # For each group, the computation is:
        #   addr = fvp + idx
        #   nv = scatter_load(mem, addr)  [8 loads]
        #   val ^= nv
        #   val = hash(val)  [12 valu ops: 3 multiply_add + 3*(2 independent + 1 combine)]
        #   bit = val & 1; branch = bit + 1
        #   idx = 2*idx + branch
        #   mask = idx < n_nodes; idx *= mask
        #
        # Hash critical path per group: 12 sequential valu ops
        # With 6 valu slots per cycle, we can overlap 6 groups' hash ops
        # But we're limited to 2 load slots and 4 batch temps
        #
        # Strategy: process BATCH=4 groups together, interleaving their ops

        def gen_load_ops(j, g, s):
            """Generate addr computation and scatter loads for group g
            using ping-pong set s's nv_tmp[j] and addr_tmp[j]."""
            idx = idx_vecs[g]
            nv = nv_tmp[s][j]
            addr = addr_tmp[s][j]

            ops = []
            ops.append(("valu", ("+", addr, idx, fvp_vec)))
            for lane in range(V):
                ops.append(("load", ("load_offset", nv, addr, lane)))
            return ops

        def gen_compute_ops(j, g, s):
            """Generate XOR + hash + post-hash for group g using set s's nv_tmp[j]."""
            v = val_vecs[g]
            idx = idx_vecs[g]
            h1 = ht1_tmp[s][j]
            h2 = ht2_tmp[s][j]
            br = branch_tmp[s][j]
            nv = nv_tmp[s][j]

            ops = []

            # XOR
            ops.append(("valu", ("^", v, v, nv)))

            # Hash stage 0: multiply_add -> val = val * 4097 + hc0
            ops.append(("valu", ("multiply_add", v, v, mult_4097_vec, hc[0])))

            # Hash stage 1: t1 = val ^ hc1; t2 = val >> 19; val = t1 ^ t2
            ops.append(("valu", ("^", h1, v, hc[2])))
            ops.append(("valu", (">>", h2, v, shift_19_vec)))
            ops.append(("valu", ("^", v, h1, h2)))

            # Hash stage 2: multiply_add -> val = val * 33 + hc2
            ops.append(("valu", ("multiply_add", v, v, mult_33_vec, hc[4])))

            # Hash stage 3: t1 = val + hc3; t2 = val << 9; val = t1 ^ t2
            ops.append(("valu", ("+", h1, v, hc[6])))
            ops.append(("valu", ("<<", h2, v, shift_9_vec)))
            ops.append(("valu", ("^", v, h1, h2)))

            # Hash stage 4: multiply_add -> val = val * 9 + hc4
            ops.append(("valu", ("multiply_add", v, v, mult_9_vec, hc[8])))

            # Hash stage 5: t1 = val ^ hc5; t2 = val >> 16; val = t1 ^ t2
            ops.append(("valu", ("^", h1, v, hc[10])))
            ops.append(("valu", (">>", h2, v, shift_16_vec)))
            ops.append(("valu", ("^", v, h1, h2)))

            # Post-hash
            ops.append(("valu", ("&", h1, v, one_vec)))
            ops.append(("valu", ("+", br, h1, one_vec)))
            ops.append(("valu", ("multiply_add", idx, idx, two_vec, br)))
            ops.append(("valu", ("<", h1, idx, n_nodes_vec)))
            ops.append(("valu", ("*", idx, idx, h1)))

            return ops

        def gen_compute_only_ops(j, g, nv_src, s=0):
            """Generate XOR + hash + post-hash for group g using a pre-loaded nv vector."""
            v = val_vecs[g]
            idx = idx_vecs[g]
            h1 = ht1_tmp[s][j]
            h2 = ht2_tmp[s][j]
            br = branch_tmp[s][j]

            ops = []
            # XOR with pre-loaded node values
            ops.append(("valu", ("^", v, v, nv_src)))
            # Hash
            ops.append(("valu", ("multiply_add", v, v, mult_4097_vec, hc[0])))
            ops.append(("valu", ("^", h1, v, hc[2])))
            ops.append(("valu", (">>", h2, v, shift_19_vec)))
            ops.append(("valu", ("^", v, h1, h2)))
            ops.append(("valu", ("multiply_add", v, v, mult_33_vec, hc[4])))
            ops.append(("valu", ("+", h1, v, hc[6])))
            ops.append(("valu", ("<<", h2, v, shift_9_vec)))
            ops.append(("valu", ("^", v, h1, h2)))
            ops.append(("valu", ("multiply_add", v, v, mult_9_vec, hc[8])))
            ops.append(("valu", ("^", h1, v, hc[10])))
            ops.append(("valu", (">>", h2, v, shift_16_vec)))
            ops.append(("valu", ("^", v, h1, h2)))
            # Post-hash
            ops.append(("valu", ("&", h1, v, one_vec)))
            ops.append(("valu", ("+", br, h1, one_vec)))
            ops.append(("valu", ("multiply_add", idx, idx, two_vec, br)))
            ops.append(("valu", ("<", h1, idx, n_nodes_vec)))
            ops.append(("flow", ("vselect", idx, h1, idx, zero_vec)))
            return ops

        # Pre-allocate scratch for level-optimized rounds
        nv_broadcast = self.alloc_scratch("nv_bcast", V)
        tv1_s = self.alloc_scratch("tv1_s")
        tv2_s = self.alloc_scratch("tv2_s")
        tv1_vec = self.alloc_scratch("tv1v", V)
        tv2_vec = self.alloc_scratch("tv2v", V)

        # Level 2: 4 tree values at indices 3-6
        tv_l2 = [self.alloc_scratch(f"tv2_{k}", V) for k in range(4)]
        tv_l2_s = [self.alloc_scratch(f"ts2_{k}") for k in range(4)]
        four_vec = self.alloc_scratch("four_v", V)
        pair_lo = self.alloc_scratch("pair_lo", V)
        pair_hi = self.alloc_scratch("pair_hi", V)
        # Need four_s for broadcast
        four_s = self.alloc_scratch("four_s")
        init_body.append(("load", ("const", four_s, 4)))
        init_body.append(("valu", ("vbroadcast", four_vec, four_s)))

        self.instrs.extend(self.build(init_body))

        # Build the main body with software-pipelined loads/compute
        body = []

        # Determine the tree level for each round
        # Rounds 0-9: level 0-9 (first descent)
        # Round 10: level 10, all wrap to 0 after
        # Rounds 11-15: level 0-4 (second descent)
        def get_level(rnd):
            if rnd <= 10:
                return rnd
            else:
                return rnd - 11

        for rnd in range(rounds):
            level = get_level(rnd)
            n_batches = n_groups // BATCH

            if level == 0:
                # Level 0: all idx = 0, one tree value
                # Load tree_values[0] = mem[forest_values_p + 0]
                body.append(("load", ("load", idx_addr_s, self.scratch["forest_values_p"])))
                body.append(("valu", ("vbroadcast", nv_broadcast, idx_addr_s)))

                # Compute for all groups using the broadcast value
                for batch_idx in range(n_batches):
                    bg = batch_idx * BATCH
                    all_compute = []
                    for j in range(BATCH):
                        all_compute.append(gen_compute_only_ops(j, bg + j, nv_broadcast))

                    max_ops = max(len(ops) for ops in all_compute)
                    for op_idx in range(max_ops):
                        for j in range(BATCH):
                            if op_idx < len(all_compute[j]):
                                body.append(all_compute[j][op_idx])

            elif level == 1:
                # Level 1: idx is either 1 or 2
                body.append(("flow", ("add_imm", tv1_s, self.scratch["forest_values_p"], 1)))
                body.append(("load", ("load", tv1_s, tv1_s)))
                body.append(("flow", ("add_imm", tv2_s, self.scratch["forest_values_p"], 2)))
                body.append(("load", ("load", tv2_s, tv2_s)))
                body.append(("valu", ("vbroadcast", tv1_vec, tv1_s)))
                body.append(("valu", ("vbroadcast", tv2_vec, tv2_s)))

                for batch_idx in range(n_batches):
                    bg = batch_idx * BATCH
                    s = batch_idx % 2
                    for j in range(BATCH):
                        g = bg + j
                        body.append(("valu", ("<", ht1_tmp[s][j], idx_vecs[g], two_vec)))
                        body.append(("flow", ("vselect", nv_tmp[s][j], ht1_tmp[s][j], tv1_vec, tv2_vec)))

                    all_compute = []
                    for j in range(BATCH):
                        g = bg + j
                        all_compute.append(gen_compute_only_ops(j, g, nv_tmp[s][j], s))

                    max_ops = max(len(ops) for ops in all_compute)
                    for op_idx in range(max_ops):
                        for j in range(BATCH):
                            if op_idx < len(all_compute[j]):
                                body.append(all_compute[j][op_idx])

            else:
                # General case: scatter loads
                all_load_ops = []
                all_compute_ops = []

                for batch_idx in range(n_batches):
                    bg = batch_idx * BATCH
                    s = batch_idx % 2

                    batch_loads = []
                    batch_compute = []
                    for j in range(BATCH):
                        batch_loads.append(gen_load_ops(j, bg + j, s))
                        batch_compute.append(gen_compute_ops(j, bg + j, s))
                    all_load_ops.append(batch_loads)
                    all_compute_ops.append(batch_compute)

                # Prologue: load batch 0
                for op_idx in range(max(len(ops) for ops in all_load_ops[0])):
                    for j in range(BATCH):
                        if op_idx < len(all_load_ops[0][j]):
                            body.append(all_load_ops[0][j][op_idx])

                # Steady state
                for batch_idx in range(n_batches - 1):
                    compute_lists = all_compute_ops[batch_idx]
                    load_lists = all_load_ops[batch_idx + 1]

                    compute_flat = []
                    max_c = max(len(ops) for ops in compute_lists)
                    for op_idx in range(max_c):
                        for j in range(BATCH):
                            if op_idx < len(compute_lists[j]):
                                compute_flat.append(compute_lists[j][op_idx])

                    load_flat = []
                    max_l = max(len(ops) for ops in load_lists)
                    for op_idx in range(max_l):
                        for j in range(BATCH):
                            if op_idx < len(load_lists[j]):
                                load_flat.append(load_lists[j][op_idx])

                    ci, li = 0, 0
                    while ci < len(compute_flat) or li < len(load_flat):
                        for _ in range(3):
                            if ci < len(compute_flat):
                                body.append(compute_flat[ci])
                                ci += 1
                        if li < len(load_flat):
                            body.append(load_flat[li])
                            li += 1

                # Epilogue
                last_compute = all_compute_ops[n_batches - 1]
                max_c = max(len(ops) for ops in last_compute)
                for op_idx in range(max_c):
                    for j in range(BATCH):
                        if op_idx < len(last_compute[j]):
                            body.append(last_compute[j][op_idx])

        # Final store - use incremental addressing
        # First store: addr = inp_indices_p + 0
        eight_s = self.alloc_scratch("eight_s")
        body.append(("load", ("const", eight_s, V)))
        # Set initial addresses
        body.append(("flow", ("add_imm", idx_addr_s, self.scratch["inp_indices_p"], 0)))
        body.append(("flow", ("add_imm", val_addr_s, self.scratch["inp_values_p"], 0)))

        for g in range(n_groups):
            body.append(("store", ("vstore", idx_addr_s, idx_vecs[g])))
            body.append(("store", ("vstore", val_addr_s, val_vecs[g])))
            if g < n_groups - 1:
                body.append(("alu", ("+", idx_addr_s, idx_addr_s, eight_s)))
                body.append(("alu", ("+", val_addr_s, val_addr_s, eight_s)))

        self.instrs.extend(self.build(body))
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
