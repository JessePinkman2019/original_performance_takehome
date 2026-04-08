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

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
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

    def _slot_reads_writes(self, engine, slot):
        """Return (set_of_scratch_reads, set_of_scratch_writes) for a slot."""
        reads = set()
        writes = set()

        if engine == "alu":
            _, dest, a1, a2 = slot
            reads.add(a1)
            reads.add(a2)
            writes.add(dest)
        elif engine == "valu":
            if slot[0] == "vbroadcast":
                _, dest, src = slot
                reads.add(src)
                for i in range(VLEN):
                    writes.add(dest + i)
            elif slot[0] == "multiply_add":
                _, dest, a, b, c = slot
                for i in range(VLEN):
                    reads.add(a + i)
                    reads.add(b + i)
                    reads.add(c + i)
                    writes.add(dest + i)
            else:
                _, dest, a1, a2 = slot
                for i in range(VLEN):
                    reads.add(a1 + i)
                    reads.add(a2 + i)
                    writes.add(dest + i)
        elif engine == "load":
            if slot[0] == "const":
                _, dest, val = slot
                writes.add(dest)
            elif slot[0] == "load":
                _, dest, addr = slot
                reads.add(addr)
                writes.add(dest)
            elif slot[0] == "load_offset":
                _, dest, addr, offset = slot
                reads.add(addr + offset)
                writes.add(dest + offset)
            elif slot[0] == "vload":
                _, dest, addr = slot
                reads.add(addr)
                for i in range(VLEN):
                    writes.add(dest + i)
        elif engine == "store":
            if slot[0] == "store":
                _, addr, src = slot
                reads.add(addr)
                reads.add(src)
            elif slot[0] == "vstore":
                _, addr, src = slot
                reads.add(addr)
                for i in range(VLEN):
                    reads.add(src + i)
        elif engine == "flow":
            if slot[0] == "select":
                _, dest, cond, a, b = slot
                reads.add(cond)
                reads.add(a)
                reads.add(b)
                writes.add(dest)
            elif slot[0] == "vselect":
                _, dest, cond, a, b = slot
                for i in range(VLEN):
                    reads.add(cond + i)
                    reads.add(a + i)
                    reads.add(b + i)
                    writes.add(dest + i)
            elif slot[0] == "add_imm":
                _, dest, a, imm = slot
                reads.add(a)
                writes.add(dest)
            elif slot[0] in ("cond_jump", "cond_jump_rel"):
                reads.add(slot[1])
            elif slot[0] == "jump_indirect":
                reads.add(slot[1])
            elif slot[0] == "trace_write":
                reads.add(slot[1])
            elif slot[0] == "coreid":
                writes.add(slot[1])
        elif engine == "debug":
            pass

        return reads, writes

    def pack_slots(self, slots):
        """
        Pack slots into VLIW instruction bundles respecting data dependencies
        and slot limits. Uses topological ordering based on true dependencies.

        A slot B depends on slot A (must come after A) if:
        - A writes an address that B reads (RAW - true dependency), AND A comes before B
        - A reads an address that B writes (WAR - anti dependency), AND A comes before B
        - A writes an address that B writes (WAW - output dependency), AND A comes before B
        """
        if not slots:
            return []

        n = len(slots)
        # Precompute reads/writes for all slots
        slot_rw = []
        for i in range(n):
            engine, slot = slots[i]
            if engine == "debug":
                slot_rw.append((set(), set()))
            else:
                slot_rw.append(self._slot_reads_writes(engine, slot))

        # For each slot, find which slots it MUST wait for (predecessors)
        # A slot i depends on the LAST slot j < i that writes to any address that i reads/writes,
        # or reads any address that i writes.
        # We use "last writer/reader" tracking for efficiency.

        # last_writer[addr] = last slot index that writes to addr
        # last_readers[addr] = set of slot indices that read addr since last write
        last_writer = {}
        last_readers = defaultdict(set)
        deps = [set() for _ in range(n)]  # deps[i] = set of slots that must complete before i

        for i in range(n):
            reads_i, writes_i = slot_rw[i]

            # RAW: i reads from addr, and some j < i wrote to addr
            for addr in reads_i:
                if addr in last_writer:
                    deps[i].add(last_writer[addr])

            # WAR: i writes to addr, and some j < i read from addr
            # We need to ensure i doesn't execute before those readers
            for addr in writes_i:
                for reader in last_readers.get(addr, set()):
                    deps[i].add(reader)

            # WAW: i writes to addr, and some j < i also wrote to addr
            for addr in writes_i:
                if addr in last_writer:
                    deps[i].add(last_writer[addr])

            # Update tracking: record reads
            for addr in reads_i:
                last_readers[addr].add(i)

            # Update tracking: record writes (resets readers for that addr)
            for addr in writes_i:
                last_writer[addr] = i
                last_readers[addr] = set()  # Reset readers since this write supersedes

        # Compute depth (longest path from a root) for priority
        depth = [0] * n
        computed = [False] * n

        def compute_depth(i):
            if computed[i]:
                return depth[i]
            computed[i] = True
            if deps[i]:
                depth[i] = max(compute_depth(d) for d in deps[i]) + 1
            return depth[i]

        for i in range(n):
            compute_depth(i)

        # List-scheduling: pick ready slots by depth (critical path first)
        remaining_deps = [set(d) for d in deps]  # Mutable copy
        # Build reverse dep map: successors[i] = set of slots that depend on i
        successors = [set() for _ in range(n)]
        for i in range(n):
            for d in deps[i]:
                successors[d].add(i)

        ready = set()
        for i in range(n):
            if not remaining_deps[i]:
                ready.add(i)

        instructions = []

        while ready:
            bundle = {}
            bundle_writes = set()
            bundle_reads = set()
            bundle_counts = defaultdict(int)
            packed = []

            # Two-pass scheduling: first fill loads (bottleneck), then fill other engines
            # Pass 1: Pack load slots
            load_ready = [x for x in ready if slots[x][0] == 'load']
            load_ready.sort(key=lambda x: -depth[x])
            for idx in load_ready:
                engine, slot = slots[idx]
                if bundle_counts[engine] >= SLOT_LIMITS[engine]:
                    continue
                reads, writes = slot_rw[idx]
                if reads & bundle_writes:
                    continue
                if writes & bundle_writes:
                    continue
                if writes & bundle_reads:
                    continue
                if engine not in bundle:
                    bundle[engine] = []
                bundle[engine].append(slot)
                bundle_counts[engine] += 1
                bundle_writes |= writes
                bundle_reads |= reads
                packed.append(idx)

            # Pass 2: Pack remaining engines
            packed_set = set(packed)
            other_ready = sorted([x for x in ready if x not in packed_set],
                                  key=lambda x: -depth[x])

            for idx in other_ready:
                engine, slot = slots[idx]

                if engine == "debug":
                    packed.append(idx)
                    continue

                if bundle_counts[engine] >= SLOT_LIMITS[engine]:
                    continue

                reads, writes = slot_rw[idx]

                if reads & bundle_writes:
                    continue
                if writes & bundle_writes:
                    continue
                if writes & bundle_reads:
                    continue

                if engine not in bundle:
                    bundle[engine] = []
                bundle[engine].append(slot)
                bundle_counts[engine] += 1
                bundle_writes |= writes
                bundle_reads |= reads
                packed.append(idx)

            for idx in packed:
                ready.discard(idx)
                # Update successors
                for succ in successors[idx]:
                    remaining_deps[succ].discard(idx)
                    if not remaining_deps[succ]:
                        ready.add(succ)

            if bundle:
                instructions.append(bundle)

        return instructions

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel using VLIW packing, SIMD vectorization,
        and ALU-based per-lane address computation.
        """
        # --- Scratch allocation ---
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))

        # --- Vector scratch allocation ---
        BATCH_SIZE_GROUPS = 32  # Process ALL vector batches in one pass

        n_vec_batches = batch_size // VLEN
        assert batch_size % VLEN == 0
        # Cap groups to fit in scratch
        actual_max_groups = min(BATCH_SIZE_GROUPS, n_vec_batches)

        # Allocate per-group vector scratch
        # Reuse addr_vec as tmp_v1 and node_val_vec as tmp_v2
        # (they don't overlap in usage within the pipeline)
        group_data = []
        for g in range(actual_max_groups):
            gd = {}
            gd['idx_vec'] = self.alloc_scratch(f"idx_vec_{g}", VLEN)
            gd['val_vec'] = self.alloc_scratch(f"val_vec_{g}", VLEN)
            # addr_vec doubles as tmp_v1 (used for scatter addrs, then hash temps)
            gd['addr_vec'] = self.alloc_scratch(f"av_{g}", VLEN)
            gd['tmp_v1'] = gd['addr_vec']  # ALIAS
            # node_val_vec doubles as tmp_v2 (used for scatter results, then hash temps)
            gd['node_val_vec'] = self.alloc_scratch(f"nv_{g}", VLEN)
            gd['tmp_v2'] = gd['node_val_vec']  # ALIAS
            # Dedicated scalar addresses for load/store base pointers
            gd['idx_addr_s'] = self.alloc_scratch(f"ias_{g}", 1)
            gd['val_addr_s'] = self.alloc_scratch(f"vas_{g}", 1)
            group_data.append(gd)

        # Shared vector constants
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)

        # Hash constant vectors
        hash_const_vecs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hcv = {}
            hcv['val1_vec'] = self.alloc_scratch(f"hc_v1_{hi}", VLEN)
            hcv['val3_vec'] = self.alloc_scratch(f"hc_v3_{hi}", VLEN)

            # Check if this stage can use multiply_add optimization:
            # op2 == "+" and op3 == "<<" -> val = val * (1 + 2^val3) + op1(val, val1)
            # But only if op1 == "+" since multiply_add does a*b+c
            if op2 == "+" and op3 == "<<" and op1 == "+":
                mult_val = 1 + (1 << val3)
                hcv['use_ma'] = True
                hcv['mult_vec'] = self.alloc_scratch(f"hc_mult_{hi}", VLEN)
                hcv['mult_val'] = mult_val
            else:
                hcv['use_ma'] = False
            hash_const_vecs.append(hcv)

        # n_nodes vector for wrap comparison
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)

        # Pre-allocated scratch for broadcast of forest_values[0..2]
        fv0_scalar = self.alloc_scratch("fv0_scalar")
        fv0_vec = self.alloc_scratch("fv0_vec", VLEN)
        fv1_scalar = self.alloc_scratch("fv1_scalar")
        fv1_vec = self.alloc_scratch("fv1_vec", VLEN)
        fv2_scalar = self.alloc_scratch("fv2_scalar")
        fv2_vec = self.alloc_scratch("fv2_vec", VLEN)
        fv_diff_vec = self.alloc_scratch("fv_diff_vec", VLEN)  # fv1 - fv2 for round 12 select
        # For round 13 optimization
        fv3_scalar = self.alloc_scratch("fv3_scalar")
        fv3_vec = self.alloc_scratch("fv3_vec", VLEN)
        fv5_scalar = self.alloc_scratch("fv5_scalar")
        fv5_vec = self.alloc_scratch("fv5_vec", VLEN)
        fv4m3_vec = self.alloc_scratch("fv4m3_vec", VLEN)  # fv3 - fv4 (odd - even for select)
        fv6m5_vec = self.alloc_scratch("fv6m5_vec", VLEN)  # fv5 - fv6 (odd - even for select)
        fv4_scalar = self.alloc_scratch("fv4_scalar")
        fv4_vec = self.alloc_scratch("fv4_vec", VLEN)
        fv6_scalar = self.alloc_scratch("fv6_scalar")
        fv6_vec = self.alloc_scratch("fv6_vec", VLEN)
        shared_tmp = self.alloc_scratch("shared_tmp", VLEN)  # Extra temp for round 13

        fvp_scalar = self.scratch["forest_values_p"]

        # --- Initialization: broadcast vector constants ---
        init_slots = []
        init_slots.append(("valu", ("vbroadcast", zero_vec, zero_const)))
        init_slots.append(("valu", ("vbroadcast", one_vec, one_const)))
        init_slots.append(("valu", ("vbroadcast", two_vec, two_const)))
        init_slots.append(("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"])))
        three_const = self.scratch_const(3)
        three_vec = self.alloc_scratch("three_vec", VLEN)
        init_slots.append(("valu", ("vbroadcast", three_vec, three_const)))

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v1_const = self.scratch_const(val1)
            v3_const = self.scratch_const(val3)
            init_slots.append(("valu", ("vbroadcast", hash_const_vecs[hi]['val1_vec'], v1_const)))
            init_slots.append(("valu", ("vbroadcast", hash_const_vecs[hi]['val3_vec'], v3_const)))
            if hash_const_vecs[hi]['use_ma']:
                mult_const = self.scratch_const(hash_const_vecs[hi]['mult_val'])
                init_slots.append(("valu", ("vbroadcast", hash_const_vecs[hi]['mult_vec'], mult_const)))

        init_instrs = self.pack_slots(init_slots)
        self.instrs.extend(init_instrs)

        # --- Main loop body ---
        # Process all rounds for each batch of groups before moving to next batch.
        # This keeps idx/val in scratch across rounds, avoiding load/store per round.
        for vb_start in range(0, n_vec_batches, BATCH_SIZE_GROUPS):
            actual_groups = min(BATCH_SIZE_GROUPS, n_vec_batches - vb_start)

            # Load initial val from memory and set idx to zero (all initial indices are 0)
            load_slots = []
            for g in range(actual_groups):
                vb = vb_start + g
                gd = group_data[g]
                base_elem = vb * VLEN
                base_offset_const = self.scratch_const(base_elem)

                val_addr_s = gd['val_addr_s']
                idx_vec = gd['idx_vec']
                val_vec = gd['val_vec']

                load_slots.append(("alu", ("+", val_addr_s, self.scratch["inp_values_p"], base_offset_const)))
                load_slots.append(("load", ("vload", val_vec, val_addr_s)))
                # Set idx_vec to zero (all initial indices are 0)
                load_slots.append(("valu", ("+", idx_vec, zero_vec, zero_vec)))

            load_instrs = self.pack_slots(load_slots)
            self.instrs.extend(load_instrs)

            # Process all rounds with idx/val staying in scratch
            # Pack ALL rounds together for maximum ILP
            body_slots = []

            # Load forest values for broadcast-optimized rounds
            fvp1_const = self.scratch_const(1)
            fvp2_const = self.scratch_const(2)
            fvp3_const = self.scratch_const(3)
            fvp4_const = self.scratch_const(4)
            fvp5_const = self.scratch_const(5)
            fvp6_const = self.scratch_const(6)
            body_slots.append(("load", ("load", fv0_scalar, fvp_scalar)))
            body_slots.append(("alu", ("+", fv1_scalar, fvp_scalar, fvp1_const)))
            body_slots.append(("alu", ("+", fv2_scalar, fvp_scalar, fvp2_const)))
            body_slots.append(("alu", ("+", fv3_scalar, fvp_scalar, fvp3_const)))
            body_slots.append(("alu", ("+", fv5_scalar, fvp_scalar, fvp5_const)))
            body_slots.append(("load", ("load", fv1_scalar, fv1_scalar)))
            body_slots.append(("load", ("load", fv2_scalar, fv2_scalar)))
            body_slots.append(("load", ("load", fv3_scalar, fv3_scalar)))
            body_slots.append(("load", ("load", fv5_scalar, fv5_scalar)))
            body_slots.append(("valu", ("vbroadcast", fv0_vec, fv0_scalar)))
            body_slots.append(("valu", ("vbroadcast", fv1_vec, fv1_scalar)))
            body_slots.append(("valu", ("vbroadcast", fv2_vec, fv2_scalar)))
            body_slots.append(("valu", ("vbroadcast", fv3_vec, fv3_scalar)))
            body_slots.append(("valu", ("vbroadcast", fv5_vec, fv5_scalar)))
            body_slots.append(("valu", ("-", fv_diff_vec, fv1_vec, fv2_vec)))
            # Compute fv3-fv4 and fv5-fv6 diff vectors for round 13
            # (odd_val - even_val, with even as base)
            body_slots.append(("alu", ("+", fv4_scalar, fvp_scalar, fvp4_const)))
            body_slots.append(("load", ("load", fv4_scalar, fv4_scalar)))
            body_slots.append(("valu", ("vbroadcast", fv4_vec, fv4_scalar)))
            body_slots.append(("valu", ("-", fv4m3_vec, fv3_vec, fv4_vec)))  # fv3 - fv4
            body_slots.append(("alu", ("+", fv6_scalar, fvp_scalar, fvp6_const)))
            body_slots.append(("load", ("load", fv6_scalar, fv6_scalar)))
            body_slots.append(("valu", ("vbroadcast", fv6_vec, fv6_scalar)))
            body_slots.append(("valu", ("-", fv6m5_vec, fv5_vec, fv6_vec)))  # fv5 - fv6

            for rnd in range(rounds):
                for g in range(actual_groups):
                    gd = group_data[g]

                    idx_vec = gd['idx_vec']
                    val_vec = gd['val_vec']
                    node_val_vec = gd['node_val_vec']
                    addr_vec = gd['addr_vec']
                    tmp_v1 = gd['tmp_v1']
                    tmp_v2 = gd['tmp_v2']

                    all_idx_zero = (rnd == 0) or (rnd == forest_height + 1 and forest_height < rounds)
                    idx_in_1_2 = (rnd == forest_height + 2 and forest_height + 2 < rounds)
                    idx_in_3_6 = (rnd == forest_height + 3 and forest_height + 3 < rounds)

                    if all_idx_zero:
                        body_slots.append(("valu", ("+", node_val_vec, fv0_vec, zero_vec)))
                    elif idx_in_1_2:
                        body_slots.append(("valu", ("&", tmp_v1, idx_vec, one_vec)))
                        body_slots.append(("valu", ("multiply_add", node_val_vec, tmp_v1, fv_diff_vec, fv2_vec)))
                    elif idx_in_3_6:
                        # idx in {3,4,5,6}. Use 2-level multiply_add.
                        # bit0 = idx & 1 (1 for odd {3,5}, 0 for even {4,6})
                        body_slots.append(("valu", ("&", tmp_v1, idx_vec, one_vec)))
                        # low = bit0*(fv3-fv4)+fv4 : idx=3->fv3, idx=4->fv4 (stored in shared_tmp)
                        body_slots.append(("valu", ("multiply_add", shared_tmp, tmp_v1, fv4m3_vec, fv4_vec)))
                        # high = bit0*(fv5-fv6)+fv6 : idx=5->fv5, idx=6->fv6 (stored in node_val_vec)
                        body_slots.append(("valu", ("multiply_add", node_val_vec, tmp_v1, fv6m5_vec, fv6_vec)))
                        # bit1 = ((idx-3) >> 1) & 1 : 0 for {3,4}, 1 for {5,6}
                        body_slots.append(("valu", ("-", tmp_v1, idx_vec, three_vec)))
                        body_slots.append(("valu", (">>", tmp_v1, tmp_v1, one_vec)))
                        body_slots.append(("valu", ("&", tmp_v1, tmp_v1, one_vec)))
                        # diff = high - low (overwrites node_val_vec with diff)
                        body_slots.append(("valu", ("-", node_val_vec, node_val_vec, shared_tmp)))
                        # result = bit1*diff + low = multiply_add(node_val_vec, bit1, diff, low)
                        body_slots.append(("valu", ("multiply_add", node_val_vec, tmp_v1, node_val_vec, shared_tmp)))
                    else:
                        for lane in range(VLEN):
                            body_slots.append(("alu", ("+", addr_vec + lane, idx_vec + lane, fvp_scalar)))
                        for lane in range(VLEN):
                            body_slots.append(("load", ("load_offset", node_val_vec, addr_vec, lane)))

                    body_slots.append(("valu", ("^", val_vec, val_vec, node_val_vec)))

                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        v1_vec = hash_const_vecs[hi]['val1_vec']
                        v3_vec = hash_const_vecs[hi]['val3_vec']
                        if hash_const_vecs[hi]['use_ma']:
                            mult_vec = hash_const_vecs[hi]['mult_vec']
                            body_slots.append(("valu", ("multiply_add", val_vec, val_vec, mult_vec, v1_vec)))
                        else:
                            body_slots.append(("valu", (op1, tmp_v1, val_vec, v1_vec)))
                            body_slots.append(("valu", (op3, tmp_v2, val_vec, v3_vec)))
                            body_slots.append(("valu", (op2, val_vec, tmp_v1, tmp_v2)))

                    # idx = 2*idx + (val&1) + 1
                    body_slots.append(("valu", ("&", tmp_v1, val_vec, one_vec)))
                    body_slots.append(("valu", ("+", tmp_v1, tmp_v1, one_vec)))
                    body_slots.append(("valu", ("multiply_add", idx_vec, idx_vec, two_vec, tmp_v1)))

                    if rnd == forest_height:
                        # idx = 0 if idx >= n_nodes else idx
                        # Using VALU: cond = (idx < n_nodes), idx = idx * cond
                        body_slots.append(("valu", ("<", tmp_v1, idx_vec, n_nodes_vec)))
                        body_slots.append(("valu", ("*", idx_vec, idx_vec, tmp_v1)))

            packed = self.pack_slots(body_slots)
            self.instrs.extend(packed)

            # Store final idx and val back to memory
            store_slots = []
            for g in range(actual_groups):
                vb = vb_start + g
                gd = group_data[g]
                base_elem = vb * VLEN
                base_offset_const = self.scratch_const(base_elem)

                idx_addr_s = gd['idx_addr_s']
                val_addr_s = gd['val_addr_s']
                idx_vec = gd['idx_vec']
                val_vec = gd['val_vec']

                store_slots.append(("alu", ("+", idx_addr_s, self.scratch["inp_indices_p"], base_offset_const)))
                store_slots.append(("alu", ("+", val_addr_s, self.scratch["inp_values_p"], base_offset_const)))
                store_slots.append(("store", ("vstore", idx_addr_s, idx_vec)))
                store_slots.append(("store", ("vstore", val_addr_s, val_vec)))

            store_instrs = self.pack_slots(store_slots)
            self.instrs.extend(store_instrs)

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

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
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
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
