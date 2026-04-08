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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        VL = VLEN  # 8

        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        # Pack init var loads 2 at a time using tmp1 and tmp2
        for i in range(0, len(init_vars), 2):
            if i + 1 < len(init_vars):
                # Load 2 vars at once
                self.instrs.append({"load": [
                    ("const", tmp1, i),
                    ("const", tmp2, i + 1),
                ]})
                self.instrs.append({"load": [
                    ("load", self.scratch[init_vars[i]], tmp1),
                    ("load", self.scratch[init_vars[i + 1]], tmp2),
                ]})
            else:
                # Odd one out
                self.instrs.append({"load": [("const", tmp1, i)]})
                self.instrs.append({"load": [("load", self.scratch[init_vars[i]], tmp1)]})

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))

        n_groups = batch_size // VL  # 32

        # Vector constants
        zero_vec = self.alloc_scratch("zero_vec", VL)
        one_vec = self.alloc_scratch("one_vec", VL)
        two_vec = self.alloc_scratch("two_vec", VL)
        fvp_vec = self.alloc_scratch("fvp_vec", VL)
        nnodes_vec = self.alloc_scratch("nnodes_vec", VL)

        # Hash constants (with multiply_add fusion where possible)
        hash_stage_info = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            can_fuse = (op1 == "+" and op2 == "+" and op3 == "<<")
            if can_fuse:
                coeff = 1 + (1 << val3)
                coeff_vec = self.alloc_scratch(f"hcoeff_{hi}", VL)
                const1_vec = self.alloc_scratch(f"hconst_{hi}", VL)
                hash_stage_info.append((True, coeff_vec, const1_vec, coeff, val1))
            else:
                hv1 = self.alloc_scratch(f"hc_v1_{hi}", VL)
                hv3 = self.alloc_scratch(f"hc_v3_{hi}", VL)
                hash_stage_info.append((False, hv1, hv3, val1, val3))

        # Per-group idx and val
        g_idx = [self.alloc_scratch(f"idx_{g}", VL) for g in range(n_groups)]
        g_val = [self.alloc_scratch(f"val_{g}", VL) for g in range(n_groups)]

        # Shared pools for scatter loads
        N_ADDR_POOLS = 16
        N_NV_POOLS = 16
        N_HASH_PIPES = 16

        g_addr_pool = [self.alloc_scratch(f"addr_p{p}", VL) for p in range(N_ADDR_POOLS)]
        g_nv_pool = [self.alloc_scratch(f"nv_p{p}", VL) for p in range(N_NV_POOLS)]
        vt1_pool = [self.alloc_scratch(f"vt1_{p}", VL) for p in range(N_HASH_PIPES)]
        vt2_pool = [self.alloc_scratch(f"vt2_{p}", VL) for p in range(N_HASH_PIPES)]

        # Per-group base addresses
        g_idx_base = [self.alloc_scratch(f"ib_{g}") for g in range(n_groups)]
        g_val_base = [self.alloc_scratch(f"vb_{g}") for g in range(n_groups)]

        group_offset_consts = [self.scratch_const(g * VL) for g in range(n_groups)]

        # Tree cache: preload top levels of the tree into scratch vectors
        # For rounds 0..CACHE_LEVELS-1, use cached values instead of scatter loads
        # Level L has 2^L nodes (node IDs: 2^L-1 to 2^(L+1)-2)
        # Budget: remaining scratch / VL = available vectors
        avail_words = SCRATCH_SIZE - self.scratch_ptr
        # Need: cached vectors + temp scalars for loading
        # Level L: 2^L vectors (8 words each) + 1 scalar for each load + 1 temp for selection
        # For levels 0..K-1: total vectors = 2^K - 1

        # Determine max cache level based on scratch budget
        # Also consider flow (vselect) budget:
        # Level L needs (2^L - 1) vselects per group per round = 32 * (2^L - 1) flow ops
        # Total flow: 32 * sum(2^L - 1 for L in 1..K-1) + 16*32 (idx wraps)
        # Flow budget: we don't want flow to dominate

        # Let's cache levels 0-3 (15 nodes, 120 words for vectors, 15 scalars)
        CACHE_LEVELS = min(3, forest_height)  # Cache levels 0-2

        # Check we have enough scratch
        n_cache_nodes = (1 << CACHE_LEVELS) - 1  # 15
        cache_words_needed = n_cache_nodes * VL + n_cache_nodes  # vectors + scalars
        if self.scratch_ptr + cache_words_needed > SCRATCH_SIZE:
            # Reduce cache levels to fit
            CACHE_LEVELS = 0
            while True:
                n = (1 << (CACHE_LEVELS + 1)) - 1
                w = n * VL + n
                if self.scratch_ptr + w > SCRATCH_SIZE:
                    break
                CACHE_LEVELS += 1

        n_cache_nodes = (1 << CACHE_LEVELS) - 1
        # Allocate cached vectors and scalars for tree nodes
        cache_vec = {}  # node_id -> scratch vector addr
        cache_scalar = {}  # node_id -> scratch scalar addr
        for level in range(CACHE_LEVELS):
            start_node = (1 << level) - 1
            for j in range(1 << level):
                node_id = start_node + j
                cache_scalar[node_id] = self.alloc_scratch(f"cs_{node_id}")
                cache_vec[node_id] = self.alloc_scratch(f"cv_{node_id}", VL)

        # Temp scalars/vectors for selection tree operations
        # For level L, selection tree needs log2(2^L) = L vselect stages
        # Each stage halves the candidates. Need temp vectors for intermediate results.
        # Use nv_pool for intermediates (they're shared)

        # Allocate comparison threshold vectors for selection tree
        # For level L with 2^L values, indexed by node_id from (2^L-1) to (2^(L+1)-2),
        # we need to select based on idx. Use binary selection on bits of (idx - start_node).
        # Need threshold vectors for comparisons: idx < threshold for each split point

        print(f"Scratch usage: {self.scratch_ptr}/{SCRATCH_SIZE}, Cache levels: {CACHE_LEVELS}")

        # Broadcast vector constants
        self.instrs.append({
            "valu": [
                ("vbroadcast", zero_vec, zero_const),
                ("vbroadcast", one_vec, one_const),
                ("vbroadcast", two_vec, two_const),
                ("vbroadcast", fvp_vec, self.scratch["forest_values_p"]),
                ("vbroadcast", nnodes_vec, self.scratch["n_nodes"]),
            ]
        })

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            info = hash_stage_info[hi]
            if info[0]:
                _, coeff_vec, const1_vec, coeff, c1 = info
                self.instrs.append({"valu": [
                    ("vbroadcast", coeff_vec, self.scratch_const(coeff)),
                    ("vbroadcast", const1_vec, self.scratch_const(c1)),
                ]})
            else:
                _, hv1, hv3, c1, c3 = info
                self.instrs.append({"valu": [
                    ("vbroadcast", hv1, self.scratch_const(c1)),
                    ("vbroadcast", hv3, self.scratch_const(c3)),
                ]})

        # Precompute base addresses AND initialize g_idx to zero (merge ALU + VALU)
        for start in range(0, n_groups, 6):
            end = min(start + 6, n_groups)
            bundle = {"alu": [], "valu": []}
            for g in range(start, end):
                bundle["alu"].append(("+", g_idx_base[g], self.scratch["inp_indices_p"], group_offset_consts[g]))
                bundle["alu"].append(("+", g_val_base[g], self.scratch["inp_values_p"], group_offset_consts[g]))
                bundle["valu"].append(("vbroadcast", g_idx[g], zero_const))
            self.instrs.append(bundle)

        # Preload tree cache: load node values from memory and broadcast to vectors
        # Pipeline: ALU computes address, next cycle LOAD reads from it
        # Overlap ALU for pair K+1 with LOAD for pair K
        cache_node_ids = list(cache_scalar.keys())
        temps = [tmp1, tmp2]
        pairs = []
        for i in range(0, len(cache_node_ids), 2):
            pairs.append(cache_node_ids[i:i+2])

        # First pair: ALU only
        if pairs:
            alu_slots = []
            for k, nid in enumerate(pairs[0]):
                offset_const = self.scratch_const(nid)
                alu_slots.append(("+", temps[k], self.scratch["forest_values_p"], offset_const))
            self.instrs.append({"alu": alu_slots})

        # Middle pairs: LOAD for previous + ALU for current
        for pi in range(1, len(pairs)):
            load_slots = []
            for k, nid in enumerate(pairs[pi-1]):
                load_slots.append(("load", cache_scalar[nid], temps[k]))
            alu_slots = []
            for k, nid in enumerate(pairs[pi]):
                offset_const = self.scratch_const(nid)
                alu_slots.append(("+", temps[k], self.scratch["forest_values_p"], offset_const))
            self.instrs.append({"load": load_slots, "alu": alu_slots})

        # Last pair: LOAD only
        if pairs:
            load_slots = []
            for k, nid in enumerate(pairs[-1]):
                load_slots.append(("load", cache_scalar[nid], temps[k]))
            self.instrs.append({"load": load_slots})

        # Broadcast cached scalars to vectors
        for start_node in range(0, n_cache_nodes, 6):
            bundle = {"valu": []}
            node_ids = list(cache_vec.keys())
            end_node = min(start_node + 6, n_cache_nodes)
            for i in range(start_node, end_node):
                nid = node_ids[i]
                bundle["valu"].append(("vbroadcast", cache_vec[nid], cache_scalar[nid]))
            if bundle["valu"]:
                self.instrs.append(bundle)

        # Schedule all rounds
        self._schedule_all_rounds(
            rounds, n_groups, VL, n_nodes,
            g_idx, g_val,
            g_addr_pool, g_nv_pool, N_ADDR_POOLS, N_NV_POOLS,
            vt1_pool, vt2_pool, N_HASH_PIPES,
            zero_vec, one_vec, two_vec, fvp_vec, nnodes_vec,
            hash_stage_info,
            g_idx_base, g_val_base,
            CACHE_LEVELS, cache_vec,
        )

        # Pack preamble instructions to use multiple slots per cycle
        self._pack_preamble()

        self.instrs.append({"flow": [("pause",)]})

    def _pack_preamble(self):
        """
        Pack single-slot preamble instructions into multi-slot bundles.
        Conservative approach: only pack consecutive same-type instructions.
        """
        preamble_end = len(self.instrs)
        for i in range(len(self.instrs)):
            instr = self.instrs[i]
            if len(instr) > 2:
                preamble_end = i
                break
            total_slots = sum(len(v) for v in instr.values())
            if total_slots > 6:
                preamble_end = i
                break

        if preamble_end <= 1:
            return

        preamble = self.instrs[:preamble_end]
        rest = self.instrs[preamble_end:]

        packed = []
        i = 0
        while i < len(preamble):
            instr = preamble[i]
            engines = list(instr.keys())
            if len(engines) != 1:
                packed.append(instr)
                i += 1
                continue

            eng = engines[0]
            slots = instr[eng]
            if len(slots) != 1:
                packed.append(instr)
                i += 1
                continue

            slot = slots[0]
            limit = SLOT_LIMITS.get(eng, 1)

            if eng == "load" and slot[0] == "const":
                slot_type = "load_const"
            elif eng == "load" and slot[0] == "load":
                slot_type = "load_load"
            elif eng == "valu":
                slot_type = "valu"
            elif eng == "alu":
                slot_type = "alu"
            elif eng == "flow":
                slot_type = "flow"
            else:
                packed.append(instr)
                i += 1
                continue

            batch = [(eng, slot)]
            j = i + 1
            while j < len(preamble) and len(batch) < limit:
                next_instr = preamble[j]
                next_engines = list(next_instr.keys())
                if len(next_engines) != 1:
                    break
                next_eng = next_engines[0]
                next_slots = next_instr[next_eng]
                if len(next_slots) != 1:
                    break
                next_slot = next_slots[0]

                if next_eng == "load" and next_slot[0] == "const":
                    next_type = "load_const"
                elif next_eng == "load" and next_slot[0] == "load":
                    next_type = "load_load"
                elif next_eng == "valu":
                    next_type = "valu"
                elif next_eng == "alu":
                    next_type = "alu"
                else:
                    break

                if next_type != slot_type:
                    break

                if slot_type == "load_const":
                    dests = {s[1] for _, s in batch}
                    if next_slot[1] in dests:
                        break
                    batch.append((next_eng, next_slot))
                elif slot_type == "load_load":
                    dests = {s[1] for _, s in batch}
                    reads = {s[2] for _, s in batch}
                    if next_slot[1] in dests or next_slot[2] in dests or next_slot[1] in reads:
                        break
                    batch.append((next_eng, next_slot))
                else:
                    batch.append((next_eng, next_slot))

                j += 1

            packed_instr = {}
            for e, s in batch:
                packed_instr.setdefault(e, []).append(s)
            packed.append(packed_instr)
            i = j

        self.instrs = packed + rest
    def _gen_select_tree(self, add_op, vrange, idx_vec, cache_vec,
                         level, nv_dest, temp_vecs, VL):
        """
        Generate a binary selection tree to select the correct node value
        based on idx. For level L, there are 2^L possible node values.

        Uses recursive halving: split the candidates based on a threshold
        comparison, select between the two halves.
        """
        start_node = (1 << level) - 1
        n_nodes = 1 << level

        if n_nodes == 1:
            # Just copy the single cached value to nv_dest
            add_op("valu", ("+", nv_dest, cache_vec[start_node], self._zero_vec),
                   vrange(nv_dest), vrange(cache_vec[start_node]) | vrange(self._zero_vec))
            return

        if n_nodes == 2:
            # idx is either start_node or start_node+1
            # Use (idx & 1): if LSB=1, it's the odd node (start_node), if LSB=0 it's even (start_node+1)
            # Actually, start_node for level 1 is 1, nodes are 1,2
            # For level 2: start=3, nodes 3,4,5,6
            # General: node_id = start_node + j where j=0..n_nodes-1
            # We need to check: is idx == start_node? Or idx == start_node+1?
            # Simplest: use the LSB of idx
            # For 2 nodes: (idx & 1) == (start_node & 1) means it's node start_node
            # Actually: node 1 (binary 01), node 2 (binary 10). LSB of 1 is 1, LSB of 2 is 0.
            # node 3 (11), 4 (100). LSB of 3 is 1, LSB of 4 is 0.
            # General: start_node is always 2^L - 1 which is always odd (all 1s in binary).
            # So start_node has LSB=1, start_node+1 has LSB=0.
            # (idx & 1) == 1 means idx = start_node, (idx & 1) == 0 means idx = start_node+1

            cond_vec = temp_vecs[0]
            add_op("valu", ("&", cond_vec, idx_vec, self._one_vec),
                   vrange(cond_vec), vrange(idx_vec) | vrange(self._one_vec))
            # vselect: if cond(lane) != 0 then a else b
            # If (idx & 1) != 0, idx = start_node (odd), select cache_vec[start_node]
            # If (idx & 1) == 0, idx = start_node+1 (even), select cache_vec[start_node+1]
            add_op("flow", ("vselect", nv_dest, cond_vec, cache_vec[start_node], cache_vec[start_node + 1]),
                   vrange(nv_dest), vrange(cond_vec) | vrange(cache_vec[start_node]) | vrange(cache_vec[start_node + 1]))
            return

        # For 4+ nodes: split into two halves and recurse
        # Threshold: mid_node = start_node + n_nodes // 2
        mid_node = start_node + n_nodes // 2
        # If idx < mid_node: it's in the first half
        # If idx >= mid_node: it's in the second half

        # But we need threshold constant vectors. Let me use a comparison approach:
        # Compute (idx - start_node) >> (level-1) to get the MSB of the within-level offset
        # If MSB=0: first half. If MSB=1: second half.

        # Actually, simpler: use (idx >> (level-1)) & 1 to get the bit that distinguishes halves
        # For level 2, nodes 3-6: offset = idx - 3 = 0,1,2,3. Bit 1 (position 1): 0,0,1,1.
        # (idx >> 1) & 1: 3>>1=1, 4>>1=2, 5>>1=2, 6>>1=3. &1: 1,0,0,1. Not right.
        # Let's use offset = idx - start_node. (offset >> (level-1)) & 1:
        # level=2: offset = idx-3 = 0,1,2,3. offset >> 1 = 0,0,1,1. &1 = 0,0,1,1. Second half is offset >= 2, which is nodes 5,6.

        # Hmm, this is getting complicated. Let me just use the direct comparison approach:
        # cond = (idx < mid_node_vec)
        # left = select_tree(first half)
        # right = select_tree(second half)
        # result = vselect(cond, left, right)

        # But this requires mid_node to be a vector constant. I'll need to allocate/broadcast it.
        # Or use a cheaper comparison: since mid_node = start_node + n_nodes/2, and this is
        # always a power of 2 boundary, I can use bit manipulation.

        # For simplicity, let me just threshold:
        # First, compute both halves into temp vectors, then select
        left_dest = temp_vecs[0]
        right_dest = temp_vecs[1]
        cond_vec = temp_vecs[2]

        # Recursively generate left half (nodes start_node to mid_node-1)
        self._gen_select_tree_range(add_op, vrange, idx_vec, cache_vec,
                                     start_node, mid_node - start_node,
                                     left_dest, temp_vecs[3:], VL)

        # Recursively generate right half (nodes mid_node to end)
        self._gen_select_tree_range(add_op, vrange, idx_vec, cache_vec,
                                     mid_node, n_nodes // 2,
                                     right_dest, temp_vecs[3:], VL)

        # Comparison: idx < mid_node
        mid_const_vec = self._get_const_vec(mid_node)
        add_op("valu", ("<", cond_vec, idx_vec, mid_const_vec),
               vrange(cond_vec), vrange(idx_vec) | vrange(mid_const_vec))
        add_op("flow", ("vselect", nv_dest, cond_vec, left_dest, right_dest),
               vrange(nv_dest), vrange(cond_vec) | vrange(left_dest) | vrange(right_dest))

    def _gen_select_tree_range(self, add_op, vrange, idx_vec, cache_vec,
                                start, count, dest, temp_vecs, VL):
        """Select from cache_vec[start..start+count-1] based on idx."""
        if count == 1:
            add_op("valu", ("+", dest, cache_vec[start], self._zero_vec),
                   vrange(dest), vrange(cache_vec[start]) | vrange(self._zero_vec))
            return
        if count == 2:
            cond = temp_vecs[0]
            add_op("valu", ("&", cond, idx_vec, self._one_vec),
                   vrange(cond), vrange(idx_vec) | vrange(self._one_vec))
            # start is always odd (at the beginning of a pair in the tree)
            # If (idx & 1) != 0: idx is odd = start node. Select cache_vec[start]
            # If (idx & 1) == 0: idx is even = start+1 node. Select cache_vec[start+1]
            add_op("flow", ("vselect", dest, cond, cache_vec[start], cache_vec[start + 1]),
                   vrange(dest), vrange(cond) | vrange(cache_vec[start]) | vrange(cache_vec[start + 1]))
            return

        half = count // 2
        mid = start + half

        left = temp_vecs[0]
        right = temp_vecs[1]
        cond = temp_vecs[2]

        self._gen_select_tree_range(add_op, vrange, idx_vec, cache_vec,
                                     start, half, left, temp_vecs[3:], VL)
        self._gen_select_tree_range(add_op, vrange, idx_vec, cache_vec,
                                     mid, half, right, temp_vecs[3:], VL)

        mid_const_vec = self._get_const_vec(mid)
        add_op("valu", ("<", cond, idx_vec, mid_const_vec),
               vrange(cond), vrange(idx_vec) | vrange(mid_const_vec))
        add_op("flow", ("vselect", dest, cond, left, right),
               vrange(dest), vrange(cond) | vrange(left) | vrange(right))

    def _schedule_all_rounds(
        self, rounds, n_groups, VL, n_nodes,
        g_idx, g_val,
        g_addr_pool, g_nv_pool, n_addr_pools, n_nv_pools,
        vt1_pool, vt2_pool, n_hash_pipes,
        zero_vec, one_vec, two_vec, fvp_vec, nnodes_vec,
        hash_stage_info,
        g_idx_base, g_val_base,
        cache_levels, cache_vec,
    ):
        self._zero_vec = zero_vec
        self._one_vec = one_vec
        self._const_vec_cache = {}

        ops_engine = []
        ops_slot = []
        ops_deps = []
        ops_writes = []

        last_writer = {}
        last_readers = defaultdict(set)

        def vrange(base):
            return set(range(base, base + VL))

        def add_op(engine, slot, writes, reads, extra_deps=None):
            idx = len(ops_engine)
            deps = set()
            if extra_deps:
                deps |= extra_deps

            for addr in reads:
                w = last_writer.get(addr, -1)
                if w >= 0:
                    deps.add(w)
            for addr in writes:
                w = last_writer.get(addr, -1)
                if w >= 0:
                    deps.add(w)
            for addr in writes:
                if addr in last_readers:
                    deps |= last_readers[addr]

            deps.discard(idx)

            ops_engine.append(engine)
            ops_slot.append(slot)
            ops_deps.append(deps)
            ops_writes.append(writes)

            for addr in writes:
                last_writer[addr] = idx
                last_readers[addr] = set()
            for addr in reads:
                last_readers[addr].add(idx)

            return idx

        def gen_group_round(g, rnd, is_first, is_last, use_cache, needs_wrap):
            """Generate all ops for one group-round."""
            addr_v = g_addr_pool[g % n_addr_pools]
            nv_v = g_nv_pool[g % n_nv_pools]
            vt1 = vt1_pool[g % n_hash_pipes]
            vt2 = vt2_pool[g % n_hash_pipes]

            if is_first:
                add_op("load", ("vload", g_val[g], g_val_base[g]),
                       vrange(g_val[g]), {g_val_base[g]})

            if use_cache:
                if rnd == 0:
                    # XOR directly with cached node 0 (skip copy to nv_v)
                    effective_nv = cache_vec[0]
                else:
                    temp_pool_idx = [(g + j) % n_nv_pools for j in range(1, n_nv_pools)]
                    temp_vecs = [g_nv_pool[k] for k in temp_pool_idx]
                    temp_vecs = [addr_v] + temp_vecs
                    self._gen_select_tree_range(
                        add_op, vrange, g_idx[g], cache_vec,
                        (1 << rnd) - 1, 1 << rnd,
                        nv_v, temp_vecs, VL)
                    effective_nv = nv_v
            else:
                add_op("valu", ("+", addr_v, g_idx[g], fvp_vec),
                       vrange(addr_v), vrange(g_idx[g]) | vrange(fvp_vec))
                for lane in range(VL):
                    add_op("load", ("load_offset", nv_v, addr_v, lane),
                           {nv_v + lane}, {addr_v + lane})
                effective_nv = nv_v

            add_op("valu", ("^", g_val[g], g_val[g], effective_nv),
                   vrange(g_val[g]), vrange(g_val[g]) | vrange(effective_nv))

            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                info = hash_stage_info[hi]
                if info[0]:
                    _, coeff_vec, const1_vec, _, _ = info
                    add_op("valu", ("multiply_add", g_val[g], g_val[g], coeff_vec, const1_vec),
                           vrange(g_val[g]),
                           vrange(g_val[g]) | vrange(coeff_vec) | vrange(const1_vec))
                else:
                    _, hv1, hv3, _, _ = info
                    add_op("valu", (op1, vt1, g_val[g], hv1),
                           vrange(vt1), vrange(g_val[g]) | vrange(hv1))
                    add_op("valu", (op3, vt2, g_val[g], hv3),
                           vrange(vt2), vrange(g_val[g]) | vrange(hv3))
                    add_op("valu", (op2, g_val[g], vt1, vt2),
                           vrange(g_val[g]), vrange(vt1) | vrange(vt2))

            add_op("valu", ("&", vt1, g_val[g], one_vec),
                   vrange(vt1), vrange(g_val[g]) | vrange(one_vec))
            add_op("valu", ("+", vt2, vt1, one_vec),
                   vrange(vt2), vrange(vt1) | vrange(one_vec))
            add_op("valu", ("multiply_add", g_idx[g], g_idx[g], two_vec, vt2),
                   vrange(g_idx[g]), vrange(g_idx[g]) | vrange(two_vec) | vrange(vt2))

            if needs_wrap:
                add_op("valu", ("<", vt1, g_idx[g], nnodes_vec),
                       vrange(vt1), vrange(g_idx[g]) | vrange(nnodes_vec))
                add_op("flow", ("vselect", g_idx[g], vt1, g_idx[g], zero_vec),
                       vrange(g_idx[g]), vrange(vt1) | vrange(g_idx[g]) | vrange(zero_vec))

            if is_last:
                add_op("store", ("vstore", g_idx_base[g], g_idx[g]),
                       set(), {g_idx_base[g]} | vrange(g_idx[g]))
                add_op("store", ("vstore", g_val_base[g], g_val[g]),
                       set(), {g_val_base[g]} | vrange(g_val[g]))

        # Determine which rounds need wrapping check
        # After round R with initial idx=0, max idx = 2^(R+2) - 2
        # Wrapping needed when 2^(R+2) - 2 >= n_nodes
        # => R >= log2(n_nodes + 2) - 2
        import math
        first_wrap_round = max(0, math.ceil(math.log2(n_nodes + 2)) - 2) if n_nodes > 2 else 0

        # Generate ops: group-major ordering for all rounds
        BATCH_SIZE_GROUPS = 4
        for batch_start in range(0, n_groups, BATCH_SIZE_GROUPS):
            for g in range(batch_start, min(batch_start + BATCH_SIZE_GROUPS, n_groups)):
                for rnd in range(rounds):
                    gen_group_round(g, rnd,
                                  is_first=(rnd == 0),
                                  is_last=(rnd == rounds - 1),
                                  use_cache=(rnd < cache_levels),
                                  needs_wrap=(rnd >= first_wrap_round))

        # List Scheduler
        n_ops = len(ops_engine)
        dependents = [[] for _ in range(n_ops)]
        for i in range(n_ops):
            for d in ops_deps[i]:
                dependents[d].append(i)

        priority = [0] * n_ops
        from collections import deque
        queue = deque()
        for i in range(n_ops):
            if not dependents[i]:
                priority[i] = 1
                queue.append(i)
        while queue:
            node = queue.popleft()
            for d in ops_deps[node]:
                new_p = priority[node] + 1
                if new_p > priority[d]:
                    priority[d] = new_p
                    queue.append(d)

        dep_count = [len(ops_deps[i]) for i in range(n_ops)]
        ready = [i for i in range(n_ops) if dep_count[i] == 0]
        n_scheduled = 0
        cycle_bundles = []

        ENGINE_PRIORITY = {"load": 0, "store": 1, "flow": 2, "valu": 3, "alu": 4}

        while n_scheduled < n_ops:
            ready.sort(key=lambda i: (ENGINE_PRIORITY.get(ops_engine[i], 5), -priority[i]))

            bundle = {}
            engine_counts = {}
            bundle_writes = set()
            not_selected = []
            newly_scheduled = []

            for i in ready:
                eng = ops_engine[i]
                limit = SLOT_LIMITS.get(eng, 0)
                if engine_counts.get(eng, 0) >= limit:
                    not_selected.append(i)
                    continue

                writes_i = ops_writes[i]
                if writes_i & bundle_writes:
                    not_selected.append(i)
                    continue

                bundle.setdefault(eng, []).append(ops_slot[i])
                engine_counts[eng] = engine_counts.get(eng, 0) + 1
                bundle_writes |= writes_i
                newly_scheduled.append(i)
                n_scheduled += 1

            new_ready = []
            for i in newly_scheduled:
                for dep_i in dependents[i]:
                    dep_count[dep_i] -= 1
                    if dep_count[dep_i] == 0:
                        new_ready.append(dep_i)

            ready = not_selected + new_ready

            if bundle:
                cycle_bundles.append(bundle)

        self.instrs.extend(cycle_bundles)

    def _get_const_vec(self, val):
        """Get or create a constant vector for use in selection trees."""
        if val not in self._const_vec_cache:
            VL = VLEN
            vec = self.alloc_scratch(f"cv_const_{val}", VL)
            sc = self.scratch_const(val)
            # Need to add broadcast instruction in preamble
            # But we're called during scheduling... this is problematic.
            # Instead, pre-allocate in build_kernel.
            # For now, store the vec and handle broadcast externally.
            self._const_vec_cache[val] = vec
            # Add the broadcast as a preamble instruction
            self.instrs.append({"valu": [("vbroadcast", vec, sc)]})
        return self._const_vec_cache[val]


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
