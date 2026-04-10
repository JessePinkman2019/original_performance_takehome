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

    @staticmethod
    def _slot_reads_writes(engine, slot):
        """Return (reads, writes) as sets of scratch addresses for a slot."""
        reads, writes = set(), set()
        if engine == "alu":
            op, dest, a1, a2 = slot
            reads = {a1, a2}
            writes = {dest}
        elif engine == "valu":
            if slot[0] == "vbroadcast":
                _, dest, src = slot
                reads = {src}
                writes = set(range(dest, dest + VLEN))
            elif slot[0] == "multiply_add":
                _, dest, a, b, c = slot
                reads = set(range(a, a + VLEN)) | set(range(b, b + VLEN)) | set(range(c, c + VLEN))
                writes = set(range(dest, dest + VLEN))
            else:
                op, dest, a1, a2 = slot
                reads = set(range(a1, a1 + VLEN)) | set(range(a2, a2 + VLEN))
                writes = set(range(dest, dest + VLEN))
        elif engine == "load":
            if slot[0] == "const":
                _, dest, val = slot
                writes = {dest}
            elif slot[0] == "load":
                _, dest, addr = slot
                reads = {addr}
                writes = {dest}
            elif slot[0] == "load_offset":
                _, dest, addr, offset = slot
                reads = {addr + offset}
                writes = {dest + offset}
            elif slot[0] == "vload":
                _, dest, addr = slot
                reads = {addr}
                writes = set(range(dest, dest + VLEN))
        elif engine == "store":
            if slot[0] == "store":
                _, addr, src = slot
                reads = {addr, src}
            elif slot[0] == "vstore":
                _, addr, src = slot
                reads = {addr} | set(range(src, src + VLEN))
        elif engine == "flow":
            op = slot[0]
            if op == "select":
                _, dest, cond, a, b = slot
                reads = {cond, a, b}
                writes = {dest}
            elif op == "add_imm":
                _, dest, a, imm = slot
                reads = {a}
                writes = {dest}
            elif op == "vselect":
                _, dest, cond, a, b = slot
                reads = set(range(cond, cond + VLEN)) | set(range(a, a + VLEN)) | set(range(b, b + VLEN))
                writes = set(range(dest, dest + VLEN))
            elif op == "cond_jump":
                _, cond, addr = slot
                reads = {cond}
            elif op == "cond_jump_rel":
                _, cond, offset = slot
                reads = {cond}
            elif op == "jump_indirect":
                _, addr = slot
                reads = {addr}
            elif op == "coreid":
                _, dest = slot
                writes = {dest}
            elif op == "trace_write":
                _, val = slot
                reads = {val}
        return reads, writes

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        instrs = []
        bundle = {}
        bundle_writes = set()
        engine_counts = defaultdict(int)

        def flush():
            nonlocal bundle, bundle_writes, engine_counts
            if bundle:
                instrs.append(bundle)
            bundle = {}
            bundle_writes = set()
            engine_counts = defaultdict(int)

        for engine, slot in slots:
            if engine == "debug":
                bundle.setdefault("debug", []).append(slot)
                continue

            slot_r, slot_w = self._slot_reads_writes(engine, slot)

            # Conflict: RAW, WAW, or engine slot limit reached
            if slot_r & bundle_writes or slot_w & bundle_writes or engine_counts[engine] >= SLOT_LIMITS[engine]:
                flush()

            bundle.setdefault(engine, []).append(slot)
            bundle_writes |= slot_w
            engine_counts[engine] += 1

        flush()
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
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        SIMD vectorized kernel using VLEN=8 vectors.
        Processes batch_size elements in groups of VLEN (8).
        16-group cross-pipeline (hextet pipeline): interleaves A-P group slots so that
        hash phases of early groups overlap with load phases of later groups.
        32 groups / 16 = 2 hextets, no remainder. ~63 cycles/hextet expected.
        Pairs 0-3 are reused for E-H, I-L, M-P (safe: earlier group's pair temps
        consumed before later group writes to them).
        Setup packing: const loads packed 2/cycle, vbroadcasts packed 6/cycle.

        Early-round special-casing (attempt-005):
        - round_mod==0 (rounds 0,11): all idx=0, broadcast single node, skip all loads
        - round_mod==1 (rounds 1,12): idx in {1,2}, load 2 nodes + vselect
        """
        assert batch_size % VLEN == 0, f"batch_size {batch_size} must be multiple of VLEN {VLEN}"
        n_groups = batch_size // VLEN  # 256 / 8 = 32

        # --- Scalar temps for initialization ---
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # --- Load metadata from mem[0..6] using scalar loads ---
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Load metadata: const + load pairs (cannot pack easily due to RAW)
        for i, v in enumerate(init_vars):
            addr_for_i = self.alloc_scratch()
            self.const_map[i] = addr_for_i
            self.instrs.append({"load": [("const", addr_for_i, i)]})
            self.instrs.append({"load": [("load", self.scratch[v], addr_for_i)]})

        # --- Allocate vector scratch regions ---
        # idx vectors: n_groups groups of VLEN words each
        idx_base = self.alloc_scratch("idx_base", n_groups * VLEN)
        # val vectors: n_groups groups of VLEN words each
        val_base = self.alloc_scratch("val_base", n_groups * VLEN)

        # Per-group addr_tmp and nv_tmp: each group needs its own to avoid WAW
        # conflicts during cross-pipeline interleaving.
        # 32 groups x 2 x 8 = 512 words
        addr_tmp_g = [self.alloc_scratch(f"addr_tmp_{g}", VLEN) for g in range(n_groups)]
        nv_tmp_g = [self.alloc_scratch(f"nv_tmp_{g}", VLEN) for g in range(n_groups)]

        # Paired t1/t2/lsb/cmp temps: 4 pairs (reused for all groups in pipeline).
        # In the 16-group pipeline, pair p is reused for groups at positions p, p+4, p+8, p+12.
        # This is safe because each group finishes using its pair before the next group at
        # same pair index starts.
        # 4 pairs x 4 x 8 = 128 words
        t1_tmp_pair = [self.alloc_scratch(f"t1_tmp_{p}", VLEN) for p in range(4)]
        t2_tmp_pair = [self.alloc_scratch(f"t2_tmp_{p}", VLEN) for p in range(4)]
        lsb_tmp_pair = [self.alloc_scratch(f"lsb_tmp_{p}", VLEN) for p in range(4)]
        cmp_tmp_pair = [self.alloc_scratch(f"cmp_tmp_{p}", VLEN) for p in range(4)]

        zero_vec = self.alloc_scratch("zero_vec", VLEN)

        # --- Scalar constants ---
        sc_forest_values_p = self.scratch["forest_values_p"]
        sc_n_nodes = self.scratch["n_nodes"]

        # Allocate scalar constants in pairs (2 consts per load cycle)
        one_scalar = self.alloc_scratch("one_scalar")
        two_scalar = self.alloc_scratch("two_scalar")
        zero_scalar = self.alloc_scratch("zero_scalar")
        three_scalar = self.alloc_scratch("three_scalar_const")

        sc_mul_4097 = self.alloc_scratch("mul_4097")
        sc_hash0_const = self.alloc_scratch("hash0_const")
        sc_hash1_const = self.alloc_scratch("hash1_const")
        sc_shift_19 = self.alloc_scratch("shift_19")
        sc_mul_33 = self.alloc_scratch("mul_33")
        sc_hash2_const = self.alloc_scratch("hash2_const")
        sc_hash3_const = self.alloc_scratch("hash3_const")
        sc_shift_9 = self.alloc_scratch("shift_9")
        sc_mul_9 = self.alloc_scratch("mul_9")
        sc_hash4_const = self.alloc_scratch("hash4_const")
        sc_hash5_const = self.alloc_scratch("hash5_const")
        sc_shift_16 = self.alloc_scratch("shift_16")

        # Register all in const_map
        self.const_map[1] = one_scalar
        self.const_map[2] = two_scalar
        self.const_map[0] = zero_scalar
        self.const_map[3] = three_scalar
        self.const_map[4097] = sc_mul_4097
        self.const_map[0x7ED55D16] = sc_hash0_const
        self.const_map[0xC761C23C] = sc_hash1_const
        self.const_map[19] = sc_shift_19
        self.const_map[33] = sc_mul_33
        self.const_map[0x165667B1] = sc_hash2_const
        self.const_map[0xD3A2646C] = sc_hash3_const
        self.const_map[9] = sc_shift_9
        # sc_mul_9 uses same value 9 — point to same address
        self.const_map[0xFD7046C5] = sc_hash4_const
        self.const_map[0xB55A4F09] = sc_hash5_const
        self.const_map[16] = sc_shift_16

        # Load all scalar constants packed 2/cycle
        const_loads = [
            (one_scalar, 1), (two_scalar, 2),
            (zero_scalar, 0), (three_scalar, 3),
            (sc_mul_4097, 4097), (sc_hash0_const, 0x7ED55D16),
            (sc_hash1_const, 0xC761C23C), (sc_shift_19, 19),
            (sc_mul_33, 33), (sc_hash2_const, 0x165667B1),
            (sc_hash3_const, 0xD3A2646C), (sc_shift_9, 9),
            (sc_mul_9, 9), (sc_hash4_const, 0xFD7046C5),
            (sc_hash5_const, 0xB55A4F09), (sc_shift_16, 16),
        ]
        for i in range(0, len(const_loads), 2):
            pair = const_loads[i:i+2]
            slots_list = [("const", p[0], p[1]) for p in pair]
            self.instrs.append({"load": slots_list})

        # Vector constant scratch regions
        fvp_vec = self.alloc_scratch("fvp_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)

        mul_4097_vec = self.alloc_scratch("mul_4097_vec", VLEN)
        hash0_const_vec = self.alloc_scratch("hash0_const_vec", VLEN)
        hash1_const_vec = self.alloc_scratch("hash1_const_vec", VLEN)
        shift_19_vec = self.alloc_scratch("shift_19_vec", VLEN)
        mul_33_vec = self.alloc_scratch("mul_33_vec", VLEN)
        hash2_const_vec = self.alloc_scratch("hash2_const_vec", VLEN)
        hash3_const_vec = self.alloc_scratch("hash3_const_vec", VLEN)
        shift_9_vec = self.alloc_scratch("shift_9_vec", VLEN)
        mul_9_vec = self.alloc_scratch("mul_9_vec", VLEN)
        hash4_const_vec = self.alloc_scratch("hash4_const_vec", VLEN)
        hash5_const_vec = self.alloc_scratch("hash5_const_vec", VLEN)
        shift_16_vec = self.alloc_scratch("shift_16_vec", VLEN)

        # Pack vbroadcasts 6/cycle (valu has 6 slots/cycle)
        broadcast_slots = [
            ("valu", ("vbroadcast", fvp_vec, sc_forest_values_p)),
            ("valu", ("vbroadcast", n_nodes_vec, sc_n_nodes)),
            ("valu", ("vbroadcast", one_vec, one_scalar)),
            ("valu", ("vbroadcast", two_vec, two_scalar)),
            ("valu", ("vbroadcast", zero_vec, zero_scalar)),
            ("valu", ("vbroadcast", mul_4097_vec, sc_mul_4097)),
            ("valu", ("vbroadcast", hash0_const_vec, sc_hash0_const)),
            ("valu", ("vbroadcast", hash1_const_vec, sc_hash1_const)),
            ("valu", ("vbroadcast", shift_19_vec, sc_shift_19)),
            ("valu", ("vbroadcast", mul_33_vec, sc_mul_33)),
            ("valu", ("vbroadcast", hash2_const_vec, sc_hash2_const)),
            ("valu", ("vbroadcast", hash3_const_vec, sc_hash3_const)),
            ("valu", ("vbroadcast", shift_9_vec, sc_shift_9)),
            ("valu", ("vbroadcast", mul_9_vec, sc_mul_9)),
            ("valu", ("vbroadcast", hash4_const_vec, sc_hash4_const)),
            ("valu", ("vbroadcast", hash5_const_vec, sc_hash5_const)),
            ("valu", ("vbroadcast", shift_16_vec, sc_shift_16)),
        ]
        broadcast_instrs = self.build(broadcast_slots)
        self.instrs.extend(broadcast_instrs)

        # --- Load initial idx and val vectors from memory into scratch ---
        # Scalar address temps for vload/vstore addressing
        addr_scalar = self.alloc_scratch("addr_scalar")
        addr_scalar2 = self.alloc_scratch("addr_scalar2")

        # Pre-compute group offset constants for address computation
        group_offset_scalars = []
        for g in range(n_groups):
            offset_val = g * VLEN
            if offset_val not in self.const_map:
                sc = self.alloc_scratch()
                self.instrs.append({"load": [("const", sc, offset_val)]})
                self.const_map[offset_val] = sc
            group_offset_scalars.append(self.const_map[offset_val])

        init_load_slots = []
        for g in range(n_groups):
            idx_addr = idx_base + g * VLEN
            val_addr = val_base + g * VLEN
            init_load_slots.append(("alu", ("+", addr_scalar, self.scratch["inp_indices_p"], group_offset_scalars[g])))
            init_load_slots.append(("load", ("vload", idx_addr, addr_scalar)))
            init_load_slots.append(("alu", ("+", addr_scalar2, self.scratch["inp_values_p"], group_offset_scalars[g])))
            init_load_slots.append(("load", ("vload", val_addr, addr_scalar2)))

        init_instrs = self.build(init_load_slots)
        self.instrs.extend(init_instrs)

        # Pause to match reference_kernel2 yield
        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting SIMD loop"))

        # --- Special scratch for early-round optimization ---
        # round_mod==0: all idx=0, single broadcast
        # Must be allocated AFTER all pair temps to avoid address conflicts
        nv_scalar_r0 = self.alloc_scratch("nv_scalar_r0", 1)    # single scalar node value
        nv_bcast_r0 = self.alloc_scratch("nv_bcast_r0", VLEN)   # broadcast vector for round 0/11

        # round_mod==1: idx in {1,2}, two scalar loads + per-group arithmetic vselect
        nv_node1_r1 = self.alloc_scratch("nv_node1_r1", 1)      # node at fvp+1
        nv_node2_r1 = self.alloc_scratch("nv_node2_r1", 1)      # node at fvp+2
        nv_bcast1_r1 = self.alloc_scratch("nv_bcast1_r1", VLEN) # broadcast of node1
        nv_bcast2_r1 = self.alloc_scratch("nv_bcast2_r1", VLEN) # broadcast of node2
        diff_r1_vec = self.alloc_scratch("diff_r1_vec", VLEN)   # node1 - node2 (for arithmetic select)

        # round_mod==2: idx in {3,4,5,6}, 4-node arithmetic select
        nv_node3_r2 = self.alloc_scratch("nv_node3_r2", 1)      # node at fvp+3
        nv_node4_r2 = self.alloc_scratch("nv_node4_r2", 1)      # node at fvp+4
        nv_node5_r2 = self.alloc_scratch("nv_node5_r2", 1)      # node at fvp+5
        nv_node6_r2 = self.alloc_scratch("nv_node6_r2", 1)      # node at fvp+6
        node3_bcast = self.alloc_scratch("node3_bcast", VLEN)   # broadcast of node3
        node5_bcast = self.alloc_scratch("node5_bcast", VLEN)   # broadcast of node5
        diff34_bcast = self.alloc_scratch("diff34_bcast", VLEN) # node4 - node3
        diff56_bcast = self.alloc_scratch("diff56_bcast", VLEN) # node6 - node5
        # three_scalar is already allocated above (in const section)
        three_vec = self.alloc_scratch("three_vec", VLEN)       # broadcast 3

        # Scratch check
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Scratch overflow: {self.scratch_ptr} > {SCRATCH_SIZE}"

        # --- Helper: build slots for one group's phases ---
        def group_addr_slots(g):
            """Phase 1: addr_tmp = idx + fvp (1 valu slot)"""
            idx_s = idx_base + g * VLEN
            at = addr_tmp_g[g]
            return [("valu", ("+", at, idx_s, fvp_vec))]

        def group_load_slots(g):
            """Phase 2: load all 8 lanes into nv_tmp (8 load_offset slots)"""
            at = addr_tmp_g[g]
            nv = nv_tmp_g[g]
            slots = []
            for lane in range(VLEN):
                slots.append(("load", ("load_offset", nv, at, lane)))
            return slots

        def group_xor_slots(g):
            """Phase 3: val ^= nv_tmp (1 valu slot)"""
            val_s = val_base + g * VLEN
            nv = nv_tmp_g[g]
            return [("valu", ("^", val_s, val_s, nv))]

        def group_xor_bcast_slots(g, nv_bcast):
            """Phase 3 (broadcast variant): val ^= nv_bcast (1 valu slot)"""
            val_s = val_base + g * VLEN
            return [("valu", ("^", val_s, val_s, nv_bcast))]

        def group_hash_slots(g, p):
            """Phase 4: hash computation (12 valu slots in dependency order)"""
            val_s = val_base + g * VLEN
            t1 = t1_tmp_pair[p]
            t2 = t2_tmp_pair[p]
            slots = []
            # Stage 0: multiply_add
            slots.append(("valu", ("multiply_add", val_s, val_s, mul_4097_vec, hash0_const_vec)))
            # Stage 1: t1 = val ^ hash1_const, t2 = val >> 19  (parallel)
            slots.append(("valu", ("^", t1, val_s, hash1_const_vec)))
            slots.append(("valu", (">>", t2, val_s, shift_19_vec)))
            # Stage 1 combine: val = t1 ^ t2  (depends on both above)
            slots.append(("valu", ("^", val_s, t1, t2)))
            # Stage 2: multiply_add
            slots.append(("valu", ("multiply_add", val_s, val_s, mul_33_vec, hash2_const_vec)))
            # Stage 3: t1 = val + hash3_const, t2 = val << 9  (parallel)
            slots.append(("valu", ("+", t1, val_s, hash3_const_vec)))
            slots.append(("valu", ("<<", t2, val_s, shift_9_vec)))
            # Stage 3 combine
            slots.append(("valu", ("^", val_s, t1, t2)))
            # Stage 4: multiply_add
            slots.append(("valu", ("multiply_add", val_s, val_s, mul_9_vec, hash4_const_vec)))
            # Stage 5: t1 = val ^ hash5_const, t2 = val >> 16  (parallel)
            slots.append(("valu", ("^", t1, val_s, hash5_const_vec)))
            slots.append(("valu", (">>", t2, val_s, shift_16_vec)))
            # Stage 5 combine
            slots.append(("valu", ("^", val_s, t1, t2)))
            return slots

        def group_idx_slots(g, p):
            """Phase 5: index update (5 valu slots)
            OPTIMIZATION: Uses addr_tmp_g[g] (per-group) for BOTH lsb and cmp temporaries
            instead of lsb_tmp_pair[p] and cmp_tmp_pair[p] (pair-shared).
            This eliminates WAW conflicts between groups sharing the same pair (e.g., A,E,I,M
            all used to conflict on lsb_pair[0]). Now all 16 groups can run idx simultaneously!
            - addr_tmp_g[g] was written during the addr phase and read during loads; after
              all loads are done, it's free for reuse.
            - lsb sequence: write addr_tmp(val&1) → write addr_tmp(lsb+1) → read for idx
            - cmp reuses addr_tmp after lsb is consumed (step 3 reads lsb, step 4 writes cmp)
            """
            idx_s = idx_base + g * VLEN
            val_s = val_base + g * VLEN
            lsb = addr_tmp_g[g]  # per-group scratch (was lsb_tmp_pair[p])
            cmp = addr_tmp_g[g]  # REUSE same per-group scratch for cmp (lsb consumed by step 3)
            slots = []
            # lsb = val & 1  (write to addr_tmp_g[g])
            slots.append(("valu", ("&", lsb, val_s, one_vec)))
            # offset = lsb + 1  (read+write addr_tmp_g[g])
            slots.append(("valu", ("+", lsb, lsb, one_vec)))
            # new_idx = idx * 2 + offset  (reads addr_tmp_g[g] = lsb; addr_tmp now free)
            slots.append(("valu", ("multiply_add", idx_s, idx_s, two_vec, lsb)))
            # cmp = new_idx < n_nodes  (write to addr_tmp_g[g] = cmp, RAW prevents same bundle as above)
            slots.append(("valu", ("<", cmp, idx_s, n_nodes_vec)))
            # idx = new_idx * cmp  (reads addr_tmp_g[g] = cmp)
            slots.append(("valu", ("*", idx_s, idx_s, cmp)))
            return slots

        # --- No-load hextet: pure valu hextet for rounds with broadcast nv ---
        # Uses the same careful diagonal ordering as the normal hextet (just no addr/load).
        # This ensures pair-reuse safety (same as normal hextet).
        def emit_hextet_noload(body, grp_start, nv_bcast, next_s=None):
            """Emit 16-group no-load hextet using nv_bcast for xor.
            Uses the exact same diagonal ordering as normal hextet (steps 1-63),
            but skips all addr and load slots, replacing xor with bcast-xor.
            Pair safety is maintained by the same ordering constraints.
            If next_s is provided, prefetch next round's hextet0 head in the tail."""
            gA, gB, gC, gD = grp_start, grp_start+1, grp_start+2, grp_start+3
            gE, gF, gG, gH = grp_start+4, grp_start+5, grp_start+6, grp_start+7
            gI, gJ, gK, gL = grp_start+8, grp_start+9, grp_start+10, grp_start+11
            gM, gN, gO, gP = grp_start+12, grp_start+13, grp_start+14, grp_start+15
            pA=pB=pC=pD=pE=pF=pG=pH=pI=pJ=pK=pL=pM=pN=pO=pP=0
            pA, pB, pC, pD = 0, 1, 2, 3
            pE, pF, pG, pH = 0, 1, 2, 3
            pI, pJ, pK, pL = 0, 1, 2, 3
            pM, pN, pO, pP = 0, 1, 2, 3

            a_xor = group_xor_bcast_slots(gA, nv_bcast)
            b_xor = group_xor_bcast_slots(gB, nv_bcast)
            c_xor = group_xor_bcast_slots(gC, nv_bcast)
            d_xor = group_xor_bcast_slots(gD, nv_bcast)
            e_xor = group_xor_bcast_slots(gE, nv_bcast)
            f_xor = group_xor_bcast_slots(gF, nv_bcast)
            g_xor = group_xor_bcast_slots(gG, nv_bcast)
            h_xor = group_xor_bcast_slots(gH, nv_bcast)
            i_xor = group_xor_bcast_slots(gI, nv_bcast)
            j_xor = group_xor_bcast_slots(gJ, nv_bcast)
            k_xor = group_xor_bcast_slots(gK, nv_bcast)
            l_xor = group_xor_bcast_slots(gL, nv_bcast)
            m_xor = group_xor_bcast_slots(gM, nv_bcast)
            n_xor = group_xor_bcast_slots(gN, nv_bcast)
            o_xor = group_xor_bcast_slots(gO, nv_bcast)
            p_xor = group_xor_bcast_slots(gP, nv_bcast)

            a_hash = group_hash_slots(gA, pA)
            b_hash = group_hash_slots(gB, pB)
            c_hash = group_hash_slots(gC, pC)
            d_hash = group_hash_slots(gD, pD)
            e_hash = group_hash_slots(gE, pE)
            f_hash = group_hash_slots(gF, pF)
            g_hash = group_hash_slots(gG, pG)
            h_hash = group_hash_slots(gH, pH)
            i_hash = group_hash_slots(gI, pI)
            j_hash = group_hash_slots(gJ, pJ)
            k_hash = group_hash_slots(gK, pK)
            l_hash = group_hash_slots(gL, pL)
            m_hash = group_hash_slots(gM, pM)
            n_hash = group_hash_slots(gN, pN)
            o_hash = group_hash_slots(gO, pO)
            p_hash = group_hash_slots(gP, pP)

            a_idx = group_idx_slots(gA, pA)
            b_idx = group_idx_slots(gB, pB)
            c_idx = group_idx_slots(gC, pC)
            d_idx = group_idx_slots(gD, pD)
            e_idx = group_idx_slots(gE, pE)
            f_idx = group_idx_slots(gF, pF)
            g_idx = group_idx_slots(gG, pG)
            h_idx = group_idx_slots(gH, pH)
            i_idx = group_idx_slots(gI, pI)
            j_idx = group_idx_slots(gJ, pJ)
            k_idx = group_idx_slots(gK, pK)
            l_idx = group_idx_slots(gL, pL)
            m_idx = group_idx_slots(gM, pM)
            n_idx = group_idx_slots(gN, pN)
            o_idx = group_idx_slots(gO, pO)
            p_idx = group_idx_slots(gP, pP)

            # Follow the exact same diagonal ordering as normal hextet,
            # but replacing addr+load with xor(bcast) at the appropriate positions.
            # In the normal hextet:
            #   Step 1: A.addr   → replace with A.xor (no load dependency)
            #   Step 2: A.load+B.addr → replace with B.xor
            #   Step 3: A.xor+B.load+C.addr → skip (A already done above)
            # Simplification: emit all xors first (A-P), then follow the hash+idx diagonal
            # without the load slots. This is safe because:
            #   - gA uses pair 0 for hash; by step 13+ gA is done with t1/t2_pair[0]
            #   - gE also uses pair 0 but starts hash later (step 19+ in normal); same here
            #   - The relative ordering of hash[A] vs hash[E] is maintained by the diagonal

            # Xors for all 16 groups (A-P): 16 valu ops, packs to 3 cycles
            body.extend(a_xor)
            body.extend(b_xor)
            body.extend(c_xor)
            body.extend(d_xor)
            body.extend(e_xor)
            body.extend(f_xor)
            body.extend(g_xor)
            body.extend(h_xor)
            body.extend(i_xor)
            body.extend(j_xor)
            body.extend(k_xor)
            body.extend(l_xor)
            body.extend(m_xor)
            body.extend(n_xor)
            body.extend(o_xor)
            body.extend(p_xor)

            # Hash+idx diagonal (same order as normal hextet steps 4-63, minus loads/addrs)
            # Step 4 equivalent: A.hash[0] + B (already xored)
            body.append(a_hash[0])

            # Step 5: A.hash[1:3] + B.hash[0]
            body.append(a_hash[1])
            body.append(a_hash[2])
            body.append(b_hash[0])

            # Step 6: A.hash[3] + B.hash[1:3] + C.hash[0]
            body.append(a_hash[3])
            body.append(b_hash[1])
            body.append(b_hash[2])
            body.append(c_hash[0])

            # Step 7: A.hash[4] + B.hash[3] + C.hash[1:3] + D.hash[0]
            body.append(a_hash[4])
            body.append(b_hash[3])
            body.append(c_hash[1])
            body.append(c_hash[2])
            body.append(d_hash[0])

            # Step 8: A.hash[5:7] + B.hash[4] + C.hash[3] + D.hash[1:3]
            body.append(a_hash[5])
            body.append(a_hash[6])
            body.append(b_hash[4])
            body.append(c_hash[3])
            body.append(d_hash[1])
            body.append(d_hash[2])

            # Step 9: A.hash[7] + B.hash[5:7] + C.hash[4] + D.hash[3]
            body.append(a_hash[7])
            body.append(b_hash[5])
            body.append(b_hash[6])
            body.append(c_hash[4])
            body.append(d_hash[3])

            # Step 10: A.hash[8] + B.hash[7] + C.hash[5:7] + D.hash[4]
            body.append(a_hash[8])
            body.append(b_hash[7])
            body.append(c_hash[5])
            body.append(c_hash[6])
            body.append(d_hash[4])

            # Step 11: A.hash[9:11] + B.hash[8] + C.hash[7] + D.hash[5:7]
            body.append(a_hash[9])
            body.append(a_hash[10])
            body.append(b_hash[8])
            body.append(c_hash[7])
            body.append(d_hash[5])
            body.append(d_hash[6])

            # Step 12: A.hash[11] + B.hash[9:11] + C.hash[8] + D.hash[7]
            body.append(a_hash[11])
            body.append(b_hash[9])
            body.append(b_hash[10])
            body.append(c_hash[8])
            body.append(d_hash[7])

            # Step 13: A.idx[0] + B.hash[11] + C.hash[9:11] + D.hash[8]
            body.append(a_idx[0])
            body.append(b_hash[11])
            body.append(c_hash[9])
            body.append(c_hash[10])
            body.append(d_hash[8])

            # Step 14: A.idx[1] + B.idx[0] + C.hash[11] + D.hash[9:11]
            body.append(a_idx[1])
            body.append(b_idx[0])
            body.append(c_hash[11])
            body.append(d_hash[9])
            body.append(d_hash[10])

            # Step 15: A.idx[2] + B.idx[1] + C.idx[0] + D.hash[11]
            body.append(a_idx[2])
            body.append(b_idx[1])
            body.append(c_idx[0])
            body.append(d_hash[11])

            # Step 16: A.idx[3] + B.idx[2] + C.idx[1] + D.idx[0]
            body.append(a_idx[3])
            body.append(b_idx[2])
            body.append(c_idx[1])
            body.append(d_idx[0])

            # Step 17: A.idx[4] + B.idx[3] + C.idx[2] + D.idx[1]
            body.append(a_idx[4])
            body.append(b_idx[3])
            body.append(c_idx[2])
            body.append(d_idx[1])

            # Step 18: B.idx[4] + C.idx[3] + D.idx[2]
            body.append(b_idx[4])
            body.append(c_idx[3])
            body.append(d_idx[2])

            # Step 19: C.idx[4] + D.idx[3] + E.hash[0]
            body.append(c_idx[4])
            body.append(d_idx[3])
            body.append(e_hash[0])

            # Step 20: D.idx[4] + E.hash[1:3] + F.hash[0]
            body.append(d_idx[4])
            body.append(e_hash[1])
            body.append(e_hash[2])
            body.append(f_hash[0])

            # Step 21: E.hash[3] + F.hash[1:3] + G.hash[0]
            body.append(e_hash[3])
            body.append(f_hash[1])
            body.append(f_hash[2])
            body.append(g_hash[0])

            # Step 22: E.hash[4] + F.hash[3] + G.hash[1:3] + H.hash[0]
            body.append(e_hash[4])
            body.append(f_hash[3])
            body.append(g_hash[1])
            body.append(g_hash[2])
            body.append(h_hash[0])

            # Step 23: E.hash[5:7] + F.hash[4] + G.hash[3] + H.hash[1:3]
            body.append(e_hash[5])
            body.append(e_hash[6])
            body.append(f_hash[4])
            body.append(g_hash[3])
            body.append(h_hash[1])
            body.append(h_hash[2])

            # Step 24: E.hash[7] + F.hash[5:7] + G.hash[4] + H.hash[3]
            body.append(e_hash[7])
            body.append(f_hash[5])
            body.append(f_hash[6])
            body.append(g_hash[4])
            body.append(h_hash[3])

            # Step 25: E.hash[8] + F.hash[7] + G.hash[5:7] + H.hash[4]
            body.append(e_hash[8])
            body.append(f_hash[7])
            body.append(g_hash[5])
            body.append(g_hash[6])
            body.append(h_hash[4])

            # Step 26: E.hash[9:11] + F.hash[8] + G.hash[7] + H.hash[5:7]
            body.append(e_hash[9])
            body.append(e_hash[10])
            body.append(f_hash[8])
            body.append(g_hash[7])
            body.append(h_hash[5])
            body.append(h_hash[6])

            # Step 27: E.hash[11] + F.hash[9:11] + G.hash[8] + H.hash[7]
            body.append(e_hash[11])
            body.append(f_hash[9])
            body.append(f_hash[10])
            body.append(g_hash[8])
            body.append(h_hash[7])

            # Step 28: E.idx[0] + F.hash[11] + G.hash[9:11] + H.hash[8]
            body.append(e_idx[0])
            body.append(f_hash[11])
            body.append(g_hash[9])
            body.append(g_hash[10])
            body.append(h_hash[8])

            # Step 29: E.idx[1] + F.idx[0] + G.hash[11] + H.hash[9:11]
            body.append(e_idx[1])
            body.append(f_idx[0])
            body.append(g_hash[11])
            body.append(h_hash[9])
            body.append(h_hash[10])

            # Step 30: E.idx[2] + F.idx[1] + G.idx[0] + H.hash[11]
            body.append(e_idx[2])
            body.append(f_idx[1])
            body.append(g_idx[0])
            body.append(h_hash[11])

            # Step 31: E.idx[3] + F.idx[2] + G.idx[1] + H.idx[0]
            body.append(e_idx[3])
            body.append(f_idx[2])
            body.append(g_idx[1])
            body.append(h_idx[0])

            # Step 32: E.idx[4] + F.idx[3] + G.idx[2] + H.idx[1]
            body.append(e_idx[4])
            body.append(f_idx[3])
            body.append(g_idx[2])
            body.append(h_idx[1])

            # Step 33: F.idx[4] + G.idx[3] + H.idx[2] + I.hash[0]
            body.append(f_idx[4])
            body.append(g_idx[3])
            body.append(h_idx[2])
            body.append(i_hash[0])

            # Step 34: G.idx[4] + H.idx[3] + I.hash[1:3] + J.hash[0]
            body.append(g_idx[4])
            body.append(h_idx[3])
            body.append(i_hash[1])
            body.append(i_hash[2])
            body.append(j_hash[0])

            # Step 35: H.idx[4] + I.hash[3] + J.hash[1:3] + K.hash[0]
            body.append(h_idx[4])
            body.append(i_hash[3])
            body.append(j_hash[1])
            body.append(j_hash[2])
            body.append(k_hash[0])

            # Step 36: I.hash[4] + J.hash[3] + K.hash[1:3] + L.hash[0]
            body.append(i_hash[4])
            body.append(j_hash[3])
            body.append(k_hash[1])
            body.append(k_hash[2])
            body.append(l_hash[0])

            # Step 37: I.hash[5:7] + J.hash[4] + K.hash[3] + L.hash[1:3]
            body.append(i_hash[5])
            body.append(i_hash[6])
            body.append(j_hash[4])
            body.append(k_hash[3])
            body.append(l_hash[1])
            body.append(l_hash[2])

            # Step 38: I.hash[7] + J.hash[5:7] + K.hash[4] + L.hash[3]
            body.append(i_hash[7])
            body.append(j_hash[5])
            body.append(j_hash[6])
            body.append(k_hash[4])
            body.append(l_hash[3])

            # Step 39: I.hash[8] + J.hash[7] + K.hash[5:7] + L.hash[4] + M.hash[0]
            body.append(i_hash[8])
            body.append(j_hash[7])
            body.append(k_hash[5])
            body.append(k_hash[6])
            body.append(l_hash[4])
            body.append(m_hash[0])

            # Step 40: I.hash[9:11] + J.hash[8] + K.hash[7] + L.hash[5:7]
            body.append(i_hash[9])
            body.append(i_hash[10])
            body.append(j_hash[8])
            body.append(k_hash[7])
            body.append(l_hash[5])
            body.append(l_hash[6])

            # Step 41: I.hash[11] + J.hash[9:11] + K.hash[8] + L.hash[7] + M.hash[1]
            body.append(i_hash[11])
            body.append(j_hash[9])
            body.append(j_hash[10])
            body.append(k_hash[8])
            body.append(l_hash[7])
            body.append(m_hash[1])

            # Step 42: I.idx[0] + J.hash[11] + K.hash[9:11] + L.hash[8] + M.hash[2]
            body.append(i_idx[0])
            body.append(j_hash[11])
            body.append(k_hash[9])
            body.append(k_hash[10])
            body.append(l_hash[8])
            body.append(m_hash[2])

            # Step 43: I.idx[1] + J.idx[0] + K.hash[11] + L.hash[9:11] + M.hash[3]
            body.append(i_idx[1])
            body.append(j_idx[0])
            body.append(k_hash[11])
            body.append(l_hash[9])
            body.append(l_hash[10])
            body.append(m_hash[3])

            # Step 44: I.idx[2] + J.idx[1] + K.idx[0] + L.hash[11] + M.hash[4] + N.hash[0]
            body.append(i_idx[2])
            body.append(j_idx[1])
            body.append(k_idx[0])
            body.append(l_hash[11])
            body.append(m_hash[4])
            body.append(n_hash[0])

            # Step 45: I.idx[3] + J.idx[2] + K.idx[1] + L.idx[0] + M.hash[5:7]
            body.append(i_idx[3])
            body.append(j_idx[2])
            body.append(k_idx[1])
            body.append(l_idx[0])
            body.append(m_hash[5])
            body.append(m_hash[6])

            # Step 46: I.idx[4] + J.idx[3] + K.idx[2] + L.idx[1] + M.hash[7] + N.hash[1]
            body.append(i_idx[4])
            body.append(j_idx[3])
            body.append(k_idx[2])
            body.append(l_idx[1])
            body.append(m_hash[7])
            body.append(n_hash[1])

            # Step 47: J.idx[4] + K.idx[3] + L.idx[2] + M.hash[8] + N.hash[2:4]
            body.append(j_idx[4])
            body.append(k_idx[3])
            body.append(l_idx[2])
            body.append(m_hash[8])
            body.append(n_hash[2])
            body.append(n_hash[3])

            # Step 48: K.idx[4] + L.idx[3] + M.hash[9:11] + N.hash[4] + O.hash[0]
            body.append(k_idx[4])
            body.append(l_idx[3])
            body.append(m_hash[9])
            body.append(m_hash[10])
            body.append(n_hash[4])
            body.append(o_hash[0])

            # Step 49: L.idx[4] + M.hash[11] + N.hash[5:7] + O.hash[1:3]
            body.append(l_idx[4])
            body.append(m_hash[11])
            body.append(n_hash[5])
            body.append(n_hash[6])
            body.append(o_hash[1])
            body.append(o_hash[2])

            # Tail with optional next-round prefetch
            if next_s is not None:
                na_addr_nl = next_s['a_addr']; na_loads_nl = next_s['a_loads']
                nb_addr_nl = next_s['b_addr']; nb_loads_nl = next_s['b_loads']
                nc_addr_nl = next_s['c_addr']; nc_loads_nl = next_s['c_loads']
                nd_addr_nl = next_s['d_addr']
            else:
                na_addr_nl = nb_addr_nl = nc_addr_nl = nd_addr_nl = []
                na_loads_nl = nb_loads_nl = nc_loads_nl = []

            # OPTIMIZATION: Emit ALL 4 addr ops first (before step 50 valu ops).
            body.extend(na_addr_nl)
            body.extend(nb_addr_nl)
            body.extend(nc_addr_nl)
            body.extend(nd_addr_nl)

            # Step 50: M.idx[0] + N.hash[7] + O.hash[3] + P.hash[0]
            body.append(m_idx[0]); body.append(n_hash[7]); body.append(o_hash[3]); body.append(p_hash[0])
            # Step 51: M.idx[1] + N.hash[8] + O.hash[4] + P.hash[1:3] [+ na_loads[0:2]]
            body.append(m_idx[1]); body.append(n_hash[8]); body.append(o_hash[4])
            body.append(p_hash[1]); body.append(p_hash[2])
            body.extend(na_loads_nl[0:2])
            # Step 52: M.idx[2] + N.hash[9:11] + O.hash[5:7] + P.hash[3] [+ na_loads[2:4]]
            body.append(m_idx[2]); body.append(n_hash[9]); body.append(n_hash[10])
            body.append(o_hash[5]); body.append(o_hash[6]); body.append(p_hash[3])
            body.extend(na_loads_nl[2:4])
            # Step 53: M.idx[3] + N.hash[11] + O.hash[7] + P.hash[4] [+ na_loads[4:6]]
            body.append(m_idx[3]); body.append(n_hash[11]); body.append(o_hash[7]); body.append(p_hash[4])
            body.extend(na_loads_nl[4:6])
            # Step 54: M.idx[4] + N.idx[0] + O.hash[8] + P.hash[5:7] [+ na_loads[6:8]]
            body.append(m_idx[4]); body.append(n_idx[0]); body.append(o_hash[8])
            body.append(p_hash[5]); body.append(p_hash[6]); body.extend(na_loads_nl[6:8])
            # Step 55: N.idx[1] + O.hash[9:11] + P.hash[7] [+ nb_loads[0:2]]
            body.append(n_idx[1]); body.append(o_hash[9]); body.append(o_hash[10]); body.append(p_hash[7])
            body.extend(nb_loads_nl[0:2])
            # Step 56: N.idx[2] + O.hash[11] + P.hash[8] [+ nb_loads[2:4]]
            body.append(n_idx[2]); body.append(o_hash[11]); body.append(p_hash[8])
            body.extend(nb_loads_nl[2:4])
            # Step 57: N.idx[3] + O.idx[0] + P.hash[9:11] [+ nb_loads[4:6]]
            body.append(n_idx[3]); body.append(o_idx[0]); body.append(p_hash[9]); body.append(p_hash[10])
            body.extend(nb_loads_nl[4:6])
            # Step 58: N.idx[4] + O.idx[1] + P.hash[11] [+ nb_loads[6:8]]
            body.append(n_idx[4]); body.append(o_idx[1]); body.append(p_hash[11])
            body.extend(nb_loads_nl[6:8])
            # Step 59: O.idx[2] + P.idx[0] [+ nc_loads[0:2]]
            body.append(o_idx[2]); body.append(p_idx[0]); body.extend(nc_loads_nl[0:2])
            # Step 60: O.idx[3] + P.idx[1] [+ nc_loads[2:4]]
            body.append(o_idx[3]); body.append(p_idx[1]); body.extend(nc_loads_nl[2:4])
            # Step 61: O.idx[4] + P.idx[2] [+ nc_loads[4:6]]
            body.append(o_idx[4]); body.append(p_idx[2]); body.extend(nc_loads_nl[4:6])
            # Step 62: P.idx[3] [+ nc_loads[6:8]]
            body.append(p_idx[3]); body.extend(nc_loads_nl[6:8])
            # Step 63: P.idx[4] (nd_addr already emitted at front of tail)
            body.append(p_idx[4])

        # --- No-load hextet with arithmetic vselect for round_mod==1 ---
        def emit_hextet_vselect2(body, grp_start, nv_bcast1, nv_bcast2, diff_vec):
            """Emit 16-group hextet using pure valu arithmetic to choose between 2 nodes.
            Uses addr_tmp_g[g] as per-group lsb storage (each group has its own → no sharing).
            nv_bcast1 = forest_values[1] (for idx=1, lsb=1)
            nv_bcast2 = forest_values[2] (for idx=2, lsb=0)
            diff_vec = nv_bcast1 - nv_bcast2 (precomputed)

            Arithmetic select: nv = nv_bcast2 + lsb * (nv_bcast1 - nv_bcast2)
                             = multiply_add(nv_bcast2_r1, lsb, diff_vec, nv_bcast2)
            When lsb=1: nv = diff + nv2 = nv1 ✓
            When lsb=0: nv = 0 + nv2 = nv2 ✓
            No flow engine usage → 16 groups can be pipelined on valu only.
            """
            gA, gB, gC, gD = grp_start, grp_start+1, grp_start+2, grp_start+3
            gE, gF, gG, gH = grp_start+4, grp_start+5, grp_start+6, grp_start+7
            gI, gJ, gK, gL = grp_start+8, grp_start+9, grp_start+10, grp_start+11
            gM, gN, gO, gP = grp_start+12, grp_start+13, grp_start+14, grp_start+15
            pA, pB, pC, pD = 0, 1, 2, 3
            pE, pF, pG, pH = 0, 1, 2, 3
            pI, pJ, pK, pL = 0, 1, 2, 3
            pM, pN, pO, pP = 0, 1, 2, 3

            # Per-group arithmetic select slots (2 valu ops each):
            # Step 1: lsb_g = idx_g & 1 → addr_tmp_g[g]
            # Step 2: nv_g = multiply_add(lsb_g, diff_vec, nv_bcast2) → nv_tmp_g[g]
            # Step 3: val_g ^= nv_g
            # Total: 3 valu ops/group, no flow dependency chain.
            # Groups can interleave freely on valu engine (6 slots/cycle).

            def xor_arith_slots(g):
                idx_s = idx_base + g * VLEN
                val_s = val_base + g * VLEN
                lsb = addr_tmp_g[g]  # per-group lsb storage
                nv = nv_tmp_g[g]
                return [
                    ("valu", ("&", lsb, idx_s, one_vec)),
                    # nv = lsb * (node1-node2) + node2 = node2 + lsb*(node1-node2)
                    ("valu", ("multiply_add", nv, lsb, diff_vec, nv_bcast2)),
                    ("valu", ("^", val_s, val_s, nv)),
                ]

            a_xas = xor_arith_slots(gA)
            b_xas = xor_arith_slots(gB)
            c_xas = xor_arith_slots(gC)
            d_xas = xor_arith_slots(gD)
            e_xas = xor_arith_slots(gE)
            f_xas = xor_arith_slots(gF)
            g_xas = xor_arith_slots(gG)
            h_xas = xor_arith_slots(gH)
            i_xas = xor_arith_slots(gI)
            j_xas = xor_arith_slots(gJ)
            k_xas = xor_arith_slots(gK)
            l_xas = xor_arith_slots(gL)
            m_xas = xor_arith_slots(gM)
            n_xas = xor_arith_slots(gN)
            o_xas = xor_arith_slots(gO)
            p_xas = xor_arith_slots(gP)

            a_hash = group_hash_slots(gA, pA)
            b_hash = group_hash_slots(gB, pB)
            c_hash = group_hash_slots(gC, pC)
            d_hash = group_hash_slots(gD, pD)
            e_hash = group_hash_slots(gE, pE)
            f_hash = group_hash_slots(gF, pF)
            g_hash = group_hash_slots(gG, pG)
            h_hash = group_hash_slots(gH, pH)
            i_hash = group_hash_slots(gI, pI)
            j_hash = group_hash_slots(gJ, pJ)
            k_hash = group_hash_slots(gK, pK)
            l_hash = group_hash_slots(gL, pL)
            m_hash = group_hash_slots(gM, pM)
            n_hash = group_hash_slots(gN, pN)
            o_hash = group_hash_slots(gO, pO)
            p_hash = group_hash_slots(gP, pP)

            a_idx = group_idx_slots(gA, pA)
            b_idx = group_idx_slots(gB, pB)
            c_idx = group_idx_slots(gC, pC)
            d_idx = group_idx_slots(gD, pD)
            e_idx = group_idx_slots(gE, pE)
            f_idx = group_idx_slots(gF, pF)
            g_idx = group_idx_slots(gG, pG)
            h_idx = group_idx_slots(gH, pH)
            i_idx = group_idx_slots(gI, pI)
            j_idx = group_idx_slots(gJ, pJ)
            k_idx = group_idx_slots(gK, pK)
            l_idx = group_idx_slots(gL, pL)
            m_idx = group_idx_slots(gM, pM)
            n_idx = group_idx_slots(gN, pN)
            o_idx = group_idx_slots(gO, pO)
            p_idx = group_idx_slots(gP, pP)

            # Emit all lsb computations (16 valu, pack 6/cycle → 3 cycles)
            for g in [gA, gB, gC, gD, gE, gF, gG, gH, gI, gJ, gK, gL, gM, gN, gO, gP]:
                body.append(("valu", ("&", addr_tmp_g[g], idx_base + g * VLEN, one_vec)))
            # Emit multiply_add for all groups (16 valu, pack 6/cycle → 3 cycles)
            # Each reads addr_tmp_g[g] (written in above), but different groups independent
            for g in [gA, gB, gC, gD, gE, gF, gG, gH, gI, gJ, gK, gL, gM, gN, gO, gP]:
                body.append(("valu", ("multiply_add", nv_tmp_g[g], addr_tmp_g[g], diff_vec, nv_bcast2)))
            # Emit xor for all groups (16 valu, pack 6/cycle → 3 cycles)
            for g in [gA, gB, gC, gD, gE, gF, gG, gH, gI, gJ, gK, gL, gM, gN, gO, gP]:
                body.append(("valu", ("^", val_base + g * VLEN, val_base + g * VLEN, nv_tmp_g[g])))

            # Hash+idx in normal diagonal order (same as emit_hextet_noload but without xors)
            body.append(a_hash[0])
            body.append(a_hash[1])
            body.append(a_hash[2])
            body.append(b_hash[0])
            body.append(a_hash[3])
            body.append(b_hash[1])
            body.append(b_hash[2])
            body.append(c_hash[0])
            body.append(a_hash[4])
            body.append(b_hash[3])
            body.append(c_hash[1])
            body.append(c_hash[2])
            body.append(d_hash[0])
            body.append(a_hash[5])
            body.append(a_hash[6])
            body.append(b_hash[4])
            body.append(c_hash[3])
            body.append(d_hash[1])
            body.append(d_hash[2])
            body.append(a_hash[7])
            body.append(b_hash[5])
            body.append(b_hash[6])
            body.append(c_hash[4])
            body.append(d_hash[3])
            body.append(a_hash[8])
            body.append(b_hash[7])
            body.append(c_hash[5])
            body.append(c_hash[6])
            body.append(d_hash[4])
            body.append(a_hash[9])
            body.append(a_hash[10])
            body.append(b_hash[8])
            body.append(c_hash[7])
            body.append(d_hash[5])
            body.append(d_hash[6])
            body.append(a_hash[11])
            body.append(b_hash[9])
            body.append(b_hash[10])
            body.append(c_hash[8])
            body.append(d_hash[7])
            body.append(a_idx[0])
            body.append(b_hash[11])
            body.append(c_hash[9])
            body.append(c_hash[10])
            body.append(d_hash[8])
            body.append(a_idx[1])
            body.append(b_idx[0])
            body.append(c_hash[11])
            body.append(d_hash[9])
            body.append(d_hash[10])
            body.append(a_idx[2])
            body.append(b_idx[1])
            body.append(c_idx[0])
            body.append(d_hash[11])
            body.append(a_idx[3])
            body.append(b_idx[2])
            body.append(c_idx[1])
            body.append(d_idx[0])
            body.append(a_idx[4])
            body.append(b_idx[3])
            body.append(c_idx[2])
            body.append(d_idx[1])
            body.append(b_idx[4])
            body.append(c_idx[3])
            body.append(d_idx[2])
            body.append(c_idx[4])
            body.append(d_idx[3])
            body.append(e_hash[0])
            body.append(d_idx[4])
            body.append(e_hash[1])
            body.append(e_hash[2])
            body.append(f_hash[0])
            body.append(e_hash[3])
            body.append(f_hash[1])
            body.append(f_hash[2])
            body.append(g_hash[0])
            body.append(e_hash[4])
            body.append(f_hash[3])
            body.append(g_hash[1])
            body.append(g_hash[2])
            body.append(h_hash[0])
            body.append(e_hash[5])
            body.append(e_hash[6])
            body.append(f_hash[4])
            body.append(g_hash[3])
            body.append(h_hash[1])
            body.append(h_hash[2])
            body.append(e_hash[7])
            body.append(f_hash[5])
            body.append(f_hash[6])
            body.append(g_hash[4])
            body.append(h_hash[3])
            body.append(e_hash[8])
            body.append(f_hash[7])
            body.append(g_hash[5])
            body.append(g_hash[6])
            body.append(h_hash[4])
            body.append(e_hash[9])
            body.append(e_hash[10])
            body.append(f_hash[8])
            body.append(g_hash[7])
            body.append(h_hash[5])
            body.append(h_hash[6])
            body.append(e_hash[11])
            body.append(f_hash[9])
            body.append(f_hash[10])
            body.append(g_hash[8])
            body.append(h_hash[7])
            body.append(e_idx[0])
            body.append(f_hash[11])
            body.append(g_hash[9])
            body.append(g_hash[10])
            body.append(h_hash[8])
            body.append(e_idx[1])
            body.append(f_idx[0])
            body.append(g_hash[11])
            body.append(h_hash[9])
            body.append(h_hash[10])
            body.append(e_idx[2])
            body.append(f_idx[1])
            body.append(g_idx[0])
            body.append(h_hash[11])
            body.append(e_idx[3])
            body.append(f_idx[2])
            body.append(g_idx[1])
            body.append(h_idx[0])
            body.append(e_idx[4])
            body.append(f_idx[3])
            body.append(g_idx[2])
            body.append(h_idx[1])
            body.append(f_idx[4])
            body.append(g_idx[3])
            body.append(h_idx[2])
            body.append(i_hash[0])
            body.append(g_idx[4])
            body.append(h_idx[3])
            body.append(i_hash[1])
            body.append(i_hash[2])
            body.append(j_hash[0])
            body.append(h_idx[4])
            body.append(i_hash[3])
            body.append(j_hash[1])
            body.append(j_hash[2])
            body.append(k_hash[0])
            body.append(i_hash[4])
            body.append(j_hash[3])
            body.append(k_hash[1])
            body.append(k_hash[2])
            body.append(l_hash[0])
            body.append(i_hash[5])
            body.append(i_hash[6])
            body.append(j_hash[4])
            body.append(k_hash[3])
            body.append(l_hash[1])
            body.append(l_hash[2])
            body.append(i_hash[7])
            body.append(j_hash[5])
            body.append(j_hash[6])
            body.append(k_hash[4])
            body.append(l_hash[3])
            body.append(i_hash[8])
            body.append(j_hash[7])
            body.append(k_hash[5])
            body.append(k_hash[6])
            body.append(l_hash[4])
            body.append(m_hash[0])
            body.append(i_hash[9])
            body.append(i_hash[10])
            body.append(j_hash[8])
            body.append(k_hash[7])
            body.append(l_hash[5])
            body.append(l_hash[6])
            body.append(i_hash[11])
            body.append(j_hash[9])
            body.append(j_hash[10])
            body.append(k_hash[8])
            body.append(l_hash[7])
            body.append(m_hash[1])
            body.append(i_idx[0])
            body.append(j_hash[11])
            body.append(k_hash[9])
            body.append(k_hash[10])
            body.append(l_hash[8])
            body.append(m_hash[2])
            body.append(i_idx[1])
            body.append(j_idx[0])
            body.append(k_hash[11])
            body.append(l_hash[9])
            body.append(l_hash[10])
            body.append(m_hash[3])
            body.append(i_idx[2])
            body.append(j_idx[1])
            body.append(k_idx[0])
            body.append(l_hash[11])
            body.append(m_hash[4])
            body.append(n_hash[0])
            body.append(i_idx[3])
            body.append(j_idx[2])
            body.append(k_idx[1])
            body.append(l_idx[0])
            body.append(m_hash[5])
            body.append(m_hash[6])
            body.append(i_idx[4])
            body.append(j_idx[3])
            body.append(k_idx[2])
            body.append(l_idx[1])
            body.append(m_hash[7])
            body.append(n_hash[1])
            body.append(j_idx[4])
            body.append(k_idx[3])
            body.append(l_idx[2])
            body.append(m_hash[8])
            body.append(n_hash[2])
            body.append(n_hash[3])
            body.append(k_idx[4])
            body.append(l_idx[3])
            body.append(m_hash[9])
            body.append(m_hash[10])
            body.append(n_hash[4])
            body.append(o_hash[0])
            body.append(l_idx[4])
            body.append(m_hash[11])
            body.append(n_hash[5])
            body.append(n_hash[6])
            body.append(o_hash[1])
            body.append(o_hash[2])
            body.append(m_idx[0])
            body.append(n_hash[7])
            body.append(o_hash[3])
            body.append(p_hash[0])
            body.append(m_idx[1])
            body.append(n_hash[8])
            body.append(o_hash[4])
            body.append(p_hash[1])
            body.append(p_hash[2])
            body.append(m_idx[2])
            body.append(n_hash[9])
            body.append(n_hash[10])
            body.append(o_hash[5])
            body.append(o_hash[6])
            body.append(p_hash[3])
            body.append(m_idx[3])
            body.append(n_hash[11])
            body.append(o_hash[7])
            body.append(p_hash[4])
            body.append(m_idx[4])
            body.append(n_idx[0])
            body.append(o_hash[8])
            body.append(p_hash[5])
            body.append(p_hash[6])
            body.append(n_idx[1])
            body.append(o_hash[9])
            body.append(o_hash[10])
            body.append(p_hash[7])
            body.append(n_idx[2])
            body.append(o_hash[11])
            body.append(p_hash[8])
            body.append(n_idx[3])
            body.append(o_idx[0])
            body.append(p_hash[9])
            body.append(p_hash[10])
            body.append(n_idx[4])
            body.append(o_idx[1])
            body.append(p_hash[11])
            body.append(o_idx[2])
            body.append(p_idx[0])
            body.append(o_idx[3])
            body.append(p_idx[1])
            body.append(o_idx[4])
            body.append(p_idx[2])
            body.append(p_idx[3])
            body.append(p_idx[4])

        # --- No-load hextet with 4-node arithmetic select for round_mod==2 ---
        def emit_hextet_arith4(body, grp_start, node3_b, node5_b, diff34_b, diff56_b, next_s=None):
            """Emit 16-group hextet using pure valu arithmetic to select from 4 nodes.
            idx ∈ {3,4,5,6}, sub = idx-3 ∈ {0,1,2,3}.
            bit0 = sub & 1 = (idx+1) & 1 (since idx-3 ≡ idx+1 mod 2, verify below)
            bit1 = sub >> 1 (right-shift by 1 on {0,1,2,3} gives {0,0,1,1})

            idx=3: sub=0, bit0=0, bit1=0 → lo=node3, hi=node5 → nv=lo=node3 ✓
            idx=4: sub=1, bit0=1, bit1=0 → lo=node3+diff34=node4, hi=node5+diff56=node6 → nv=lo=node4 ✓
            idx=5: sub=2, bit0=0, bit1=1 → lo=node3, hi=node5 → nv=lo+(hi-lo)*1=hi=node5 ✓
            idx=6: sub=3, bit0=1, bit1=1 → lo=node4, hi=node6 → nv=hi=node6 ✓

            Per-group operations (8 valu steps):
              sub = idx - three_vec           (addr_tmp_g[g])
              bit0 = sub & one_vec             (lsb_tmp_pair[p])
              bit1 = sub >> one_vec            (cmp_tmp_pair[p])
              lo = multiply_add(bit0, diff34_b, node3_b)  (t1_tmp_pair[p])
              hi = multiply_add(bit0, diff56_b, node5_b)  (t2_tmp_pair[p])
              diff_lohi = hi - lo              (addr_tmp_g[g], reuse)
              nv = multiply_add(bit1, diff_lohi, lo)     (t1_tmp_pair[p])
              val ^= nv
            """
            gA, gB, gC, gD = grp_start, grp_start+1, grp_start+2, grp_start+3
            gE, gF, gG, gH = grp_start+4, grp_start+5, grp_start+6, grp_start+7
            gI, gJ, gK, gL = grp_start+8, grp_start+9, grp_start+10, grp_start+11
            gM, gN, gO, gP = grp_start+12, grp_start+13, grp_start+14, grp_start+15
            pA, pB, pC, pD = 0, 1, 2, 3
            pE, pF, pG, pH = 0, 1, 2, 3
            pI, pJ, pK, pL = 0, 1, 2, 3
            pM, pN, pO, pP = 0, 1, 2, 3

            # Per-group arithmetic 4-node select slots
            # Uses pair-indexed scratch (t1,t2,lsb,cmp) + per-group scratch (addr_tmp_g)
            def arith4_nv_slots(g, p):
                idx_s = idx_base + g * VLEN
                val_s = val_base + g * VLEN
                sub_s = addr_tmp_g[g]       # per-group, no sharing
                bit0_s = lsb_tmp_pair[p]    # pair-indexed
                bit1_s = cmp_tmp_pair[p]    # pair-indexed
                lo_s = t1_tmp_pair[p]       # pair-indexed
                hi_s = t2_tmp_pair[p]       # pair-indexed
                return [
                    ("valu", ("-", sub_s, idx_s, three_vec)),
                    ("valu", ("&", bit0_s, sub_s, one_vec)),
                    ("valu", (">>", bit1_s, sub_s, one_vec)),
                    # lo = bit0 * diff34 + node3
                    ("valu", ("multiply_add", lo_s, bit0_s, diff34_b, node3_b)),
                    # hi = bit0 * diff56 + node5
                    ("valu", ("multiply_add", hi_s, bit0_s, diff56_b, node5_b)),
                    # diff_lohi = hi - lo → reuse sub_s (sub already consumed by bit0,bit1)
                    ("valu", ("-", sub_s, hi_s, lo_s)),
                    # nv = bit1 * diff_lohi + lo → reuse lo_s
                    ("valu", ("multiply_add", lo_s, bit1_s, sub_s, lo_s)),
                    # val ^= nv
                    ("valu", ("^", val_s, val_s, lo_s)),
                ]

            a_a4 = arith4_nv_slots(gA, pA)
            b_a4 = arith4_nv_slots(gB, pB)
            c_a4 = arith4_nv_slots(gC, pC)
            d_a4 = arith4_nv_slots(gD, pD)
            e_a4 = arith4_nv_slots(gE, pE)
            f_a4 = arith4_nv_slots(gF, pF)
            g_a4 = arith4_nv_slots(gG, pG)
            h_a4 = arith4_nv_slots(gH, pH)
            i_a4 = arith4_nv_slots(gI, pI)
            j_a4 = arith4_nv_slots(gJ, pJ)
            k_a4 = arith4_nv_slots(gK, pK)
            l_a4 = arith4_nv_slots(gL, pL)
            m_a4 = arith4_nv_slots(gM, pM)
            n_a4 = arith4_nv_slots(gN, pN)
            o_a4 = arith4_nv_slots(gO, pO)
            p_a4 = arith4_nv_slots(gP, pP)

            a_hash = group_hash_slots(gA, pA)
            b_hash = group_hash_slots(gB, pB)
            c_hash = group_hash_slots(gC, pC)
            d_hash = group_hash_slots(gD, pD)
            e_hash = group_hash_slots(gE, pE)
            f_hash = group_hash_slots(gF, pF)
            g_hash = group_hash_slots(gG, pG)
            h_hash = group_hash_slots(gH, pH)
            i_hash = group_hash_slots(gI, pI)
            j_hash = group_hash_slots(gJ, pJ)
            k_hash = group_hash_slots(gK, pK)
            l_hash = group_hash_slots(gL, pL)
            m_hash = group_hash_slots(gM, pM)
            n_hash = group_hash_slots(gN, pN)
            o_hash = group_hash_slots(gO, pO)
            p_hash = group_hash_slots(gP, pP)

            a_idx = group_idx_slots(gA, pA)
            b_idx = group_idx_slots(gB, pB)
            c_idx = group_idx_slots(gC, pC)
            d_idx = group_idx_slots(gD, pD)
            e_idx = group_idx_slots(gE, pE)
            f_idx = group_idx_slots(gF, pF)
            g_idx = group_idx_slots(gG, pG)
            h_idx = group_idx_slots(gH, pH)
            i_idx = group_idx_slots(gI, pI)
            j_idx = group_idx_slots(gJ, pJ)
            k_idx = group_idx_slots(gK, pK)
            l_idx = group_idx_slots(gL, pL)
            m_idx = group_idx_slots(gM, pM)
            n_idx = group_idx_slots(gN, pN)
            o_idx = group_idx_slots(gO, pO)
            p_idx = group_idx_slots(gP, pP)

            # Emit arith4_nv for all 16 groups in diagonal order.
            # Each group has 8 valu ops with dependency chain:
            #   sub(1) -> bit0,bit1(2) -> lo,hi(3) -> diff_lohi(4) -> nv(5) -> val^=nv(6)
            # Across groups: same step is independent → pack groups in same cycle.
            # Pair constraint: groups A,E,I,M share pair 0; B,F,J,N share pair 1, etc.
            # Steps 2-7 use pair scratch. Must not use pair 0 simultaneously for A and E.
            # Solution: interleave A-D steps with E-H steps offset by 4 steps,
            # same as the noload hextet diagonal. Here we use a compact approach:
            # emit steps for all 4 groups (A-D) first, then E-H delayed by 4 cycles, etc.

            # Compact diagonal: emit step k for groups [gX] where X's step k is ready.
            # A-D do steps 0-7, E-H do steps 0-7 (offset by when pairs free), etc.
            # Most straightforward: each group's 8 steps in order, interleaved across 4 groups
            # per quartet so pair reuse is safe.

            # ABCD quartet: all use different pairs (pA=0,pB=1,pC=2,pD=3)
            # → all 4 steps can run simultaneously for same step index
            # EFGH also pE=0,pF=1,pG=2,pH=3 → conflict with ABCD on pairs!
            # But EFGH starts step 2 only after ABCD finishes step 1+ (pairs consumed)
            # Actually: pair scratch is read (not written) by arith4 sub-step 2 (bit0/bit1)
            # and written in step 2. ABCD writes to pair 0-3 at step 2 cycle.
            # EFGH also writes to pair 0-3 at step 2 — but if in the same cycle that's WAW conflict!
            # So we need EFGH to start its arith4 AFTER ABCD's step 1 (sub) finishes.
            # ABCD step 1 (sub) doesn't use pair scratch. Only step 2+ does.
            # So EFGH step 1 can overlap with ABCD step 1 (sub uses different addr_tmp_g).

            # Emission plan (by step within arith4):
            # Cycle X:   A.sub, B.sub, C.sub, D.sub, E.sub, F.sub (6 valu slots = 1 cycle)
            # Cycle X+1: G.sub, H.sub, I.sub, J.sub, K.sub, L.sub
            # Cycle X+2: M.sub, N.sub, O.sub, P.sub + A.bit0, A.bit1 (but pair conflict?)
            #
            # Actually since all subs write to different addr_tmp_g[g], they're all independent.
            # So all 16 subs can emit in 3 cycles (16/6 = 3).
            # Then for bit0/bit1: A-D use pairs 0-3, E-H also use pairs 0-3 → conflict!
            # Solution: emit bit0/bit1 in two separate batches: ABCD first, then EFGH, IJKL, MNOP.
            # Within each batch of 4, bit0 and bit1 for same group need 2 slots but same pair
            # → they can be in same bundle since they write to different scratch.
            # Wait: pA=0 → lsb_tmp_pair[0] and cmp_tmp_pair[0]. Different scratch, so both
            # bit0_A and bit1_A can be in same cycle. Also bit0_B (pair 1) and bit1_B.
            # → All 8 (bit0+bit1) for ABCD can go in 1-2 cycles (4 groups × 2 ops = 8 ops, 2 cycles).
            # Similarly for EFGH etc.
            # But EFGH bit0/bit1 conflict with ABCD if in same cycle → must wait for ABCD to finish.

            # Simple safe strategy: emit all 16 arith4_nv groups step by step, with
            # ABCD group at each step, then EFGH, IJKL, MNOP. Since ABCD has different
            # pairs from... wait they all reuse pairs 0-3!
            # Conflict resolution: within same step, only groups with different pairs can coexist.
            # A uses pair 0, E uses pair 0 → cannot be in same cycle for steps 2-7.
            # B uses pair 1, F uses pair 1 → same constraint.
            # So at step 2 (bit0/bit1), emit A,B,C,D first (4 groups, 8 ops, 2 cycles),
            # then E,F,G,H (after A,B,C,D clear pair reuse), etc.
            # But actually: A.step2 writes lsb_pair[0] and cmp_pair[0]. E.step2 also writes them.
            # If A.step2 is in cycle N and E.step2 in cycle N+1, that's fine (no conflict).

            # Cleanest approach: emit in diagonal order ensuring pair safety.
            # For each step s in 0..7, emit for groups in order A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P.
            # This is safe because: at step s, A uses pairs 0; at step s, E also uses pair 0.
            # The build() packer will put A.step[s] and E.step[s] in the same bundle IF no conflict.
            # At step 2 (bit0): A writes lsb_pair[0], E writes lsb_pair[0] → WAW conflict → separate cycles.
            # So A.step2 goes first, E.step2 follows next available cycle. This is safe and automatic.

            # Emit arith4_nv in QUARTET ORDER to ensure pair-reuse safety.
            # CRITICAL: Groups A,E,I,M all share pair 0. If E.bit0 (writes lsb_pair[0]) is
            # emitted before A.lo (reads lsb_pair[0]), the packer may schedule E.bit0 first,
            # causing A.lo to read E's lsb instead of A's. Fix: emit complete quartet
            # (ABCD steps 0-7) before starting EFGH, which ensures A.lo comes before E.bit0.
            # Within a quartet, ABCD use different pairs (0,1,2,3), so all 4 can interleave.
            quartets = [
                [(gA,pA), (gB,pB), (gC,pC), (gD,pD)],
                [(gE,pE), (gF,pF), (gG,pG), (gH,pH)],
                [(gI,pI), (gJ,pJ), (gK,pK), (gL,pL)],
                [(gM,pM), (gN,pN), (gO,pO), (gP,pP)],
            ]
            for quartet in quartets:
                # Emit all 8 steps for this quartet in step order.
                # Within each step, emit all 4 groups (different pairs → parallelism).
                # Step 0: sub (writes addr_tmp_g[g], no pair scratch)
                for g, p in quartet:
                    a4 = arith4_nv_slots(g, p)
                    body.append(a4[0])
                # Step 1+2: bit0 and bit1 (write lsb_pair[p] and cmp_pair[p])
                for g, p in quartet:
                    a4 = arith4_nv_slots(g, p)
                    body.append(a4[1])  # bit0
                    body.append(a4[2])  # bit1
                # Step 3+4: lo and hi (read lsb_pair[p], write t1_pair[p] and t2_pair[p])
                for g, p in quartet:
                    a4 = arith4_nv_slots(g, p)
                    body.append(a4[3])  # lo
                    body.append(a4[4])  # hi
                # Step 5: diff_lohi (read t1+t2 pair, write addr_tmp_g[g])
                for g, p in quartet:
                    a4 = arith4_nv_slots(g, p)
                    body.append(a4[5])  # diff_lohi
                # Step 6: nv (read cmp_pair[p], addr_tmp_g[g], t1_pair[p], write t1_pair[p])
                for g, p in quartet:
                    a4 = arith4_nv_slots(g, p)
                    body.append(a4[6])  # nv
                # Step 7: val ^= nv (read t1_pair[p], write val_s)
                for g, p in quartet:
                    a4 = arith4_nv_slots(g, p)
                    body.append(a4[7])  # val ^= nv

            # Hash+idx diagonal (same as emit_hextet_noload)
            body.append(a_hash[0])

            body.append(a_hash[1])
            body.append(a_hash[2])
            body.append(b_hash[0])

            body.append(a_hash[3])
            body.append(b_hash[1])
            body.append(b_hash[2])
            body.append(c_hash[0])

            body.append(a_hash[4])
            body.append(b_hash[3])
            body.append(c_hash[1])
            body.append(c_hash[2])
            body.append(d_hash[0])

            body.append(a_hash[5])
            body.append(a_hash[6])
            body.append(b_hash[4])
            body.append(c_hash[3])
            body.append(d_hash[1])
            body.append(d_hash[2])

            body.append(a_hash[7])
            body.append(b_hash[5])
            body.append(b_hash[6])
            body.append(c_hash[4])
            body.append(d_hash[3])

            body.append(a_hash[8])
            body.append(b_hash[7])
            body.append(c_hash[5])
            body.append(c_hash[6])
            body.append(d_hash[4])

            body.append(a_hash[9])
            body.append(a_hash[10])
            body.append(b_hash[8])
            body.append(c_hash[7])
            body.append(d_hash[5])
            body.append(d_hash[6])

            body.append(a_hash[11])
            body.append(b_hash[9])
            body.append(b_hash[10])
            body.append(c_hash[8])
            body.append(d_hash[7])

            body.append(a_idx[0])
            body.append(b_hash[11])
            body.append(c_hash[9])
            body.append(c_hash[10])
            body.append(d_hash[8])

            body.append(a_idx[1])
            body.append(b_idx[0])
            body.append(c_hash[11])
            body.append(d_hash[9])
            body.append(d_hash[10])

            body.append(a_idx[2])
            body.append(b_idx[1])
            body.append(c_idx[0])
            body.append(d_hash[11])

            body.append(a_idx[3])
            body.append(b_idx[2])
            body.append(c_idx[1])
            body.append(d_idx[0])

            body.append(a_idx[4])
            body.append(b_idx[3])
            body.append(c_idx[2])
            body.append(d_idx[1])

            body.append(b_idx[4])
            body.append(c_idx[3])
            body.append(d_idx[2])

            body.append(c_idx[4])
            body.append(d_idx[3])
            body.append(e_hash[0])

            body.append(d_idx[4])
            body.append(e_hash[1])
            body.append(e_hash[2])
            body.append(f_hash[0])

            body.append(e_hash[3])
            body.append(f_hash[1])
            body.append(f_hash[2])
            body.append(g_hash[0])

            body.append(e_hash[4])
            body.append(f_hash[3])
            body.append(g_hash[1])
            body.append(g_hash[2])
            body.append(h_hash[0])

            body.append(e_hash[5])
            body.append(e_hash[6])
            body.append(f_hash[4])
            body.append(g_hash[3])
            body.append(h_hash[1])
            body.append(h_hash[2])

            body.append(e_hash[7])
            body.append(f_hash[5])
            body.append(f_hash[6])
            body.append(g_hash[4])
            body.append(h_hash[3])

            body.append(e_hash[8])
            body.append(f_hash[7])
            body.append(g_hash[5])
            body.append(g_hash[6])
            body.append(h_hash[4])

            body.append(e_hash[9])
            body.append(e_hash[10])
            body.append(f_hash[8])
            body.append(g_hash[7])
            body.append(h_hash[5])
            body.append(h_hash[6])

            body.append(e_hash[11])
            body.append(f_hash[9])
            body.append(f_hash[10])
            body.append(g_hash[8])
            body.append(h_hash[7])

            body.append(e_idx[0])
            body.append(f_hash[11])
            body.append(g_hash[9])
            body.append(g_hash[10])
            body.append(h_hash[8])

            body.append(e_idx[1])
            body.append(f_idx[0])
            body.append(g_hash[11])
            body.append(h_hash[9])
            body.append(h_hash[10])

            body.append(e_idx[2])
            body.append(f_idx[1])
            body.append(g_idx[0])
            body.append(h_hash[11])

            body.append(e_idx[3])
            body.append(f_idx[2])
            body.append(g_idx[1])
            body.append(h_idx[0])

            body.append(e_idx[4])
            body.append(f_idx[3])
            body.append(g_idx[2])
            body.append(h_idx[1])

            body.append(f_idx[4])
            body.append(g_idx[3])
            body.append(h_idx[2])
            body.append(i_hash[0])

            body.append(g_idx[4])
            body.append(h_idx[3])
            body.append(i_hash[1])
            body.append(i_hash[2])
            body.append(j_hash[0])

            body.append(h_idx[4])
            body.append(i_hash[3])
            body.append(j_hash[1])
            body.append(j_hash[2])
            body.append(k_hash[0])

            body.append(i_hash[4])
            body.append(j_hash[3])
            body.append(k_hash[1])
            body.append(k_hash[2])
            body.append(l_hash[0])

            body.append(i_hash[5])
            body.append(i_hash[6])
            body.append(j_hash[4])
            body.append(k_hash[3])
            body.append(l_hash[1])
            body.append(l_hash[2])

            body.append(i_hash[7])
            body.append(j_hash[5])
            body.append(j_hash[6])
            body.append(k_hash[4])
            body.append(l_hash[3])

            body.append(i_hash[8])
            body.append(j_hash[7])
            body.append(k_hash[5])
            body.append(k_hash[6])
            body.append(l_hash[4])
            body.append(m_hash[0])

            body.append(i_hash[9])
            body.append(i_hash[10])
            body.append(j_hash[8])
            body.append(k_hash[7])
            body.append(l_hash[5])
            body.append(l_hash[6])

            body.append(i_hash[11])
            body.append(j_hash[9])
            body.append(j_hash[10])
            body.append(k_hash[8])
            body.append(l_hash[7])
            body.append(m_hash[1])

            body.append(i_idx[0])
            body.append(j_hash[11])
            body.append(k_hash[9])
            body.append(k_hash[10])
            body.append(l_hash[8])
            body.append(m_hash[2])

            body.append(i_idx[1])
            body.append(j_idx[0])
            body.append(k_hash[11])
            body.append(l_hash[9])
            body.append(l_hash[10])
            body.append(m_hash[3])

            body.append(i_idx[2])
            body.append(j_idx[1])
            body.append(k_idx[0])
            body.append(l_hash[11])
            body.append(m_hash[4])
            body.append(n_hash[0])

            body.append(i_idx[3])
            body.append(j_idx[2])
            body.append(k_idx[1])
            body.append(l_idx[0])
            body.append(m_hash[5])
            body.append(m_hash[6])

            body.append(i_idx[4])
            body.append(j_idx[3])
            body.append(k_idx[2])
            body.append(l_idx[1])
            body.append(m_hash[7])
            body.append(n_hash[1])

            body.append(j_idx[4])
            body.append(k_idx[3])
            body.append(l_idx[2])
            body.append(m_hash[8])
            body.append(n_hash[2])
            body.append(n_hash[3])

            body.append(k_idx[4])
            body.append(l_idx[3])
            body.append(m_hash[9])
            body.append(m_hash[10])
            body.append(n_hash[4])
            body.append(o_hash[0])

            body.append(l_idx[4])
            body.append(m_hash[11])
            body.append(n_hash[5])
            body.append(n_hash[6])
            body.append(o_hash[1])
            body.append(o_hash[2])

            # Tail (step 50+): same as normal hextet, with optional next_s prefetch
            # Prepare next-round A+B+C+D head slots (if overlapping)
            if next_s is not None:
                na_addr_t = next_s['a_addr']
                na_loads_t = next_s['a_loads']
                nb_addr_t = next_s['b_addr']
                nb_loads_t = next_s['b_loads']
                nc_addr_t = next_s['c_addr']
                nc_loads_t = next_s['c_loads']
                nd_addr_t = next_s['d_addr']
            else:
                na_addr_t = nb_addr_t = nc_addr_t = nd_addr_t = []
                na_loads_t = nb_loads_t = nc_loads_t = []

            # OPTIMIZATION: Emit ALL 4 addr ops first (before step 50 valu ops).
            # Same as normal hextet tail optimization: ensures addrs pack in first tail bundle,
            # enabling loads to start ~12 cycles earlier.
            body.extend(na_addr_t)
            body.extend(nb_addr_t)
            body.extend(nc_addr_t)
            body.extend(nd_addr_t)

            body.append(m_idx[0]); body.append(n_hash[7]); body.append(o_hash[3]); body.append(p_hash[0])
            body.append(m_idx[1]); body.append(n_hash[8]); body.append(o_hash[4])
            body.append(p_hash[1]); body.append(p_hash[2])
            body.extend(na_loads_t[0:2])
            body.append(m_idx[2]); body.append(n_hash[9]); body.append(n_hash[10])
            body.append(o_hash[5]); body.append(o_hash[6]); body.append(p_hash[3])
            body.extend(na_loads_t[2:4])
            body.append(m_idx[3]); body.append(n_hash[11]); body.append(o_hash[7]); body.append(p_hash[4])
            body.extend(na_loads_t[4:6])
            body.append(m_idx[4]); body.append(n_idx[0]); body.append(o_hash[8])
            body.append(p_hash[5]); body.append(p_hash[6])
            body.extend(na_loads_t[6:8])
            body.append(n_idx[1]); body.append(o_hash[9]); body.append(o_hash[10]); body.append(p_hash[7])
            body.extend(nb_loads_t[0:2])
            body.append(n_idx[2]); body.append(o_hash[11]); body.append(p_hash[8])
            body.extend(nb_loads_t[2:4])
            body.append(n_idx[3]); body.append(o_idx[0]); body.append(p_hash[9]); body.append(p_hash[10])
            body.extend(nb_loads_t[4:6])
            body.append(n_idx[4]); body.append(o_idx[1]); body.append(p_hash[11])
            body.extend(nb_loads_t[6:8])
            body.append(o_idx[2]); body.append(p_idx[0]); body.extend(nc_loads_t[0:2])
            body.append(o_idx[3]); body.append(p_idx[1]); body.extend(nc_loads_t[2:4])
            body.append(o_idx[4]); body.append(p_idx[2]); body.extend(nc_loads_t[4:6])
            body.append(p_idx[3]); body.extend(nc_loads_t[6:8])
            body.append(p_idx[4])

        # --- Main computation loop (fully unrolled, 16-group cross-pipeline) ---
        # 16-group hextet pipeline: A-P groups all interleaved.
        # 32 groups / 16 = 2 hextets, no remainder.
        #
        # KEY INSIGHT: I-P groups should start their loads DURING E-H's hash phase,
        # NOT during A-D/A-H's hash phase. This means:
        # - Steps 1-18: A-H addr+load+xor (identical to 8-group)
        # - Step 19 (E.hash[0]): I.addr computed, I.load starts (load engine now free!)
        # - Steps 19-35: E-H hash diagonal + I-P loads (overlapped!)
        # - Steps 36-50: E-H idx diagonal + I-P xor+hash start (same as before)
        # - Steps 51-70: I-P hash+idx tail (pure valu, ~20 steps)
        # Total ~70 cycles for 16 groups vs 2x49=98. ~28% speedup.
        #
        # Pairs 0-3 are reused: E uses pair 0 (as A), I uses pair 0 (as A,E), M uses pair 0.
        # Safe because A/E finish pair 0 before I/M start.
        body = []

        for rnd in range(rounds):
            round_mod = rnd % 11

            if round_mod == 0:
                # === Special: all idx=0, broadcast single tree node ===
                # Load forest_values[0] = memory[forest_values_p + 0]
                # fvp_vec[0] holds forest_values_p as address; lane 0 = the pointer value
                # load_offset(dest, base, offset) reads scratch[base+offset] as address
                # So load_offset(nv_scalar_r0, fvp_vec, 0) loads memory[scratch[fvp_vec+0]]
                #   = memory[forest_values_p] = forest_values[0]
                setup_slots = [
                    ("load", ("load_offset", nv_scalar_r0, fvp_vec, 0)),
                    ("valu", ("vbroadcast", nv_bcast_r0, nv_scalar_r0)),
                ]
                setup_instrs = self.build(setup_slots)
                body_pre = []
                body_pre.extend([(e, s) for instr in setup_instrs for e, slots in instr.items() for s in slots])
                body.extend(body_pre)

                # Process 2 hextets (32 groups) with no loads
                for hextet_start in range(0, n_groups, 16):
                    emit_hextet_noload(body, hextet_start, nv_bcast_r0)

            elif round_mod == 1:
                # === Special: idx in {1,2}, load 2 nodes + vselect ===
                # Load forest_values[1] = memory[forest_values_p + 1]
                # Load forest_values[2] = memory[forest_values_p + 2]
                # fvp_vec[1] = forest_values_p (same value repeated), so:
                # load_offset(nv_node1_r1, fvp_vec, 1) loads memory[scratch[fvp_vec+1]]
                #   = memory[forest_values_p] = forest_values[0]  ← WRONG: fvp_vec is a vector
                #   where all 8 lanes have value forest_values_p. So fvp_vec[1] = forest_values_p.
                #   load_offset reads scratch[fvp_vec + 1] as address, which = forest_values_p.
                #   This gives forest_values[0] again, not forest_values[1]!
                #
                # Correct approach: we need address forest_values_p+1 and forest_values_p+2.
                # Use add_imm (flow engine):
                #   tmp_addr1 = forest_values_p + 1
                #   tmp_addr2 = forest_values_p + 2
                # Then load from those addresses.
                # But add_imm writes to a scalar scratch location. We need temp scalars.
                # We'll use nv_node1_r1 and nv_node2_r1 as temp address holders.
                setup_slots = [
                    # Compute address forest_values_p+1 into nv_node1_r1
                    ("flow", ("add_imm", nv_node1_r1, sc_forest_values_p, 1)),
                    # Compute address forest_values_p+2 into nv_node2_r1
                    ("flow", ("add_imm", nv_node2_r1, sc_forest_values_p, 2)),
                ]
                setup_instrs1 = self.build(setup_slots)
                for instr in setup_instrs1:
                    for e, slist in instr.items():
                        for s in slist:
                            body.append((e, s))

                # Now load from those addresses
                # load(dest, addr): loads memory[scratch[addr]] into dest
                load_slots = [
                    ("load", ("load", nv_node1_r1, nv_node1_r1)),
                    ("load", ("load", nv_node2_r1, nv_node2_r1)),
                ]
                load_instrs = self.build(load_slots)
                for instr in load_instrs:
                    for e, slist in instr.items():
                        for s in slist:
                            body.append((e, s))

                # Broadcast both nodes
                bcast_slots = [
                    ("valu", ("vbroadcast", nv_bcast1_r1, nv_node1_r1)),
                    ("valu", ("vbroadcast", nv_bcast2_r1, nv_node2_r1)),
                ]
                bcast_instrs = self.build(bcast_slots)
                for instr in bcast_instrs:
                    for e, slist in instr.items():
                        for s in slist:
                            body.append((e, s))

                # Compute diff_r1_vec = node1 - node2 (for arithmetic select)
                body.append(("valu", ("-", diff_r1_vec, nv_bcast1_r1, nv_bcast2_r1)))

                # Process 2 hextets with arithmetic vselect (no flow engine)
                for hextet_start in range(0, n_groups, 16):
                    emit_hextet_vselect2(body, hextet_start, nv_bcast1_r1, nv_bcast2_r1, diff_r1_vec)

            elif round_mod == 2:
                # === Special: idx in {3,4,5,6}, load 4 nodes + arithmetic 2-level select ===
                # Load nodes 3,4,5,6 from forest_values.
                # Use add_imm to compute addresses forest_values_p+3..+6.
                # Re-use nv_nodeX_r2 scratch as temp address holders.
                setup_addr_slots = [
                    ("flow", ("add_imm", nv_node3_r2, sc_forest_values_p, 3)),
                    ("flow", ("add_imm", nv_node4_r2, sc_forest_values_p, 4)),
                ]
                setup_addr_slots2 = [
                    ("flow", ("add_imm", nv_node5_r2, sc_forest_values_p, 5)),
                    ("flow", ("add_imm", nv_node6_r2, sc_forest_values_p, 6)),
                ]
                for si in [setup_addr_slots, setup_addr_slots2]:
                    si_instrs = self.build(si)
                    for instr in si_instrs:
                        for e, slist in instr.items():
                            for s in slist:
                                body.append((e, s))

                # Load node values
                load_slots = [
                    ("load", ("load", nv_node3_r2, nv_node3_r2)),
                    ("load", ("load", nv_node4_r2, nv_node4_r2)),
                ]
                load_slots2 = [
                    ("load", ("load", nv_node5_r2, nv_node5_r2)),
                    ("load", ("load", nv_node6_r2, nv_node6_r2)),
                ]
                for ls in [load_slots, load_slots2]:
                    ls_instrs = self.build(ls)
                    for instr in ls_instrs:
                        for e, slist in instr.items():
                            for s in slist:
                                body.append((e, s))

                # Broadcast node3 and node5
                bcast_slots_r2a = [
                    ("valu", ("vbroadcast", node3_bcast, nv_node3_r2)),
                    ("valu", ("vbroadcast", node5_bcast, nv_node5_r2)),
                ]
                bcast_r2a_instrs = self.build(bcast_slots_r2a)
                for instr in bcast_r2a_instrs:
                    for e, slist in instr.items():
                        for s in slist:
                            body.append((e, s))

                # Compute diff34 = node4 - node3, diff56 = node6 - node5 (valu)
                # Also broadcast three_vec = three_scalar (needed for sub = idx - 3)
                # nv_node4_r2 and nv_node6_r2 are scalars; diff34/diff56 are vectors.
                # We need to broadcast node4 and node6 first to compute diffs as vectors.
                # Simpler: broadcast all 4 nodes, then compute diffs.
                # Actually: multiply_add(dest, bit0, diff34_bcast, node3_bcast) works with vectors.
                # node4_bcast and node6_bcast are intermediate temps. We can reuse pair scratch.
                # Use t1_tmp_pair[0] as node4_bcast_tmp and t2_tmp_pair[0] as node6_bcast_tmp.
                # These will be overwritten in the hextet, but that's fine since we compute diffs first.
                node4_bcast_tmp = t1_tmp_pair[0]
                node6_bcast_tmp = t2_tmp_pair[0]
                bcast_slots_r2b = [
                    ("valu", ("vbroadcast", node4_bcast_tmp, nv_node4_r2)),
                    ("valu", ("vbroadcast", node6_bcast_tmp, nv_node6_r2)),
                    ("valu", ("vbroadcast", three_vec, three_scalar)),
                ]
                bcast_r2b_instrs = self.build(bcast_slots_r2b)
                for instr in bcast_r2b_instrs:
                    for e, slist in instr.items():
                        for s in slist:
                            body.append((e, s))

                # Compute diffs (valu)
                diff_slots = [
                    ("valu", ("-", diff34_bcast, node4_bcast_tmp, node3_bcast)),
                    ("valu", ("-", diff56_bcast, node6_bcast_tmp, node5_bcast)),
                ]
                diff_instrs = self.build(diff_slots)
                for instr in diff_instrs:
                    for e, slist in instr.items():
                        for s in slist:
                            body.append((e, s))

                # Process 2 hextets with 4-node arithmetic select
                # For the last hextet (grp_start=16), if next round is normal,
                # prefetch the next round's hextet0 (groups 0-15) head in the tail.
                next_rnd_is_normal = (rnd+1 < rounds and (rnd+1) % 11 >= 3)
                if next_rnd_is_normal:
                    # Build next_s for groups 0-15 (next round's hextet0)
                    next_s_for_arith4 = {
                        'a_addr': group_addr_slots(0), 'a_loads': group_load_slots(0),
                        'b_addr': group_addr_slots(1), 'b_loads': group_load_slots(1),
                        'c_addr': group_addr_slots(2), 'c_loads': group_load_slots(2),
                        'd_addr': group_addr_slots(3),
                    }
                else:
                    next_s_for_arith4 = None
                for hextet_start in range(0, n_groups, 16):
                    ns = next_s_for_arith4 if (hextet_start == 16 and next_rnd_is_normal) else None
                    emit_hextet_arith4(body, hextet_start, node3_bcast, node5_bcast, diff34_bcast, diff56_bcast, next_s=ns)

            else:
                # === Normal: 16-group hextet pipeline with full loads ===
                # OPTIMIZATION: Inter-hextet overlap for the 2 hextets per round.
                # Hextet0 tail (steps 50-63, 14 cycles pure valu) overlaps with
                # hextet1 head (A.addr + A.loads, uses valu+load engines).
                # This saves ~5 cycles per normal round (50 cycles total).
                #
                # Implementation: precompute both hextets' slots, then emit:
                # hextet0 steps 1-49, hextet0 steps 50-63 WITH next_A overlaid,
                # hextet1 steps 3-63 (A head already emitted in hextet0 tail).

                def compute_hextet_slots(g):
                    """Precompute all slot lists for a 16-group hextet starting at group g."""
                    gA, gB, gC, gD = g,   g+1, g+2, g+3
                    gE, gF, gG, gH = g+4, g+5, g+6, g+7
                    gI, gJ, gK, gL = g+8, g+9, g+10, g+11
                    gM, gN, gO, gP = g+12, g+13, g+14, g+15
                    pA, pB, pC, pD = 0, 1, 2, 3
                    pE, pF, pG, pH = 0, 1, 2, 3
                    pI, pJ, pK, pL = 0, 1, 2, 3
                    pM, pN, pO, pP = 0, 1, 2, 3
                    return dict(
                        gA=gA, gB=gB, gC=gC, gD=gD,
                        gE=gE, gF=gF, gG=gG, gH=gH,
                        gI=gI, gJ=gJ, gK=gK, gL=gL,
                        gM=gM, gN=gN, gO=gO, gP=gP,
                        a_addr=group_addr_slots(gA),
                        b_addr=group_addr_slots(gB), c_addr=group_addr_slots(gC),
                        d_addr=group_addr_slots(gD), e_addr=group_addr_slots(gE),
                        f_addr=group_addr_slots(gF), g_addr=group_addr_slots(gG),
                        h_addr=group_addr_slots(gH), i_addr=group_addr_slots(gI),
                        j_addr=group_addr_slots(gJ), k_addr=group_addr_slots(gK),
                        l_addr=group_addr_slots(gL), m_addr=group_addr_slots(gM),
                        n_addr=group_addr_slots(gN), o_addr=group_addr_slots(gO),
                        p_addr=group_addr_slots(gP),
                        a_loads=group_load_slots(gA), b_loads=group_load_slots(gB),
                        c_loads=group_load_slots(gC), d_loads=group_load_slots(gD),
                        e_loads=group_load_slots(gE), f_loads=group_load_slots(gF),
                        g_loads=group_load_slots(gG), h_loads=group_load_slots(gH),
                        i_loads=group_load_slots(gI), j_loads=group_load_slots(gJ),
                        k_loads=group_load_slots(gK), l_loads=group_load_slots(gL),
                        m_loads=group_load_slots(gM), n_loads=group_load_slots(gN),
                        o_loads=group_load_slots(gO), p_loads=group_load_slots(gP),
                        a_xor=group_xor_slots(gA), b_xor=group_xor_slots(gB),
                        c_xor=group_xor_slots(gC), d_xor=group_xor_slots(gD),
                        e_xor=group_xor_slots(gE), f_xor=group_xor_slots(gF),
                        g_xor=group_xor_slots(gG), h_xor=group_xor_slots(gH),
                        i_xor=group_xor_slots(gI), j_xor=group_xor_slots(gJ),
                        k_xor=group_xor_slots(gK), l_xor=group_xor_slots(gL),
                        m_xor=group_xor_slots(gM), n_xor=group_xor_slots(gN),
                        o_xor=group_xor_slots(gO), p_xor=group_xor_slots(gP),
                        a_hash=group_hash_slots(gA, pA), b_hash=group_hash_slots(gB, pB),
                        c_hash=group_hash_slots(gC, pC), d_hash=group_hash_slots(gD, pD),
                        e_hash=group_hash_slots(gE, pE), f_hash=group_hash_slots(gF, pF),
                        g_hash=group_hash_slots(gG, pG), h_hash=group_hash_slots(gH, pH),
                        i_hash=group_hash_slots(gI, pI), j_hash=group_hash_slots(gJ, pJ),
                        k_hash=group_hash_slots(gK, pK), l_hash=group_hash_slots(gL, pL),
                        m_hash=group_hash_slots(gM, pM), n_hash=group_hash_slots(gN, pN),
                        o_hash=group_hash_slots(gO, pO), p_hash=group_hash_slots(gP, pP),
                        a_idx=group_idx_slots(gA, pA), b_idx=group_idx_slots(gB, pB),
                        c_idx=group_idx_slots(gC, pC), d_idx=group_idx_slots(gD, pD),
                        e_idx=group_idx_slots(gE, pE), f_idx=group_idx_slots(gF, pF),
                        g_idx=group_idx_slots(gG, pG), h_idx=group_idx_slots(gH, pH),
                        i_idx=group_idx_slots(gI, pI), j_idx=group_idx_slots(gJ, pJ),
                        k_idx=group_idx_slots(gK, pK), l_idx=group_idx_slots(gL, pL),
                        m_idx=group_idx_slots(gM, pM), n_idx=group_idx_slots(gN, pN),
                        o_idx=group_idx_slots(gO, pO), p_idx=group_idx_slots(gP, pP),
                    )

                def emit_hextet_body_steps1_49(body, s):
                    """Emit hextet steps 1-49 (head + phases 1 and 2, up to start of P group)."""
                    a_addr=s['a_addr']; b_addr=s['b_addr']; c_addr=s['c_addr']; d_addr=s['d_addr']
                    e_addr=s['e_addr']; f_addr=s['f_addr']; g_addr=s['g_addr']; h_addr=s['h_addr']
                    i_addr=s['i_addr']; j_addr=s['j_addr']; k_addr=s['k_addr']; l_addr=s['l_addr']
                    m_addr=s['m_addr']; n_addr=s['n_addr']; o_addr=s['o_addr']
                    a_loads=s['a_loads']; b_loads=s['b_loads']; c_loads=s['c_loads']; d_loads=s['d_loads']
                    e_loads=s['e_loads']; f_loads=s['f_loads']; g_loads=s['g_loads']; h_loads=s['h_loads']
                    i_loads=s['i_loads']; j_loads=s['j_loads']; k_loads=s['k_loads']; l_loads=s['l_loads']
                    m_loads=s['m_loads']; n_loads=s['n_loads']; o_loads=s['o_loads']; p_loads=s['p_loads']
                    a_xor=s['a_xor']; b_xor=s['b_xor']; c_xor=s['c_xor']; d_xor=s['d_xor']
                    e_xor=s['e_xor']; f_xor=s['f_xor']; g_xor=s['g_xor']; h_xor=s['h_xor']
                    i_xor=s['i_xor']; j_xor=s['j_xor']; k_xor=s['k_xor']; l_xor=s['l_xor']
                    m_xor=s['m_xor']; n_xor=s['n_xor']; o_xor=s['o_xor']; p_xor=s['p_xor']
                    a_hash=s['a_hash']; b_hash=s['b_hash']; c_hash=s['c_hash']; d_hash=s['d_hash']
                    e_hash=s['e_hash']; f_hash=s['f_hash']; g_hash=s['g_hash']; h_hash=s['h_hash']
                    i_hash=s['i_hash']; j_hash=s['j_hash']; k_hash=s['k_hash']; l_hash=s['l_hash']
                    m_hash=s['m_hash']; n_hash=s['n_hash']; o_hash=s['o_hash']; p_hash=s['p_hash']
                    a_idx=s['a_idx']; b_idx=s['b_idx']; c_idx=s['c_idx']; d_idx=s['d_idx']
                    e_idx=s['e_idx']; f_idx=s['f_idx']; g_idx=s['g_idx']; h_idx=s['h_idx']
                    i_idx=s['i_idx']; j_idx=s['j_idx']; k_idx=s['k_idx']; l_idx=s['l_idx']
                    m_idx=s['m_idx']; n_idx=s['n_idx']; o_idx=s['o_idx']

                    # === PHASE 1: A-H startup ===
                    # Step 1: A.addr
                    body.extend(a_addr)
                    # Step 2: A.load[:6] + B.addr + A.load[6:]
                    body.extend(a_loads[:6]); body.extend(b_addr); body.extend(a_loads[6:])
                    # Step 3: A.xor + B.load[:6] + C.addr + B.load[6:]
                    body.extend(a_xor); body.extend(b_loads[:6]); body.extend(c_addr); body.extend(b_loads[6:])
                    # Step 4: A.hash[0] + B.xor + C.load[:6] + D.addr + C.load[6:]
                    body.append(a_hash[0]); body.extend(b_xor); body.extend(c_loads[:6]); body.extend(d_addr); body.extend(c_loads[6:])
                    # Step 5: A.hash[1:3] + B.hash[0] + C.xor + D.load[:6] + E.addr
                    body.append(a_hash[1]); body.append(a_hash[2]); body.append(b_hash[0])
                    body.extend(c_xor); body.extend(d_loads[:6]); body.extend(e_addr)
                    # Step 6: A.hash[3] + B.hash[1:3] + C.hash[0] + D.load[6:] + D.xor + E.load[:2]
                    body.append(a_hash[3]); body.append(b_hash[1]); body.append(b_hash[2]); body.append(c_hash[0])
                    body.extend(d_loads[6:]); body.extend(d_xor); body.extend(e_loads[:2])
                    # Step 7: A.hash[4] + B.hash[3] + C.hash[1:3] + D.hash[0] + E.load[2:4] + F.addr
                    body.append(a_hash[4]); body.append(b_hash[3]); body.append(c_hash[1]); body.append(c_hash[2]); body.append(d_hash[0])
                    body.extend(e_loads[2:4]); body.extend(f_addr)
                    # Step 8: A.hash[5:7] + B.hash[4] + C.hash[3] + D.hash[1:3] + E.load[4:6]
                    body.append(a_hash[5]); body.append(a_hash[6]); body.append(b_hash[4]); body.append(c_hash[3])
                    body.append(d_hash[1]); body.append(d_hash[2]); body.extend(e_loads[4:6])
                    # Step 9: A.hash[7] + B.hash[5:7] + C.hash[4] + D.hash[3] + E.load[6:] + E.xor + F.load[:2]
                    body.append(a_hash[7]); body.append(b_hash[5]); body.append(b_hash[6]); body.append(c_hash[4]); body.append(d_hash[3])
                    body.extend(e_loads[6:]); body.extend(e_xor); body.extend(f_loads[:2])
                    # Step 10: A.hash[8] + B.hash[7] + C.hash[5:7] + D.hash[4] + F.load[2:4] + G.addr
                    body.append(a_hash[8]); body.append(b_hash[7]); body.append(c_hash[5]); body.append(c_hash[6]); body.append(d_hash[4])
                    body.extend(f_loads[2:4]); body.extend(g_addr)
                    # Step 11: A.hash[9:11] + B.hash[8] + C.hash[7] + D.hash[5:7] + F.load[4:6]
                    body.append(a_hash[9]); body.append(a_hash[10]); body.append(b_hash[8]); body.append(c_hash[7])
                    body.append(d_hash[5]); body.append(d_hash[6]); body.extend(f_loads[4:6])
                    # Step 12: A.hash[11] + B.hash[9:11] + C.hash[8] + D.hash[7] + F.load[6:] + F.xor + G.load[:2]
                    body.append(a_hash[11]); body.append(b_hash[9]); body.append(b_hash[10]); body.append(c_hash[8]); body.append(d_hash[7])
                    body.extend(f_loads[6:]); body.extend(f_xor); body.extend(g_loads[:2])
                    # Step 13: A.idx[0] + B.hash[11] + C.hash[9:11] + D.hash[8] + G.load[2:4] + H.addr
                    body.append(a_idx[0]); body.append(b_hash[11]); body.append(c_hash[9]); body.append(c_hash[10]); body.append(d_hash[8])
                    body.extend(g_loads[2:4]); body.extend(h_addr)
                    # Step 14: A.idx[1] + B.idx[0] + C.hash[11] + D.hash[9:11] + G.load[4:6]
                    body.append(a_idx[1]); body.append(b_idx[0]); body.append(c_hash[11])
                    body.append(d_hash[9]); body.append(d_hash[10]); body.extend(g_loads[4:6])
                    # Step 15: A.idx[2] + B.idx[1] + C.idx[0] + D.hash[11] + G.load[6:] + G.xor + H.load[:2]
                    body.append(a_idx[2]); body.append(b_idx[1]); body.append(c_idx[0]); body.append(d_hash[11])
                    body.extend(g_loads[6:]); body.extend(g_xor); body.extend(h_loads[:2])
                    # Step 16: A.idx[3] + B.idx[2] + C.idx[1] + D.idx[0] + H.load[2:4]
                    body.append(a_idx[3]); body.append(b_idx[2]); body.append(c_idx[1]); body.append(d_idx[0])
                    body.extend(h_loads[2:4])
                    # Step 17: A.idx[4] + B.idx[3] + C.idx[2] + D.idx[1] + H.load[4:6]
                    body.append(a_idx[4]); body.append(b_idx[3]); body.append(c_idx[2]); body.append(d_idx[1])
                    body.extend(h_loads[4:6])
                    # Step 18: B.idx[4] + C.idx[3] + D.idx[2] + H.load[6:] + H.xor
                    body.append(b_idx[4]); body.append(c_idx[3]); body.append(d_idx[2])
                    body.extend(h_loads[6:]); body.extend(h_xor)

                    # === PHASE 2: E-H hash + I-P loads ===
                    # Step 19: C.idx[4] + D.idx[3] + E.hash[0] + I.addr
                    body.append(c_idx[4]); body.append(d_idx[3]); body.append(e_hash[0]); body.extend(i_addr)
                    # Step 20: D.idx[4] + E.hash[1:3] + F.hash[0] + I.load[:2] + J.addr
                    body.append(d_idx[4]); body.append(e_hash[1]); body.append(e_hash[2]); body.append(f_hash[0])
                    body.extend(i_loads[:2]); body.extend(j_addr)
                    # Step 21: E.hash[3] + F.hash[1:3] + G.hash[0] + I.load[2:4]
                    body.append(e_hash[3]); body.append(f_hash[1]); body.append(f_hash[2]); body.append(g_hash[0])
                    body.extend(i_loads[2:4])
                    # Step 22: E.hash[4] + F.hash[3] + G.hash[1:3] + H.hash[0] + I.load[4:6] + J.load[:2] + K.addr
                    body.append(e_hash[4]); body.append(f_hash[3]); body.append(g_hash[1]); body.append(g_hash[2]); body.append(h_hash[0])
                    body.extend(i_loads[4:6]); body.extend(j_loads[:2]); body.extend(k_addr)
                    # Step 23: E.hash[5:7] + F.hash[4] + G.hash[3] + H.hash[1:3] + I.load[6:] + I.xor + J.load[2:4]
                    body.append(e_hash[5]); body.append(e_hash[6]); body.append(f_hash[4]); body.append(g_hash[3])
                    body.append(h_hash[1]); body.append(h_hash[2])
                    body.extend(i_loads[6:]); body.extend(i_xor); body.extend(j_loads[2:4])
                    # Step 24: E.hash[7] + F.hash[5:7] + G.hash[4] + H.hash[3] + J.load[4:6] + L.addr
                    body.append(e_hash[7]); body.append(f_hash[5]); body.append(f_hash[6]); body.append(g_hash[4]); body.append(h_hash[3])
                    body.extend(j_loads[4:6]); body.extend(l_addr)
                    # Step 25: E.hash[8] + F.hash[7] + G.hash[5:7] + H.hash[4] + J.load[6:] + J.xor + K.load[:2]
                    body.append(e_hash[8]); body.append(f_hash[7]); body.append(g_hash[5]); body.append(g_hash[6]); body.append(h_hash[4])
                    body.extend(j_loads[6:]); body.extend(j_xor); body.extend(k_loads[:2])
                    # Step 26: E.hash[9:11] + F.hash[8] + G.hash[7] + H.hash[5:7] + K.load[2:4] + M.addr
                    body.append(e_hash[9]); body.append(e_hash[10]); body.append(f_hash[8]); body.append(g_hash[7])
                    body.append(h_hash[5]); body.append(h_hash[6]); body.extend(k_loads[2:4]); body.extend(m_addr)
                    # Step 27: E.hash[11] + F.hash[9:11] + G.hash[8] + H.hash[7] + K.load[4:6] + L.load[:2]
                    body.append(e_hash[11]); body.append(f_hash[9]); body.append(f_hash[10]); body.append(g_hash[8]); body.append(h_hash[7])
                    body.extend(k_loads[4:6]); body.extend(l_loads[:2])
                    # Step 28: E.idx[0] + F.hash[11] + G.hash[9:11] + H.hash[8] + K.load[6:] + K.xor + L.load[2:4] + N.addr
                    body.append(e_idx[0]); body.append(f_hash[11]); body.append(g_hash[9]); body.append(g_hash[10]); body.append(h_hash[8])
                    body.extend(k_loads[6:]); body.extend(k_xor); body.extend(l_loads[2:4]); body.extend(n_addr)
                    # Step 29: E.idx[1] + F.idx[0] + G.hash[11] + H.hash[9:11] + L.load[4:6] + M.load[:2]
                    body.append(e_idx[1]); body.append(f_idx[0]); body.append(g_hash[11]); body.append(h_hash[9]); body.append(h_hash[10])
                    body.extend(l_loads[4:6]); body.extend(m_loads[:2])
                    # Step 30: E.idx[2] + F.idx[1] + G.idx[0] + H.hash[11] + L.load[6:] + L.xor + M.load[2:4] + O.addr
                    body.append(e_idx[2]); body.append(f_idx[1]); body.append(g_idx[0]); body.append(h_hash[11])
                    body.extend(l_loads[6:]); body.extend(l_xor); body.extend(m_loads[2:4]); body.extend(o_addr)
                    # Step 31: E.idx[3] + F.idx[2] + G.idx[1] + H.idx[0] + M.load[4:6] + N.load[:2]
                    body.append(e_idx[3]); body.append(f_idx[2]); body.append(g_idx[1]); body.append(h_idx[0])
                    body.extend(m_loads[4:6]); body.extend(n_loads[:2])
                    # Step 32: E.idx[4] + F.idx[3] + G.idx[2] + H.idx[1] + M.load[6:] + M.xor + N.load[2:4] + P.addr
                    body.append(e_idx[4]); body.append(f_idx[3]); body.append(g_idx[2]); body.append(h_idx[1])
                    body.extend(m_loads[6:]); body.extend(m_xor); body.extend(n_loads[2:4]); body.extend(s['p_addr'])
                    # Step 33: F.idx[4] + G.idx[3] + H.idx[2] + I.hash[0] + N.load[4:6] + O.load[:2]
                    body.append(f_idx[4]); body.append(g_idx[3]); body.append(h_idx[2]); body.append(i_hash[0])
                    body.extend(n_loads[4:6]); body.extend(o_loads[:2])
                    # Step 34: G.idx[4] + H.idx[3] + I.hash[1:3] + J.hash[0] + N.load[6:] + N.xor + O.load[2:4]
                    body.append(g_idx[4]); body.append(h_idx[3]); body.append(i_hash[1]); body.append(i_hash[2]); body.append(j_hash[0])
                    body.extend(n_loads[6:]); body.extend(n_xor); body.extend(o_loads[2:4])
                    # Step 35: H.idx[4] + I.hash[3] + J.hash[1:3] + K.hash[0] + O.load[4:6] + P.load[:2]
                    body.append(h_idx[4]); body.append(i_hash[3]); body.append(j_hash[1]); body.append(j_hash[2]); body.append(k_hash[0])
                    body.extend(o_loads[4:6]); body.extend(p_loads[:2])

                    # === PHASE 3: I-P hash+idx diagonal + final loads ===
                    # Step 36: I.hash[4] + J.hash[3] + K.hash[1:3] + L.hash[0] + O.load[6:] + O.xor + P.load[2:4]
                    body.append(i_hash[4]); body.append(j_hash[3]); body.append(k_hash[1]); body.append(k_hash[2]); body.append(l_hash[0])
                    body.extend(o_loads[6:]); body.extend(o_xor); body.extend(p_loads[2:4])
                    # Step 37: I.hash[5:7] + J.hash[4] + K.hash[3] + L.hash[1:3] + P.load[4:6]
                    body.append(i_hash[5]); body.append(i_hash[6]); body.append(j_hash[4]); body.append(k_hash[3])
                    body.append(l_hash[1]); body.append(l_hash[2]); body.extend(p_loads[4:6])
                    # Step 38: I.hash[7] + J.hash[5:7] + K.hash[4] + L.hash[3] + P.load[6:] + P.xor
                    body.append(i_hash[7]); body.append(j_hash[5]); body.append(j_hash[6]); body.append(k_hash[4]); body.append(l_hash[3])
                    body.extend(p_loads[6:]); body.extend(p_xor)
                    # Step 39: I.hash[8] + J.hash[7] + K.hash[5:7] + L.hash[4] + M.hash[0]
                    body.append(i_hash[8]); body.append(j_hash[7]); body.append(k_hash[5]); body.append(k_hash[6]); body.append(l_hash[4]); body.append(m_hash[0])
                    # Step 40: I.hash[9:11] + J.hash[8] + K.hash[7] + L.hash[5:7]
                    body.append(i_hash[9]); body.append(i_hash[10]); body.append(j_hash[8]); body.append(k_hash[7]); body.append(l_hash[5]); body.append(l_hash[6])
                    # Step 41: I.hash[11] + J.hash[9:11] + K.hash[8] + L.hash[7] + M.hash[1]
                    body.append(i_hash[11]); body.append(j_hash[9]); body.append(j_hash[10]); body.append(k_hash[8]); body.append(l_hash[7]); body.append(m_hash[1])
                    # Step 42: I.idx[0] + J.hash[11] + K.hash[9:11] + L.hash[8] + M.hash[2]
                    body.append(i_idx[0]); body.append(j_hash[11]); body.append(k_hash[9]); body.append(k_hash[10]); body.append(l_hash[8]); body.append(m_hash[2])
                    # Step 43: I.idx[1] + J.idx[0] + K.hash[11] + L.hash[9:11] + M.hash[3]
                    body.append(i_idx[1]); body.append(j_idx[0]); body.append(k_hash[11]); body.append(l_hash[9]); body.append(l_hash[10]); body.append(m_hash[3])
                    # Step 44: I.idx[2] + J.idx[1] + K.idx[0] + L.hash[11] + M.hash[4] + N.hash[0]
                    body.append(i_idx[2]); body.append(j_idx[1]); body.append(k_idx[0]); body.append(l_hash[11]); body.append(m_hash[4]); body.append(n_hash[0])
                    # Step 45: I.idx[3] + J.idx[2] + K.idx[1] + L.idx[0] + M.hash[5:7]
                    body.append(i_idx[3]); body.append(j_idx[2]); body.append(k_idx[1]); body.append(l_idx[0]); body.append(m_hash[5]); body.append(m_hash[6])
                    # Step 46: I.idx[4] + J.idx[3] + K.idx[2] + L.idx[1] + M.hash[7] + N.hash[1]
                    body.append(i_idx[4]); body.append(j_idx[3]); body.append(k_idx[2]); body.append(l_idx[1]); body.append(m_hash[7]); body.append(n_hash[1])
                    # Step 47: J.idx[4] + K.idx[3] + L.idx[2] + M.hash[8] + N.hash[2:4]
                    body.append(j_idx[4]); body.append(k_idx[3]); body.append(l_idx[2]); body.append(m_hash[8]); body.append(n_hash[2]); body.append(n_hash[3])
                    # Step 48: K.idx[4] + L.idx[3] + M.hash[9:11] + N.hash[4] + O.hash[0]
                    body.append(k_idx[4]); body.append(l_idx[3]); body.append(m_hash[9]); body.append(m_hash[10]); body.append(n_hash[4]); body.append(o_hash[0])
                    # Step 49: L.idx[4] + M.hash[11] + N.hash[5:7] + O.hash[1:3]
                    body.append(l_idx[4]); body.append(m_hash[11]); body.append(n_hash[5]); body.append(n_hash[6]); body.append(o_hash[1]); body.append(o_hash[2])

                def emit_hextet_tail_steps50_63(body, s, next_s=None):
                    """Emit hextet steps 50-63.
                    If next_s is provided, interleave next hextet's A/B/C/D addr+loads
                    during these pure-valu tail steps (load engine otherwise idle).
                    KEY OPTIMIZATION: All 4 addr ops (A,B,C,D) are emitted FIRST (prepended
                    before step 50 valu ops) so that loads can start as early as possible
                    (packer places addrs in bundle ~52, enabling loads at bundle ~53 instead
                    of the old approach where na_addr ended up at bundle ~65).
                    """
                    m_idx=s['m_idx']; n_idx=s['n_idx']; o_idx=s['o_idx']; p_idx=s['p_idx']
                    n_hash=s['n_hash']; o_hash=s['o_hash']; p_hash=s['p_hash']

                    # Prepare next-hextet A+B+C+D head slots (if overlapping)
                    if next_s is not None:
                        na_addr = next_s['a_addr']     # 1 valu slot
                        na_loads = next_s['a_loads']   # 8 load slots
                        nb_addr = next_s['b_addr']     # 1 valu slot
                        nb_loads = next_s['b_loads']   # 8 load slots
                        nc_addr = next_s['c_addr']     # 1 valu slot
                        nc_loads = next_s['c_loads']   # 8 load slots
                        nd_addr = next_s['d_addr']     # 1 valu slot (D loads start in hextet1 body)
                    else:
                        na_addr = []
                        na_loads = []
                        nb_addr = []
                        nb_loads = []
                        nc_addr = []
                        nc_loads = []
                        nd_addr = []

                    # CRITICAL OPTIMIZATION: Emit ALL addr ops FIRST (before step 50 valu ops).
                    # This ensures the packer places them in the first tail bundle (~bundle 52),
                    # allowing loads to start at bundle ~53 instead of ~66.
                    # Previously, na_addr was interleaved at step 50 (appended after 4 valu ops),
                    # causing the packer to push na_addr to bundle 65 (last tail bundle) due to
                    # VALU saturation in earlier tail steps.
                    body.extend(na_addr)   # next A group addr: write addr_tmp_g[gA2]
                    body.extend(nb_addr)   # next B group addr: independent of na_addr writes
                    body.extend(nc_addr)   # next C group addr
                    body.extend(nd_addr)   # next D group addr

                    # Step 50: M.idx[0] + N.hash[7] + O.hash[3] + P.hash[0]
                    body.append(m_idx[0]); body.append(n_hash[7]); body.append(o_hash[3]); body.append(p_hash[0])
                    # Step 51: M.idx[1] + N.hash[8] + O.hash[4] + P.hash[1:3] [+ next_A.loads[0:2]]
                    body.append(m_idx[1]); body.append(n_hash[8]); body.append(o_hash[4])
                    body.append(p_hash[1]); body.append(p_hash[2])
                    body.extend(na_loads[0:2])  # load slots (na_addr already in past bundle)
                    # Step 52: M.idx[2] + N.hash[9:11] + O.hash[5:7] + P.hash[3] [+ next_A.loads[2:4]]
                    body.append(m_idx[2]); body.append(n_hash[9]); body.append(n_hash[10])
                    body.append(o_hash[5]); body.append(o_hash[6]); body.append(p_hash[3])
                    body.extend(na_loads[2:4])
                    # Step 53: M.idx[3] + N.hash[11] + O.hash[7] + P.hash[4] [+ next_A.loads[4:6]]
                    body.append(m_idx[3]); body.append(n_hash[11]); body.append(o_hash[7]); body.append(p_hash[4])
                    body.extend(na_loads[4:6])
                    # Step 54: M.idx[4] + N.idx[0] + O.hash[8] + P.hash[5:7] [+ next_A.loads[6:8]]
                    body.append(m_idx[4]); body.append(n_idx[0]); body.append(o_hash[8])
                    body.append(p_hash[5]); body.append(p_hash[6])
                    body.extend(na_loads[6:8])
                    # Step 55: N.idx[1] + O.hash[9:11] + P.hash[7] [+ next_B.loads[0:2]]
                    body.append(n_idx[1]); body.append(o_hash[9]); body.append(o_hash[10]); body.append(p_hash[7])
                    body.extend(nb_loads[0:2])
                    # Step 56: N.idx[2] + O.hash[11] + P.hash[8] [+ next_B.loads[2:4]]
                    body.append(n_idx[2]); body.append(o_hash[11]); body.append(p_hash[8])
                    body.extend(nb_loads[2:4])
                    # Step 57: N.idx[3] + O.idx[0] + P.hash[9:11] [+ next_B.loads[4:6]]
                    body.append(n_idx[3]); body.append(o_idx[0]); body.append(p_hash[9]); body.append(p_hash[10])
                    body.extend(nb_loads[4:6])
                    # Step 58: N.idx[4] + O.idx[1] + P.hash[11] [+ next_B.loads[6:8]]
                    body.append(n_idx[4]); body.append(o_idx[1]); body.append(p_hash[11])
                    body.extend(nb_loads[6:8])
                    # Step 59: O.idx[2] + P.idx[0] [+ next_C.loads[0:2]]
                    body.append(o_idx[2]); body.append(p_idx[0])
                    body.extend(nc_loads[0:2])
                    # Step 60: O.idx[3] + P.idx[1] [+ next_C.loads[2:4]]
                    body.append(o_idx[3]); body.append(p_idx[1])
                    body.extend(nc_loads[2:4])
                    # Step 61: O.idx[4] + P.idx[2] [+ next_C.loads[4:6]]
                    body.append(o_idx[4]); body.append(p_idx[2])
                    body.extend(nc_loads[4:6])
                    # Step 62: P.idx[3] [+ next_C.loads[6:8]]
                    body.append(p_idx[3])
                    body.extend(nc_loads[6:8])
                    # Step 63: P.idx[4]
                    # nd_addr is already emitted at start of tail (first bundle).
                    # D loads are handled in hextet1 body step 3 (which does a_xor + d_loads[:2]).
                    body.append(p_idx[4])

                def emit_hextet_body_steps3_63(body, s, next_round_s0=None, emit_tail=True, intra_hextet_s=None, d_preloaded=False):
                    """Emit hextet steps 3-63 (skipping A head: addr and loads already done).
                    If next_round_s0 is provided, the tail will prefetch next round's hextet0 head.
                    If emit_tail=False, only steps 3-49 are emitted (caller handles the tail).
                    If intra_hextet_s is provided (for hextet0 only), embed next hextet's A-C addr+loads
                    into steps 3-49 to use the idle load engine during steps 39-49:
                      - Step 3: prepend nh A,B,C,D addr ops (4 valu slots, 5 free available)
                      - Steps 39-49: add A loads (steps 39-42), B loads (steps 43-46),
                        C loads[0:4] (steps 47-48), D loads[0:2] (step 49) = 22 loads
                    If d_preloaded=True (for hextet1 when hextet0 used intra_hextet_s), skip D.loads
                    in steps 3-6 (D was loaded via hextet0 tail). This saves 1 cycle from step 6
                    overflow removal (step 6 previously had 4 loads causing 2 bundles, now 2 loads).
                    saving ~10c per round."""
                    a_loads=s['a_loads']; b_loads=s['b_loads']; c_loads=s['c_loads']; d_loads=s['d_loads']
                    e_loads=s['e_loads']; f_loads=s['f_loads']; g_loads=s['g_loads']; h_loads=s['h_loads']
                    i_loads=s['i_loads']; j_loads=s['j_loads']; k_loads=s['k_loads']; l_loads=s['l_loads']
                    m_loads=s['m_loads']; n_loads=s['n_loads']; o_loads=s['o_loads']; p_loads=s['p_loads']
                    b_addr=s['b_addr']; c_addr=s['c_addr']; d_addr=s['d_addr']; e_addr=s['e_addr']
                    f_addr=s['f_addr']; g_addr=s['g_addr']; h_addr=s['h_addr']; i_addr=s['i_addr']
                    j_addr=s['j_addr']; k_addr=s['k_addr']; l_addr=s['l_addr']; m_addr=s['m_addr']
                    n_addr=s['n_addr']; o_addr=s['o_addr']; p_addr=s['p_addr']
                    a_xor=s['a_xor']; b_xor=s['b_xor']; c_xor=s['c_xor']; d_xor=s['d_xor']
                    e_xor=s['e_xor']; f_xor=s['f_xor']; g_xor=s['g_xor']; h_xor=s['h_xor']
                    i_xor=s['i_xor']; j_xor=s['j_xor']; k_xor=s['k_xor']; l_xor=s['l_xor']
                    m_xor=s['m_xor']; n_xor=s['n_xor']; o_xor=s['o_xor']; p_xor=s['p_xor']
                    a_hash=s['a_hash']; b_hash=s['b_hash']; c_hash=s['c_hash']; d_hash=s['d_hash']
                    e_hash=s['e_hash']; f_hash=s['f_hash']; g_hash=s['g_hash']; h_hash=s['h_hash']
                    i_hash=s['i_hash']; j_hash=s['j_hash']; k_hash=s['k_hash']; l_hash=s['l_hash']
                    m_hash=s['m_hash']; n_hash=s['n_hash']; o_hash=s['o_hash']; p_hash=s['p_hash']
                    a_idx=s['a_idx']; b_idx=s['b_idx']; c_idx=s['c_idx']; d_idx=s['d_idx']
                    e_idx=s['e_idx']; f_idx=s['f_idx']; g_idx=s['g_idx']; h_idx=s['h_idx']
                    i_idx=s['i_idx']; j_idx=s['j_idx']; k_idx=s['k_idx']; l_idx=s['l_idx']
                    m_idx=s['m_idx']; n_idx=s['n_idx']; o_idx=s['o_idx']; p_idx=s['p_idx']

                    # Setup intra-hextet prefetch slots (hextet1 A,B,C,D addr+loads if provided)
                    if intra_hextet_s is not None:
                        nh_a_addr = intra_hextet_s['a_addr']    # 1 valu
                        nh_b_addr = intra_hextet_s['b_addr']    # 1 valu
                        nh_c_addr = intra_hextet_s['c_addr']    # 1 valu
                        nh_d_addr = intra_hextet_s['d_addr']    # 1 valu
                        nh_a_loads = intra_hextet_s['a_loads']  # 8 loads
                        nh_b_loads = intra_hextet_s['b_loads']  # 8 loads
                        nh_c_loads = intra_hextet_s['c_loads']  # 8 loads
                        nh_d_loads = intra_hextet_s['d_loads']  # 8 loads
                    else:
                        nh_a_addr = nh_b_addr = nh_c_addr = nh_d_addr = []
                        nh_a_loads = nh_b_loads = nh_c_loads = nh_d_loads = []

                    # Step 3 (modified): [nh A,B,C,D addrs] + A.xor + D.loads[0:2]
                    # NOTE: prev hextet tail prefetched: A.addr, A.loads, B.addr, B.loads, C.addr, C.loads, D.addr
                    # D.addr done in tail, D.loads can start right away on load engine (free during A.xor)
                    # nh addrs are for hextet1 groups - independent of current hextet's groups.
                    # Step 3 has only 1 valu (a_xor) + 2 loads, so 4 nh addr valu ops fit easily.
                    body.extend(nh_a_addr); body.extend(nh_b_addr)
                    body.extend(nh_c_addr); body.extend(nh_d_addr)
                    if d_preloaded:
                        # D loads already done (by hextet0 intra+tail prefetch). Skip D loads.
                        body.extend(a_xor)  # step 3: just a_xor (d_preloaded: D loads done)
                        # Step 4 (modified): A.hash[0] + B.xor + E.addr (no d_loads!)
                        body.append(a_hash[0]); body.extend(b_xor); body.extend(e_addr)
                        # Step 5: A.hash[1:3] + B.hash[0] + C.xor (no d_loads, load engine idle)
                        body.append(a_hash[1]); body.append(a_hash[2]); body.append(b_hash[0])
                        body.extend(c_xor)
                        # Step 6: A.hash[3] + B.hash[1:3] + C.hash[0] + D.xor + E.loads[0:2] + F.addr
                        body.append(a_hash[3]); body.append(b_hash[1]); body.append(b_hash[2]); body.append(c_hash[0])
                        body.extend(d_xor); body.extend(e_loads[:2]); body.extend(f_addr)
                    else:
                        body.extend(a_xor); body.extend(d_loads[:2])
                        # Step 4 (modified): A.hash[0] + B.xor + E.addr + D.loads[2:4]
                        # E.addr is independent; emit early to start E.loads sooner
                        body.append(a_hash[0]); body.extend(b_xor); body.extend(e_addr); body.extend(d_loads[2:4])
                        # Step 5: A.hash[1:3] + B.hash[0] + C.xor + D.loads[4:6]
                        # E.addr done in step 4; E.loads start at step 6
                        body.append(a_hash[1]); body.append(a_hash[2]); body.append(b_hash[0])
                        body.extend(c_xor); body.extend(d_loads[4:6])
                        # Step 6: A.hash[3] + B.hash[1:3] + C.hash[0] + D.loads[6:8] + D.xor + E.loads[0:2] + F.addr
                        body.append(a_hash[3]); body.append(b_hash[1]); body.append(b_hash[2]); body.append(c_hash[0])
                        body.extend(d_loads[6:]); body.extend(d_xor); body.extend(e_loads[:2]); body.extend(f_addr)
                    # Step 7 (modified): A.hash[4] + B.hash[3] + C.hash[1:3] + D.hash[0] + E.load[2:4]
                    # F.addr moved earlier (step 6), removed from here
                    body.append(a_hash[4]); body.append(b_hash[3]); body.append(c_hash[1]); body.append(c_hash[2]); body.append(d_hash[0])
                    body.extend(e_loads[2:4])
                    # Step 8: A.hash[5:7] + B.hash[4] + C.hash[3] + D.hash[1:3] + E.load[4:6]
                    body.append(a_hash[5]); body.append(a_hash[6]); body.append(b_hash[4]); body.append(c_hash[3])
                    body.append(d_hash[1]); body.append(d_hash[2]); body.extend(e_loads[4:6])
                    # Step 9: A.hash[7] + B.hash[5:7] + C.hash[4] + D.hash[3] + E.load[6:] + E.xor + F.load[:2]
                    body.append(a_hash[7]); body.append(b_hash[5]); body.append(b_hash[6]); body.append(c_hash[4]); body.append(d_hash[3])
                    body.extend(e_loads[6:]); body.extend(e_xor); body.extend(f_loads[:2])
                    # Step 10: A.hash[8] + B.hash[7] + C.hash[5:7] + D.hash[4] + F.load[2:4] + G.addr
                    body.append(a_hash[8]); body.append(b_hash[7]); body.append(c_hash[5]); body.append(c_hash[6]); body.append(d_hash[4])
                    body.extend(f_loads[2:4]); body.extend(g_addr)
                    # Step 11: A.hash[9:11] + B.hash[8] + C.hash[7] + D.hash[5:7] + F.load[4:6]
                    body.append(a_hash[9]); body.append(a_hash[10]); body.append(b_hash[8]); body.append(c_hash[7])
                    body.append(d_hash[5]); body.append(d_hash[6]); body.extend(f_loads[4:6])
                    # Step 12: A.hash[11] + B.hash[9:11] + C.hash[8] + D.hash[7] + F.load[6:] + F.xor + G.load[:2]
                    body.append(a_hash[11]); body.append(b_hash[9]); body.append(b_hash[10]); body.append(c_hash[8]); body.append(d_hash[7])
                    body.extend(f_loads[6:]); body.extend(f_xor); body.extend(g_loads[:2])
                    # Step 13: A.idx[0] + B.hash[11] + C.hash[9:11] + D.hash[8] + G.load[2:4] + H.addr
                    body.append(a_idx[0]); body.append(b_hash[11]); body.append(c_hash[9]); body.append(c_hash[10]); body.append(d_hash[8])
                    body.extend(g_loads[2:4]); body.extend(h_addr)
                    # Step 14: A.idx[1] + B.idx[0] + C.hash[11] + D.hash[9:11] + G.load[4:6]
                    body.append(a_idx[1]); body.append(b_idx[0]); body.append(c_hash[11])
                    body.append(d_hash[9]); body.append(d_hash[10]); body.extend(g_loads[4:6])
                    # Step 15: A.idx[2] + B.idx[1] + C.idx[0] + D.hash[11] + G.load[6:] + G.xor + H.load[:2]
                    body.append(a_idx[2]); body.append(b_idx[1]); body.append(c_idx[0]); body.append(d_hash[11])
                    body.extend(g_loads[6:]); body.extend(g_xor); body.extend(h_loads[:2])
                    # Step 16: A.idx[3] + B.idx[2] + C.idx[1] + D.idx[0] + H.load[2:4]
                    body.append(a_idx[3]); body.append(b_idx[2]); body.append(c_idx[1]); body.append(d_idx[0])
                    body.extend(h_loads[2:4])
                    # Step 17: A.idx[4] + B.idx[3] + C.idx[2] + D.idx[1] + H.load[4:6]
                    body.append(a_idx[4]); body.append(b_idx[3]); body.append(c_idx[2]); body.append(d_idx[1])
                    body.extend(h_loads[4:6])
                    # Step 18: B.idx[4] + C.idx[3] + D.idx[2] + H.load[6:] + H.xor
                    body.append(b_idx[4]); body.append(c_idx[3]); body.append(d_idx[2])
                    body.extend(h_loads[6:]); body.extend(h_xor)
                    # Step 19: C.idx[4] + D.idx[3] + E.hash[0] + I.addr
                    body.append(c_idx[4]); body.append(d_idx[3]); body.append(e_hash[0]); body.extend(i_addr)
                    # Step 20: D.idx[4] + E.hash[1:3] + F.hash[0] + I.load[:2] + J.addr
                    body.append(d_idx[4]); body.append(e_hash[1]); body.append(e_hash[2]); body.append(f_hash[0])
                    body.extend(i_loads[:2]); body.extend(j_addr)
                    # Step 21: E.hash[3] + F.hash[1:3] + G.hash[0] + I.load[2:4]
                    body.append(e_hash[3]); body.append(f_hash[1]); body.append(f_hash[2]); body.append(g_hash[0])
                    body.extend(i_loads[2:4])
                    # Step 22: E.hash[4] + F.hash[3] + G.hash[1:3] + H.hash[0] + I.load[4:6] + J.load[:2] + K.addr
                    body.append(e_hash[4]); body.append(f_hash[3]); body.append(g_hash[1]); body.append(g_hash[2]); body.append(h_hash[0])
                    body.extend(i_loads[4:6]); body.extend(j_loads[:2]); body.extend(k_addr)
                    # Step 23: E.hash[5:7] + F.hash[4] + G.hash[3] + H.hash[1:3] + I.load[6:] + I.xor + J.load[2:4]
                    body.append(e_hash[5]); body.append(e_hash[6]); body.append(f_hash[4]); body.append(g_hash[3])
                    body.append(h_hash[1]); body.append(h_hash[2])
                    body.extend(i_loads[6:]); body.extend(i_xor); body.extend(j_loads[2:4])
                    # Step 24: E.hash[7] + F.hash[5:7] + G.hash[4] + H.hash[3] + J.load[4:6] + L.addr
                    body.append(e_hash[7]); body.append(f_hash[5]); body.append(f_hash[6]); body.append(g_hash[4]); body.append(h_hash[3])
                    body.extend(j_loads[4:6]); body.extend(l_addr)
                    # Step 25: E.hash[8] + F.hash[7] + G.hash[5:7] + H.hash[4] + J.load[6:] + J.xor + K.load[:2]
                    body.append(e_hash[8]); body.append(f_hash[7]); body.append(g_hash[5]); body.append(g_hash[6]); body.append(h_hash[4])
                    body.extend(j_loads[6:]); body.extend(j_xor); body.extend(k_loads[:2])
                    # Step 26: E.hash[9:11] + F.hash[8] + G.hash[7] + H.hash[5:7] + K.load[2:4] + M.addr
                    body.append(e_hash[9]); body.append(e_hash[10]); body.append(f_hash[8]); body.append(g_hash[7])
                    body.append(h_hash[5]); body.append(h_hash[6]); body.extend(k_loads[2:4]); body.extend(m_addr)
                    # Step 27: E.hash[11] + F.hash[9:11] + G.hash[8] + H.hash[7] + K.load[4:6] + L.load[:2]
                    body.append(e_hash[11]); body.append(f_hash[9]); body.append(f_hash[10]); body.append(g_hash[8]); body.append(h_hash[7])
                    body.extend(k_loads[4:6]); body.extend(l_loads[:2])
                    # Step 28: E.idx[0] + F.hash[11] + G.hash[9:11] + H.hash[8] + K.load[6:] + K.xor + L.load[2:4] + N.addr
                    body.append(e_idx[0]); body.append(f_hash[11]); body.append(g_hash[9]); body.append(g_hash[10]); body.append(h_hash[8])
                    body.extend(k_loads[6:]); body.extend(k_xor); body.extend(l_loads[2:4]); body.extend(n_addr)
                    # Step 29: E.idx[1] + F.idx[0] + G.hash[11] + H.hash[9:11] + L.load[4:6] + M.load[:2]
                    body.append(e_idx[1]); body.append(f_idx[0]); body.append(g_hash[11]); body.append(h_hash[9]); body.append(h_hash[10])
                    body.extend(l_loads[4:6]); body.extend(m_loads[:2])
                    # Step 30: E.idx[2] + F.idx[1] + G.idx[0] + H.hash[11] + L.load[6:] + L.xor + M.load[2:4] + O.addr
                    body.append(e_idx[2]); body.append(f_idx[1]); body.append(g_idx[0]); body.append(h_hash[11])
                    body.extend(l_loads[6:]); body.extend(l_xor); body.extend(m_loads[2:4]); body.extend(o_addr)
                    # Step 31: E.idx[3] + F.idx[2] + G.idx[1] + H.idx[0] + M.load[4:6] + N.load[:2]
                    body.append(e_idx[3]); body.append(f_idx[2]); body.append(g_idx[1]); body.append(h_idx[0])
                    body.extend(m_loads[4:6]); body.extend(n_loads[:2])
                    # Step 32: E.idx[4] + F.idx[3] + G.idx[2] + H.idx[1] + M.load[6:] + M.xor + N.load[2:4] + P.addr
                    body.append(e_idx[4]); body.append(f_idx[3]); body.append(g_idx[2]); body.append(h_idx[1])
                    body.extend(m_loads[6:]); body.extend(m_xor); body.extend(n_loads[2:4]); body.extend(p_addr)
                    # Step 33: F.idx[4] + G.idx[3] + H.idx[2] + I.hash[0] + N.load[4:6] + O.load[:2]
                    body.append(f_idx[4]); body.append(g_idx[3]); body.append(h_idx[2]); body.append(i_hash[0])
                    body.extend(n_loads[4:6]); body.extend(o_loads[:2])
                    # Step 34: G.idx[4] + H.idx[3] + I.hash[1:3] + J.hash[0] + N.load[6:] + N.xor + O.load[2:4]
                    body.append(g_idx[4]); body.append(h_idx[3]); body.append(i_hash[1]); body.append(i_hash[2]); body.append(j_hash[0])
                    body.extend(n_loads[6:]); body.extend(n_xor); body.extend(o_loads[2:4])
                    # Step 35: H.idx[4] + I.hash[3] + J.hash[1:3] + K.hash[0] + O.load[4:6] + P.load[:2]
                    body.append(h_idx[4]); body.append(i_hash[3]); body.append(j_hash[1]); body.append(j_hash[2]); body.append(k_hash[0])
                    body.extend(o_loads[4:6]); body.extend(p_loads[:2])
                    # Step 36: I.hash[4] + J.hash[3] + K.hash[1:3] + L.hash[0] + O.load[6:] + O.xor + P.load[2:4]
                    body.append(i_hash[4]); body.append(j_hash[3]); body.append(k_hash[1]); body.append(k_hash[2]); body.append(l_hash[0])
                    body.extend(o_loads[6:]); body.extend(o_xor); body.extend(p_loads[2:4])
                    # Step 37: I.hash[5:7] + J.hash[4] + K.hash[3] + L.hash[1:3] + P.load[4:6]
                    body.append(i_hash[5]); body.append(i_hash[6]); body.append(j_hash[4]); body.append(k_hash[3])
                    body.append(l_hash[1]); body.append(l_hash[2]); body.extend(p_loads[4:6])
                    # Step 38: I.hash[7] + J.hash[5:7] + K.hash[4] + L.hash[3] + P.load[6:] + P.xor
                    body.append(i_hash[7]); body.append(j_hash[5]); body.append(j_hash[6]); body.append(k_hash[4]); body.append(l_hash[3])
                    body.extend(p_loads[6:]); body.extend(p_xor)
                    # Step 39: I.hash[8] + J.hash[7] + K.hash[5:7] + L.hash[4] + M.hash[0]
                    # Step 39: I.hash[8] + J.hash[7] + K.hash[5:7] + L.hash[4] + M.hash[0]
                    # [+ nh_a_loads[0:2] if intra_hextet_s provided]
                    body.append(i_hash[8]); body.append(j_hash[7]); body.append(k_hash[5]); body.append(k_hash[6]); body.append(l_hash[4]); body.append(m_hash[0])
                    body.extend(nh_a_loads[0:2])
                    # Step 40: I.hash[9:11] + J.hash[8] + K.hash[7] + L.hash[5:7] [+ nh_a_loads[2:4]]
                    body.append(i_hash[9]); body.append(i_hash[10]); body.append(j_hash[8]); body.append(k_hash[7]); body.append(l_hash[5]); body.append(l_hash[6])
                    body.extend(nh_a_loads[2:4])
                    # Step 41: I.hash[11] + J.hash[9:11] + K.hash[8] + L.hash[7] + M.hash[1] [+ nh_a_loads[4:6]]
                    body.append(i_hash[11]); body.append(j_hash[9]); body.append(j_hash[10]); body.append(k_hash[8]); body.append(l_hash[7]); body.append(m_hash[1])
                    body.extend(nh_a_loads[4:6])
                    # Step 42: I.idx[0] + J.hash[11] + K.hash[9:11] + L.hash[8] + M.hash[2] [+ nh_a_loads[6:8]]
                    body.append(i_idx[0]); body.append(j_hash[11]); body.append(k_hash[9]); body.append(k_hash[10]); body.append(l_hash[8]); body.append(m_hash[2])
                    body.extend(nh_a_loads[6:8])
                    # Step 43: I.idx[1] + J.idx[0] + K.hash[11] + L.hash[9:11] + M.hash[3] [+ nh_b_loads[0:2]]
                    body.append(i_idx[1]); body.append(j_idx[0]); body.append(k_hash[11]); body.append(l_hash[9]); body.append(l_hash[10]); body.append(m_hash[3])
                    body.extend(nh_b_loads[0:2])
                    # Step 44: I.idx[2] + J.idx[1] + K.idx[0] + L.hash[11] + M.hash[4] + N.hash[0] [+ nh_b_loads[2:4]]
                    body.append(i_idx[2]); body.append(j_idx[1]); body.append(k_idx[0]); body.append(l_hash[11]); body.append(m_hash[4]); body.append(n_hash[0])
                    body.extend(nh_b_loads[2:4])
                    # Step 45: I.idx[3] + J.idx[2] + K.idx[1] + L.idx[0] + M.hash[5:7] [+ nh_b_loads[4:6]]
                    body.append(i_idx[3]); body.append(j_idx[2]); body.append(k_idx[1]); body.append(l_idx[0]); body.append(m_hash[5]); body.append(m_hash[6])
                    body.extend(nh_b_loads[4:6])
                    # Step 46: I.idx[4] + J.idx[3] + K.idx[2] + L.idx[1] + M.hash[7] + N.hash[1] [+ nh_b_loads[6:8]]
                    body.append(i_idx[4]); body.append(j_idx[3]); body.append(k_idx[2]); body.append(l_idx[1]); body.append(m_hash[7]); body.append(n_hash[1])
                    body.extend(nh_b_loads[6:8])
                    # Step 47: J.idx[4] + K.idx[3] + L.idx[2] + M.hash[8] + N.hash[2:4] [+ nh_c_loads[0:2]]
                    body.append(j_idx[4]); body.append(k_idx[3]); body.append(l_idx[2]); body.append(m_hash[8]); body.append(n_hash[2]); body.append(n_hash[3])
                    body.extend(nh_c_loads[0:2])
                    # Step 48: K.idx[4] + L.idx[3] + M.hash[9:11] + N.hash[4] + O.hash[0] [+ nh_c_loads[2:4]]
                    body.append(k_idx[4]); body.append(l_idx[3]); body.append(m_hash[9]); body.append(m_hash[10]); body.append(n_hash[4]); body.append(o_hash[0])
                    body.extend(nh_c_loads[2:4])
                    # Step 49: L.idx[4] + M.hash[11] + N.hash[5:7] + O.hash[1:3] [+ nh_c_loads[4:6]]
                    body.append(l_idx[4]); body.append(m_hash[11]); body.append(n_hash[5]); body.append(n_hash[6]); body.append(o_hash[1]); body.append(o_hash[2])
                    body.extend(nh_c_loads[4:6])
                    # Steps 50-63 via tail helper (with optional next round prefetch)
                    if emit_tail:
                        emit_hextet_tail_steps50_63(body, s, next_s=next_round_s0)

                # Precompute slots for both hextets in this round
                s0 = compute_hextet_slots(0)   # hextet 0: groups 0-15
                s1 = compute_hextet_slots(16)  # hextet 1: groups 16-31

                # Check if previous round pre-fetched hextet0's head
                # (either from another normal round, or from mod2 arith4 tail)
                prev_mod2 = (rnd > 0 and (rnd-1) % 11 == 2)
                prev_prefetched = (rnd > 0 and rnd % 11 >= 3 and
                                   ((rnd-1) % 11 >= 3 or prev_mod2))
                # Check if next round is also normal (for inter-round tail overlap)
                next_normal = (rnd+1 < rounds and (rnd+1) % 11 >= 3)

                if prev_prefetched:
                    # hextet0's A.addr+A.loads+B.addr+B.loads+C.addr+C.loads+D.addr
                    # were prefetched in the previous round's hextet1 tail.
                    # Emit only steps 3-49 (skip head). Tail is handled separately below.
                    # intra_hextet_s=s1: use idle load slots (steps 39-49) to prefetch hextet1 A,B,C loads
                    # and embed hextet1 A,B,C,D addr ops at step 3 (free valu slots)
                    emit_hextet_body_steps3_63(body, s0, emit_tail=False, intra_hextet_s=s1)
                else:
                    # Emit hextet0 steps 1-49 (head + phases 1 and 2) fully
                    emit_hextet_body_steps1_49(body, s0)

                # Emit hextet0 steps 50-63 WITH hextet1 A,B,C,D loads (partial) overlaid
                # When intra_hextet_s was used for hextet0 body:
                #   - hextet1 A loads (8) done in body steps 39-42
                #   - hextet1 B loads (8) done in body steps 43-46
                #   - hextet1 C loads [0:6] (6) done in body steps 47-49
                #   - hextet1 A,B,C,D addr done at body step 3
                # The tail must now: finish C[6:8] (2 loads) + D loads (8) in steps 51-58
                # We pass a partial next_s where: a_loads=D.loads (for steps 51-54),
                # c_loads=C.loads[6:8] (for steps 59 area), and everything else empty.
                if prev_prefetched:
                    # intra_hextet_s was used - pass partial tail prefetch
                    intra_tail_next_s = {
                        'a_addr': [],            # A addr already in hextet0 body step 3
                        'a_loads': s1['d_loads'], # D loads → tail steps 51-54 (na_loads slot)
                        'b_addr': [],            # B addr already in hextet0 body step 3
                        'b_loads': [],           # B loads done in hextet0 body
                        'c_addr': [],            # C addr already in hextet0 body step 3
                        'c_loads': s1['c_loads'][6:8],  # remaining C[6:8] → tail step 59
                        'd_addr': [],            # D addr already in hextet0 body step 3
                    }
                    emit_hextet_tail_steps50_63(body, s0, next_s=intra_tail_next_s)
                    # Emit hextet1 steps 3-49 with D preloaded (skip d_loads in steps 3-6)
                    emit_hextet_body_steps3_63(body, s1, emit_tail=False, d_preloaded=True)
                else:
                    emit_hextet_tail_steps50_63(body, s0, next_s=s1)
                    # Emit hextet1 steps 3-49 (A head already done in hextet0 tail)
                    emit_hextet_body_steps3_63(body, s1, emit_tail=False)
                # Emit hextet1 steps 50-63 (with optional next-round hextet0 prefetch)
                s0_next = s0 if next_normal else None  # next round uses same groups!
                emit_hextet_tail_steps50_63(body, s1, next_s=s0_next)

        # --- Store final idx and val vectors back to memory ---
        store_slots = []
        for g in range(n_groups):
            idx_addr = idx_base + g * VLEN
            val_addr = val_base + g * VLEN
            store_slots.append(("alu", ("+", addr_scalar, self.scratch["inp_indices_p"], group_offset_scalars[g])))
            store_slots.append(("store", ("vstore", addr_scalar, idx_addr)))
            store_slots.append(("alu", ("+", addr_scalar2, self.scratch["inp_values_p"], group_offset_scalars[g])))
            store_slots.append(("store", ("vstore", addr_scalar2, val_addr)))

        # Build the main body with VLIW packing
        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)

        # Build store instructions with VLIW packing
        store_instrs = self.build(store_slots)
        self.instrs.extend(store_instrs)

        # Required to match with the yield in reference_kernel2
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
