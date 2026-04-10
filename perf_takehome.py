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
            (zero_scalar, 0), (sc_mul_4097, 4097),
            (sc_hash0_const, 0x7ED55D16), (sc_hash1_const, 0xC761C23C),
            (sc_shift_19, 19), (sc_mul_33, 33),
            (sc_hash2_const, 0x165667B1), (sc_hash3_const, 0xD3A2646C),
            (sc_shift_9, 9), (sc_mul_9, 9),
            (sc_hash4_const, 0xFD7046C5), (sc_hash5_const, 0xB55A4F09),
            (sc_shift_16, 16),
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
            """Phase 5: index update (5 valu slots)"""
            idx_s = idx_base + g * VLEN
            val_s = val_base + g * VLEN
            lsb = lsb_tmp_pair[p]
            cmp = cmp_tmp_pair[p]
            slots = []
            # lsb = val & 1
            slots.append(("valu", ("&", lsb, val_s, one_vec)))
            # offset = lsb + 1
            slots.append(("valu", ("+", lsb, lsb, one_vec)))
            # new_idx = idx * 2 + offset
            slots.append(("valu", ("multiply_add", idx_s, idx_s, two_vec, lsb)))
            # cmp = new_idx < n_nodes
            slots.append(("valu", ("<", cmp, idx_s, n_nodes_vec)))
            # idx = new_idx * cmp
            slots.append(("valu", ("*", idx_s, idx_s, cmp)))
            return slots

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
            # Process groups in hextets of 16 (n_groups=32 divides evenly by 16)
            for g in range(0, n_groups, 16):
                gA, gB, gC, gD = g,   g+1, g+2, g+3
                gE, gF, gG, gH = g+4, g+5, g+6, g+7
                gI, gJ, gK, gL = g+8, g+9, g+10, g+11
                gM, gN, gO, gP = g+12, g+13, g+14, g+15

                # All groups reuse pairs 0-3 in rotation (mod 4 within each quartet)
                pA, pB, pC, pD = 0, 1, 2, 3
                pE, pF, pG, pH = 0, 1, 2, 3
                pI, pJ, pK, pL = 0, 1, 2, 3
                pM, pN, pO, pP = 0, 1, 2, 3

                # Pre-compute all phase slots for all 16 groups
                a_addr = group_addr_slots(gA)
                b_addr = group_addr_slots(gB)
                c_addr = group_addr_slots(gC)
                d_addr = group_addr_slots(gD)
                e_addr = group_addr_slots(gE)
                f_addr = group_addr_slots(gF)
                g_addr = group_addr_slots(gG)
                h_addr = group_addr_slots(gH)
                i_addr = group_addr_slots(gI)
                j_addr = group_addr_slots(gJ)
                k_addr = group_addr_slots(gK)
                l_addr = group_addr_slots(gL)
                m_addr = group_addr_slots(gM)
                n_addr = group_addr_slots(gN)
                o_addr = group_addr_slots(gO)
                p_addr = group_addr_slots(gP)

                a_loads = group_load_slots(gA)
                b_loads = group_load_slots(gB)
                c_loads = group_load_slots(gC)
                d_loads = group_load_slots(gD)
                e_loads = group_load_slots(gE)
                f_loads = group_load_slots(gF)
                g_loads = group_load_slots(gG)
                h_loads = group_load_slots(gH)
                i_loads = group_load_slots(gI)
                j_loads = group_load_slots(gJ)
                k_loads = group_load_slots(gK)
                l_loads = group_load_slots(gL)
                m_loads = group_load_slots(gM)
                n_loads = group_load_slots(gN)
                o_loads = group_load_slots(gO)
                p_loads = group_load_slots(gP)

                a_xor = group_xor_slots(gA)
                b_xor = group_xor_slots(gB)
                c_xor = group_xor_slots(gC)
                d_xor = group_xor_slots(gD)
                e_xor = group_xor_slots(gE)
                f_xor = group_xor_slots(gF)
                g_xor = group_xor_slots(gG)
                h_xor = group_xor_slots(gH)
                i_xor = group_xor_slots(gI)
                j_xor = group_xor_slots(gJ)
                k_xor = group_xor_slots(gK)
                l_xor = group_xor_slots(gL)
                m_xor = group_xor_slots(gM)
                n_xor = group_xor_slots(gN)
                o_xor = group_xor_slots(gO)
                p_xor = group_xor_slots(gP)

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

                # ---- 16-group extended pipeline schedule ----
                # Steps 1-18: A-H addr+load+xor (identical to 8-group octet steps 1-18)
                # Steps 19-35: E-H hash diagonal + I-P loads (load engine was idle here in 8-group!)
                # Steps 36+: E-H idx + I-P xor+hash+idx

                # === PHASE 1: A-H startup (identical to 8-group steps 1-18) ===

                # Step 1: A.addr
                body.extend(a_addr)

                # Step 2: A.load[:6] + B.addr + A.load[6:]
                body.extend(a_loads[:6])
                body.extend(b_addr)
                body.extend(a_loads[6:])

                # Step 3: A.xor + B.load[:6] + C.addr + B.load[6:]
                body.extend(a_xor)
                body.extend(b_loads[:6])
                body.extend(c_addr)
                body.extend(b_loads[6:])

                # Step 4: A.hash[0] + B.xor + C.load[:6] + D.addr + C.load[6:]
                body.append(a_hash[0])
                body.extend(b_xor)
                body.extend(c_loads[:6])
                body.extend(d_addr)
                body.extend(c_loads[6:])

                # Step 5: A.hash[1:3] + B.hash[0] + C.xor + D.load[:6] + E.addr
                body.append(a_hash[1])
                body.append(a_hash[2])
                body.append(b_hash[0])
                body.extend(c_xor)
                body.extend(d_loads[:6])
                body.extend(e_addr)

                # Step 6: A.hash[3] + B.hash[1:3] + C.hash[0] + D.load[6:] + D.xor + E.load[:2]
                body.append(a_hash[3])
                body.append(b_hash[1])
                body.append(b_hash[2])
                body.append(c_hash[0])
                body.extend(d_loads[6:])
                body.extend(d_xor)
                body.extend(e_loads[:2])

                # Step 7: A.hash[4] + B.hash[3] + C.hash[1:3] + D.hash[0] + E.load[2:4] + F.addr
                body.append(a_hash[4])
                body.append(b_hash[3])
                body.append(c_hash[1])
                body.append(c_hash[2])
                body.append(d_hash[0])
                body.extend(e_loads[2:4])
                body.extend(f_addr)

                # Step 8: A.hash[5:7] + B.hash[4] + C.hash[3] + D.hash[1:3] + E.load[4:6]
                body.append(a_hash[5])
                body.append(a_hash[6])
                body.append(b_hash[4])
                body.append(c_hash[3])
                body.append(d_hash[1])
                body.append(d_hash[2])
                body.extend(e_loads[4:6])

                # Step 9: A.hash[7] + B.hash[5:7] + C.hash[4] + D.hash[3] + E.load[6:] + E.xor + F.load[:2]
                body.append(a_hash[7])
                body.append(b_hash[5])
                body.append(b_hash[6])
                body.append(c_hash[4])
                body.append(d_hash[3])
                body.extend(e_loads[6:])
                body.extend(e_xor)
                body.extend(f_loads[:2])

                # Step 10: A.hash[8] + B.hash[7] + C.hash[5:7] + D.hash[4] + F.load[2:4] + G.addr
                body.append(a_hash[8])
                body.append(b_hash[7])
                body.append(c_hash[5])
                body.append(c_hash[6])
                body.append(d_hash[4])
                body.extend(f_loads[2:4])
                body.extend(g_addr)

                # Step 11: A.hash[9:11] + B.hash[8] + C.hash[7] + D.hash[5:7] + F.load[4:6]
                body.append(a_hash[9])
                body.append(a_hash[10])
                body.append(b_hash[8])
                body.append(c_hash[7])
                body.append(d_hash[5])
                body.append(d_hash[6])
                body.extend(f_loads[4:6])

                # Step 12: A.hash[11] + B.hash[9:11] + C.hash[8] + D.hash[7] + F.load[6:] + F.xor + G.load[:2]
                body.append(a_hash[11])
                body.append(b_hash[9])
                body.append(b_hash[10])
                body.append(c_hash[8])
                body.append(d_hash[7])
                body.extend(f_loads[6:])
                body.extend(f_xor)
                body.extend(g_loads[:2])

                # Step 13: A.idx[0] + B.hash[11] + C.hash[9:11] + D.hash[8] + G.load[2:4] + H.addr
                body.append(a_idx[0])
                body.append(b_hash[11])
                body.append(c_hash[9])
                body.append(c_hash[10])
                body.append(d_hash[8])
                body.extend(g_loads[2:4])
                body.extend(h_addr)

                # Step 14: A.idx[1] + B.idx[0] + C.hash[11] + D.hash[9:11] + G.load[4:6]
                body.append(a_idx[1])
                body.append(b_idx[0])
                body.append(c_hash[11])
                body.append(d_hash[9])
                body.append(d_hash[10])
                body.extend(g_loads[4:6])

                # Step 15: A.idx[2] + B.idx[1] + C.idx[0] + D.hash[11] + G.load[6:] + G.xor + H.load[:2]
                body.append(a_idx[2])
                body.append(b_idx[1])
                body.append(c_idx[0])
                body.append(d_hash[11])
                body.extend(g_loads[6:])
                body.extend(g_xor)
                body.extend(h_loads[:2])

                # Step 16: A.idx[3] + B.idx[2] + C.idx[1] + D.idx[0] + H.load[2:4]
                body.append(a_idx[3])
                body.append(b_idx[2])
                body.append(c_idx[1])
                body.append(d_idx[0])
                body.extend(h_loads[2:4])

                # Step 17: A.idx[4] + B.idx[3] + C.idx[2] + D.idx[1] + H.load[4:6]
                body.append(a_idx[4])
                body.append(b_idx[3])
                body.append(c_idx[2])
                body.append(d_idx[1])
                body.extend(h_loads[4:6])

                # Step 18: B.idx[4] + C.idx[3] + D.idx[2] + H.load[6:] + H.xor
                body.append(b_idx[4])
                body.append(c_idx[3])
                body.append(d_idx[2])
                body.extend(h_loads[6:])
                body.extend(h_xor)

                # === PHASE 2: E-H hash + I-P loads (load engine was IDLE in 8-group!) ===
                # I-P's addr slots (valu) are interleaved with E-H hash at these steps.
                # I.addr depends only on idx_base+gI (already valid). No conflict with E-H hash.
                # I.load depends on I.addr (i_addr writes addr_tmp_g[gI]).

                # Step 19: C.idx[4] + D.idx[3] + E.hash[0] + I.addr
                body.append(c_idx[4])
                body.append(d_idx[3])
                body.append(e_hash[0])
                body.extend(i_addr)

                # Step 20: D.idx[4] + E.hash[1:3] + F.hash[0] + I.load[:2] + J.addr
                body.append(d_idx[4])
                body.append(e_hash[1])
                body.append(e_hash[2])
                body.append(f_hash[0])
                body.extend(i_loads[:2])
                body.extend(j_addr)

                # Step 21: E.hash[3] + F.hash[1:3] + G.hash[0] + I.load[2:4]
                body.append(e_hash[3])
                body.append(f_hash[1])
                body.append(f_hash[2])
                body.append(g_hash[0])
                body.extend(i_loads[2:4])

                # Step 22: E.hash[4] + F.hash[3] + G.hash[1:3] + H.hash[0] + I.load[4:6] + J.load[:2] + K.addr
                body.append(e_hash[4])
                body.append(f_hash[3])
                body.append(g_hash[1])
                body.append(g_hash[2])
                body.append(h_hash[0])
                body.extend(i_loads[4:6])
                body.extend(j_loads[:2])
                body.extend(k_addr)

                # Step 23: E.hash[5:7] + F.hash[4] + G.hash[3] + H.hash[1:3] + I.load[6:] + I.xor + J.load[2:4]
                body.append(e_hash[5])
                body.append(e_hash[6])
                body.append(f_hash[4])
                body.append(g_hash[3])
                body.append(h_hash[1])
                body.append(h_hash[2])
                body.extend(i_loads[6:])
                body.extend(i_xor)
                body.extend(j_loads[2:4])

                # Step 24: E.hash[7] + F.hash[5:7] + G.hash[4] + H.hash[3] + J.load[4:6] + L.addr
                body.append(e_hash[7])
                body.append(f_hash[5])
                body.append(f_hash[6])
                body.append(g_hash[4])
                body.append(h_hash[3])
                body.extend(j_loads[4:6])
                body.extend(l_addr)

                # Step 25: E.hash[8] + F.hash[7] + G.hash[5:7] + H.hash[4] + J.load[6:] + J.xor + K.load[:2]
                body.append(e_hash[8])
                body.append(f_hash[7])
                body.append(g_hash[5])
                body.append(g_hash[6])
                body.append(h_hash[4])
                body.extend(j_loads[6:])
                body.extend(j_xor)
                body.extend(k_loads[:2])

                # Step 26: E.hash[9:11] + F.hash[8] + G.hash[7] + H.hash[5:7] + K.load[2:4] + M.addr
                body.append(e_hash[9])
                body.append(e_hash[10])
                body.append(f_hash[8])
                body.append(g_hash[7])
                body.append(h_hash[5])
                body.append(h_hash[6])
                body.extend(k_loads[2:4])
                body.extend(m_addr)

                # Step 27: E.hash[11] + F.hash[9:11] + G.hash[8] + H.hash[7] + K.load[4:6] + L.load[:2]
                body.append(e_hash[11])
                body.append(f_hash[9])
                body.append(f_hash[10])
                body.append(g_hash[8])
                body.append(h_hash[7])
                body.extend(k_loads[4:6])
                body.extend(l_loads[:2])

                # Step 28: E.idx[0] + F.hash[11] + G.hash[9:11] + H.hash[8] + K.load[6:] + K.xor + L.load[2:4] + N.addr
                body.append(e_idx[0])
                body.append(f_hash[11])
                body.append(g_hash[9])
                body.append(g_hash[10])
                body.append(h_hash[8])
                body.extend(k_loads[6:])
                body.extend(k_xor)
                body.extend(l_loads[2:4])
                body.extend(n_addr)

                # Step 29: E.idx[1] + F.idx[0] + G.hash[11] + H.hash[9:11] + L.load[4:6] + M.load[:2]
                body.append(e_idx[1])
                body.append(f_idx[0])
                body.append(g_hash[11])
                body.append(h_hash[9])
                body.append(h_hash[10])
                body.extend(l_loads[4:6])
                body.extend(m_loads[:2])

                # Step 30: E.idx[2] + F.idx[1] + G.idx[0] + H.hash[11] + L.load[6:] + L.xor + M.load[2:4] + O.addr
                body.append(e_idx[2])
                body.append(f_idx[1])
                body.append(g_idx[0])
                body.append(h_hash[11])
                body.extend(l_loads[6:])
                body.extend(l_xor)
                body.extend(m_loads[2:4])
                body.extend(o_addr)

                # Step 31: E.idx[3] + F.idx[2] + G.idx[1] + H.idx[0] + M.load[4:6] + N.load[:2]
                body.append(e_idx[3])
                body.append(f_idx[2])
                body.append(g_idx[1])
                body.append(h_idx[0])
                body.extend(m_loads[4:6])
                body.extend(n_loads[:2])

                # Step 32: E.idx[4] + F.idx[3] + G.idx[2] + H.idx[1] + M.load[6:] + M.xor + N.load[2:4] + P.addr
                body.append(e_idx[4])
                body.append(f_idx[3])
                body.append(g_idx[2])
                body.append(h_idx[1])
                body.extend(m_loads[6:])
                body.extend(m_xor)
                body.extend(n_loads[2:4])
                body.extend(p_addr)

                # Step 33: F.idx[4] + G.idx[3] + H.idx[2] + I.hash[0] + N.load[4:6] + O.load[:2]
                body.append(f_idx[4])
                body.append(g_idx[3])
                body.append(h_idx[2])
                body.append(i_hash[0])
                body.extend(n_loads[4:6])
                body.extend(o_loads[:2])

                # Step 34: G.idx[4] + H.idx[3] + I.hash[1:3] + J.hash[0] + N.load[6:] + N.xor + O.load[2:4]
                body.append(g_idx[4])
                body.append(h_idx[3])
                body.append(i_hash[1])
                body.append(i_hash[2])
                body.append(j_hash[0])
                body.extend(n_loads[6:])
                body.extend(n_xor)
                body.extend(o_loads[2:4])

                # Step 35: H.idx[4] + I.hash[3] + J.hash[1:3] + K.hash[0] + O.load[4:6] + P.load[:2]
                body.append(h_idx[4])
                body.append(i_hash[3])
                body.append(j_hash[1])
                body.append(j_hash[2])
                body.append(k_hash[0])
                body.extend(o_loads[4:6])
                body.extend(p_loads[:2])

                # === PHASE 3: I-P hash+idx diagonal + final loads ===

                # Step 36: I.hash[4] + J.hash[3] + K.hash[1:3] + L.hash[0] + O.load[6:] + O.xor + P.load[2:4]
                body.append(i_hash[4])
                body.append(j_hash[3])
                body.append(k_hash[1])
                body.append(k_hash[2])
                body.append(l_hash[0])
                body.extend(o_loads[6:])
                body.extend(o_xor)
                body.extend(p_loads[2:4])

                # Step 37: I.hash[5:7] + J.hash[4] + K.hash[3] + L.hash[1:3] + P.load[4:6]
                body.append(i_hash[5])
                body.append(i_hash[6])
                body.append(j_hash[4])
                body.append(k_hash[3])
                body.append(l_hash[1])
                body.append(l_hash[2])
                body.extend(p_loads[4:6])

                # Step 38: I.hash[7] + J.hash[5:7] + K.hash[4] + L.hash[3] + P.load[6:] + P.xor
                body.append(i_hash[7])
                body.append(j_hash[5])
                body.append(j_hash[6])
                body.append(k_hash[4])
                body.append(l_hash[3])
                body.extend(p_loads[6:])
                body.extend(p_xor)

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

                # Step 50: M.idx[0] + N.hash[7] + O.hash[3] + P.hash[0]
                body.append(m_idx[0])
                body.append(n_hash[7])
                body.append(o_hash[3])
                body.append(p_hash[0])

                # Step 51: M.idx[1] + N.hash[8] + O.hash[4] + P.hash[1:3]
                body.append(m_idx[1])
                body.append(n_hash[8])
                body.append(o_hash[4])
                body.append(p_hash[1])
                body.append(p_hash[2])

                # Step 52: M.idx[2] + N.hash[9:11] + O.hash[5:7] + P.hash[3]
                body.append(m_idx[2])
                body.append(n_hash[9])
                body.append(n_hash[10])
                body.append(o_hash[5])
                body.append(o_hash[6])
                body.append(p_hash[3])

                # Step 53: M.idx[3] + N.hash[11] + O.hash[7] + P.hash[4]
                body.append(m_idx[3])
                body.append(n_hash[11])
                body.append(o_hash[7])
                body.append(p_hash[4])

                # Step 54: M.idx[4] + N.idx[0] + O.hash[8] + P.hash[5:7]
                body.append(m_idx[4])
                body.append(n_idx[0])
                body.append(o_hash[8])
                body.append(p_hash[5])
                body.append(p_hash[6])

                # Step 55: N.idx[1] + O.hash[9:11] + P.hash[7]
                body.append(n_idx[1])
                body.append(o_hash[9])
                body.append(o_hash[10])
                body.append(p_hash[7])

                # Step 56: N.idx[2] + O.hash[11] + P.hash[8]
                body.append(n_idx[2])
                body.append(o_hash[11])
                body.append(p_hash[8])

                # Step 57: N.idx[3] + O.idx[0] + P.hash[9:11]
                body.append(n_idx[3])
                body.append(o_idx[0])
                body.append(p_hash[9])
                body.append(p_hash[10])

                # Step 58: N.idx[4] + O.idx[1] + P.hash[11]
                body.append(n_idx[4])
                body.append(o_idx[1])
                body.append(p_hash[11])

                # Step 59: O.idx[2] + P.idx[0]
                body.append(o_idx[2])
                body.append(p_idx[0])

                # Step 60: O.idx[3] + P.idx[1]
                body.append(o_idx[3])
                body.append(p_idx[1])

                # Step 61: O.idx[4] + P.idx[2]
                body.append(o_idx[4])
                body.append(p_idx[2])

                # Step 62: P.idx[3]
                body.append(p_idx[3])

                # Step 63: P.idx[4]
                body.append(p_idx[4])

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
