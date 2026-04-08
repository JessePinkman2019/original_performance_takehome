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
            self.add("load", ("const", addr, val))
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
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # --- Allocate vector scratch regions ---
        # idx vectors: n_groups groups of VLEN words each
        idx_base = self.alloc_scratch("idx_base", n_groups * VLEN)
        # val vectors: n_groups groups of VLEN words each
        val_base = self.alloc_scratch("val_base", n_groups * VLEN)

        # Temp vector registers (each VLEN words)
        addr_tmp = self.alloc_scratch("addr_tmp", VLEN)
        nv_tmp = self.alloc_scratch("nv_tmp", VLEN)
        t1_tmp = self.alloc_scratch("t1_tmp", VLEN)
        t2_tmp = self.alloc_scratch("t2_tmp", VLEN)
        lsb_tmp = self.alloc_scratch("lsb_tmp", VLEN)
        cmp_tmp = self.alloc_scratch("cmp_tmp", VLEN)
        zero_vec = self.alloc_scratch("zero_vec", VLEN)

        # --- Scalar constants needed for address computation ---
        # We need a scalar scratch for each group's memory offset for vload/vstore
        # These are computed as: inp_indices_p + g * VLEN, inp_values_p + g * VLEN

        # --- Constant vectors (broadcast from scalar) ---
        # We allocate scalar constants first, then broadcast them to vectors

        # Scalar constants
        sc_forest_values_p = self.scratch["forest_values_p"]  # already loaded
        sc_n_nodes = self.scratch["n_nodes"]  # already loaded
        one_scalar = self.scratch_const(1)
        two_scalar = self.scratch_const(2)
        zero_scalar = self.scratch_const(0)

        # Vector constants (broadcast from scalars)
        fvp_vec = self.alloc_scratch("fvp_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)

        # Hash stage constant vectors
        # Stage 0: multiply_add with multiplier=4097, constant=0x7ED55D16
        sc_mul_4097 = self.scratch_const(4097, "mul_4097")
        mul_4097_vec = self.alloc_scratch("mul_4097_vec", VLEN)
        sc_hash0_const = self.scratch_const(0x7ED55D16, "hash0_const")
        hash0_const_vec = self.alloc_scratch("hash0_const_vec", VLEN)

        # Stage 1: xor with 0xC761C23C, xor with >>19
        sc_hash1_const = self.scratch_const(0xC761C23C, "hash1_const")
        hash1_const_vec = self.alloc_scratch("hash1_const_vec", VLEN)
        sc_shift_19 = self.scratch_const(19, "shift_19")
        shift_19_vec = self.alloc_scratch("shift_19_vec", VLEN)

        # Stage 2: multiply_add with multiplier=33, constant=0x165667B1
        sc_mul_33 = self.scratch_const(33, "mul_33")
        mul_33_vec = self.alloc_scratch("mul_33_vec", VLEN)
        sc_hash2_const = self.scratch_const(0x165667B1, "hash2_const")
        hash2_const_vec = self.alloc_scratch("hash2_const_vec", VLEN)

        # Stage 3: + 0xD3A2646C, xor with <<9
        sc_hash3_const = self.scratch_const(0xD3A2646C, "hash3_const")
        hash3_const_vec = self.alloc_scratch("hash3_const_vec", VLEN)
        sc_shift_9 = self.scratch_const(9, "shift_9")
        shift_9_vec = self.alloc_scratch("shift_9_vec", VLEN)

        # Stage 4: multiply_add with multiplier=9, constant=0xFD7046C5
        sc_mul_9 = self.scratch_const(9, "mul_9")
        mul_9_vec = self.alloc_scratch("mul_9_vec", VLEN)
        sc_hash4_const = self.scratch_const(0xFD7046C5, "hash4_const")
        hash4_const_vec = self.alloc_scratch("hash4_const_vec", VLEN)

        # Stage 5: xor with 0xB55A4F09, xor with >>16
        sc_hash5_const = self.scratch_const(0xB55A4F09, "hash5_const")
        hash5_const_vec = self.alloc_scratch("hash5_const_vec", VLEN)
        sc_shift_16 = self.scratch_const(16, "shift_16")
        shift_16_vec = self.alloc_scratch("shift_16_vec", VLEN)

        # Scalar address temps for vload/vstore addressing
        addr_scalar = self.alloc_scratch("addr_scalar")
        addr_scalar2 = self.alloc_scratch("addr_scalar2")

        # --- Broadcast constant vectors ---
        self.add("valu", ("vbroadcast", fvp_vec, sc_forest_values_p))
        self.add("valu", ("vbroadcast", n_nodes_vec, sc_n_nodes))
        self.add("valu", ("vbroadcast", one_vec, one_scalar))
        self.add("valu", ("vbroadcast", two_vec, two_scalar))
        self.add("valu", ("vbroadcast", zero_vec, zero_scalar))

        self.add("valu", ("vbroadcast", mul_4097_vec, sc_mul_4097))
        self.add("valu", ("vbroadcast", hash0_const_vec, sc_hash0_const))
        self.add("valu", ("vbroadcast", hash1_const_vec, sc_hash1_const))
        self.add("valu", ("vbroadcast", shift_19_vec, sc_shift_19))
        self.add("valu", ("vbroadcast", mul_33_vec, sc_mul_33))
        self.add("valu", ("vbroadcast", hash2_const_vec, sc_hash2_const))
        self.add("valu", ("vbroadcast", hash3_const_vec, sc_hash3_const))
        self.add("valu", ("vbroadcast", shift_9_vec, sc_shift_9))
        self.add("valu", ("vbroadcast", mul_9_vec, sc_mul_9))
        self.add("valu", ("vbroadcast", hash4_const_vec, sc_hash4_const))
        self.add("valu", ("vbroadcast", hash5_const_vec, sc_hash5_const))
        self.add("valu", ("vbroadcast", shift_16_vec, sc_shift_16))

        # --- Load initial idx and val vectors from memory into scratch ---
        # Pre-compute group offset constants for address computation
        group_offset_scalars = []
        for g in range(n_groups):
            offset_val = g * VLEN
            sc = self.scratch_const(offset_val)
            group_offset_scalars.append(sc)

        init_load_slots = []
        for g in range(n_groups):
            idx_addr = idx_base + g * VLEN
            val_addr = val_base + g * VLEN
            # addr_scalar = inp_indices_p + g * VLEN
            init_load_slots.append(("alu", ("+", addr_scalar, self.scratch["inp_indices_p"], group_offset_scalars[g])))
            init_load_slots.append(("load", ("vload", idx_addr, addr_scalar)))
            # addr_scalar2 = inp_values_p + g * VLEN
            init_load_slots.append(("alu", ("+", addr_scalar2, self.scratch["inp_values_p"], group_offset_scalars[g])))
            init_load_slots.append(("load", ("vload", val_addr, addr_scalar2)))

        init_instrs = self.build(init_load_slots)
        self.instrs.extend(init_instrs)

        # Pause to match reference_kernel2 yield
        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting SIMD loop"))

        # --- Main computation loop (fully unrolled) ---
        body = []

        for rnd in range(rounds):
            for g in range(n_groups):
                idx_s = idx_base + g * VLEN  # scratch addr for this group's idx vector
                val_s = val_base + g * VLEN  # scratch addr for this group's val vector

                # --- Scatter load: node_val = mem[forest_values_p + idx[lane]] ---
                # Compute addresses: addr_tmp[lane] = fvp_vec[lane] + idx[lane]
                body.append(("valu", ("+", addr_tmp, idx_s, fvp_vec)))

                # Load each lane individually using load_offset
                for lane in range(VLEN):
                    body.append(("load", ("load_offset", nv_tmp, addr_tmp, lane)))

                # --- XOR: val = val ^ node_val ---
                body.append(("valu", ("^", val_s, val_s, nv_tmp)))

                # --- Hash computation (6 stages) ---

                # Stage 0: a = (a + 0x7ED55D16) + (a << 12) = a * 4097 + 0x7ED55D16
                # Use multiply_add: val = val * 4097 + hash0_const
                body.append(("valu", ("multiply_add", val_s, val_s, mul_4097_vec, hash0_const_vec)))

                # Stage 1: a = (a ^ 0xC761C23C) ^ (a >> 19)
                # t1 = a ^ const, t2 = a >> 19, a = t1 ^ t2
                body.append(("valu", ("^", t1_tmp, val_s, hash1_const_vec)))
                body.append(("valu", (">>", t2_tmp, val_s, shift_19_vec)))
                body.append(("valu", ("^", val_s, t1_tmp, t2_tmp)))

                # Stage 2: a = (a + 0x165667B1) + (a << 5) = a * 33 + 0x165667B1
                body.append(("valu", ("multiply_add", val_s, val_s, mul_33_vec, hash2_const_vec)))

                # Stage 3: a = (a + 0xD3A2646C) ^ (a << 9)
                # t1 = a + const, t2 = a << 9, a = t1 ^ t2
                body.append(("valu", ("+", t1_tmp, val_s, hash3_const_vec)))
                body.append(("valu", ("<<", t2_tmp, val_s, shift_9_vec)))
                body.append(("valu", ("^", val_s, t1_tmp, t2_tmp)))

                # Stage 4: a = (a + 0xFD7046C5) + (a << 3) = a * 9 + 0xFD7046C5
                body.append(("valu", ("multiply_add", val_s, val_s, mul_9_vec, hash4_const_vec)))

                # Stage 5: a = (a ^ 0xB55A4F09) ^ (a >> 16)
                body.append(("valu", ("^", t1_tmp, val_s, hash5_const_vec)))
                body.append(("valu", (">>", t2_tmp, val_s, shift_16_vec)))
                body.append(("valu", ("^", val_s, t1_tmp, t2_tmp)))

                # --- Post-hash: compute new index ---
                # lsb = val & 1
                body.append(("valu", ("&", lsb_tmp, val_s, one_vec)))
                # offset = lsb + 1 (so offset is 1 if odd, 2 if even)
                # Wait: val%2==0 means even, which gives offset=1. val%2==1 means odd, gives offset=2.
                # Original: idx = 2*idx + (1 if val%2==0 else 2)
                # lsb = val & 1 => 0 if even, 1 if odd
                # We want: offset = 1 if even (lsb=0), 2 if odd (lsb=1)
                # So: offset = lsb + 1
                body.append(("valu", ("+", lsb_tmp, lsb_tmp, one_vec)))
                # new_idx = idx * 2 + offset
                body.append(("valu", ("multiply_add", idx_s, idx_s, two_vec, lsb_tmp)))
                # Wrap: if new_idx >= n_nodes, set to 0
                # cmp = (new_idx < n_nodes) ? 1 : 0
                body.append(("valu", ("<", cmp_tmp, idx_s, n_nodes_vec)))
                # idx = new_idx * cmp (if cmp=0, idx becomes 0; if cmp=1, idx stays)
                body.append(("valu", ("*", idx_s, idx_s, cmp_tmp)))

        # --- Store final idx and val vectors back to memory ---
        store_slots = []
        for g in range(n_groups):
            idx_addr = idx_base + g * VLEN
            val_addr = val_base + g * VLEN
            # addr_scalar = inp_indices_p + g * VLEN
            store_slots.append(("alu", ("+", addr_scalar, self.scratch["inp_indices_p"], group_offset_scalars[g])))
            store_slots.append(("store", ("vstore", addr_scalar, idx_addr)))
            # addr_scalar2 = inp_values_p + g * VLEN
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
