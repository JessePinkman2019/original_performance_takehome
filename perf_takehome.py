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


class Op:
    __slots__ = ['engine', 'slot', 'deps', 'group_id', 'scheduled',
                 'priority', 'successors', 'id_num']
    _counter = 0

    def __init__(self, engine, slot, deps=None, group_id=None):
        self.engine = engine
        self.slot = slot
        self.deps = deps or []
        self.group_id = group_id
        self.scheduled = False
        self.priority = 0
        self.successors = []
        Op._counter += 1
        self.id_num = Op._counter


def list_schedule(ops):
    if not ops:
        return []

    for op in ops:
        for dep in op.deps:
            dep.successors.append(op)

    visited = set()
    topo_order = []
    stack = [(op, False) for op in ops]
    while stack:
        node, processed = stack.pop()
        nid = node.id_num
        if processed:
            if nid not in visited:
                visited.add(nid)
                topo_order.append(node)
            continue
        if nid in visited:
            continue
        stack.append((node, True))
        for succ in node.successors:
            if succ.id_num not in visited:
                stack.append((succ, False))

    for op in topo_order:
        if not op.successors:
            op.priority = 1
        else:
            op.priority = max(s.priority for s in op.successors) + 1

    n_total = len(ops)
    n_scheduled = 0

    dep_count = {}
    for op in ops:
        dep_count[op.id_num] = len(op.deps)

    dependents = defaultdict(list)
    for op in ops:
        for dep in op.deps:
            dependents[dep.id_num].append(op)

    # Separate ready lists by engine type
    ready_by_engine = defaultdict(list)
    for op in ops:
        if dep_count[op.id_num] == 0:
            ready_by_engine[op.engine].append(op)

    for eng in ready_by_engine:
        ready_by_engine[eng].sort(key=lambda o: -o.priority)

    bundles = []
    engine_order = ["load", "valu", "store", "alu", "flow"]

    while n_scheduled < n_total:
        bundle = {}
        scheduled_this_cycle = []
        new_ready_by_engine = defaultdict(list)

        for eng in engine_order:
            ready = ready_by_engine.get(eng, [])
            if not ready:
                continue
            limit = SLOT_LIMITS.get(eng, 0)
            count = 0
            for op in ready:
                if count >= limit:
                    new_ready_by_engine[eng].append(op)
                else:
                    bundle.setdefault(eng, []).append(op.slot)
                    count += 1
                    scheduled_this_cycle.append(op)

        for op in scheduled_this_cycle:
            op.scheduled = True
            n_scheduled += 1

        for op in scheduled_this_cycle:
            for dep_op in dependents[op.id_num]:
                dep_count[dep_op.id_num] -= 1
                if dep_count[dep_op.id_num] == 0:
                    new_ready_by_engine[dep_op.engine].append(dep_op)

        for eng in new_ready_by_engine:
            new_ready_by_engine[eng].sort(key=lambda o: -o.priority)

        ready_by_engine = new_ready_by_engine

        if bundle:
            bundles.append(bundle)

    return bundles


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch: {self.scratch_ptr}"
        return addr

    def alloc_const(self, val):
        if val not in self.const_map:
            addr = self.alloc_scratch()
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        # ===== PHASE 1: SCRATCH ALLOCATION =====
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        n_groups = batch_size // VLEN

        group_idx = [self.alloc_scratch(f"idx_g{g}", VLEN) for g in range(n_groups)]
        group_val = [self.alloc_scratch(f"val_g{g}", VLEN) for g in range(n_groups)]
        group_addr = [self.alloc_scratch(f"adr_g{g}", VLEN) for g in range(n_groups)]
        group_nv = [self.alloc_scratch(f"nv_g{g}", VLEN) for g in range(n_groups)]

        N_POOL = 16
        pool_t1 = [self.alloc_scratch(f"t1_{p}", VLEN) for p in range(N_POOL)]
        pool_t2 = [self.alloc_scratch(f"t2_{p}", VLEN) for p in range(N_POOL)]

        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        fvp_vec = self.alloc_scratch("fvp_vec", VLEN)

        hash_stage_info = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                factor = (1 + (1 << val3)) % (2**32)
                hash_stage_info.append(('madd', val1, factor))
            else:
                hash_stage_info.append(('3op', hi))

        hash_madd_factor_vecs = {}
        hash_madd_c1_vecs = {}
        hash_3op_vecs = {}
        for hi, info in enumerate(hash_stage_info):
            if info[0] == 'madd':
                hash_madd_factor_vecs[hi] = self.alloc_scratch(f"hf{hi}", VLEN)
                hash_madd_c1_vecs[hi] = self.alloc_scratch(f"hmc{hi}", VLEN)
            else:
                v1 = self.alloc_scratch(f"h3a{hi}", VLEN)
                v3 = self.alloc_scratch(f"h3b{hi}", VLEN)
                hash_3op_vecs[hi] = (v1, v3)

        gm_idx = self.alloc_scratch("gm_idx")
        gm_val = self.alloc_scratch("gm_val")

        # Pre-allocate constants
        zero_const = self.alloc_const(0)
        one_const = self.alloc_const(1)
        two_const = self.alloc_const(2)

        hash_consts = []
        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            c1 = self.alloc_const(val1)
            c3 = self.alloc_const(val3)
            hash_consts.append((op1, c1, op2, op3, c3))

        for hi, info in enumerate(hash_stage_info):
            if info[0] == 'madd':
                _, val1, factor = info
                self.alloc_const(factor)
                self.alloc_const(val1)

        vlen_const = self.alloc_const(VLEN)

        # Scratch for round-0 optimization
        nv_scalar = self.alloc_scratch("nv_scalar")

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")
        assert self.scratch_ptr <= SCRATCH_SIZE

        # ===== PHASE 2: EMIT INIT =====
        for i in range(0, len(init_vars), 2):
            if i + 1 < len(init_vars):
                self.instrs.append({"load": [("const", tmp1, i), ("const", tmp2, i + 1)]})
                self.instrs.append({"load": [
                    ("load", self.scratch[init_vars[i]], tmp1),
                    ("load", self.scratch[init_vars[i + 1]], tmp2),
                ]})
            else:
                self.instrs.append({"load": [("const", tmp1, i)]})
                self.instrs.append({"load": [("load", self.scratch[init_vars[i]], tmp1)]})

        const_loads = []
        for val, addr in sorted(self.const_map.items(), key=lambda x: x[1]):
            const_loads.append(("const", addr, val))
        for i in range(0, len(const_loads), 2):
            self.instrs.append({"load": const_loads[i:i+2]})

        self.instrs.append({"flow": [("pause",)]})

        # Broadcasts + initial data loads
        bcast_ops = []
        bcast_ops.append(("vbroadcast", zero_vec, zero_const))
        bcast_ops.append(("vbroadcast", one_vec, one_const))
        bcast_ops.append(("vbroadcast", two_vec, two_const))
        bcast_ops.append(("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]))
        bcast_ops.append(("vbroadcast", fvp_vec, self.scratch["forest_values_p"]))
        for hi, info in enumerate(hash_stage_info):
            if info[0] == 'madd':
                _, val1, factor = info
                bcast_ops.append(("vbroadcast", hash_madd_factor_vecs[hi], self.const_map[factor]))
                bcast_ops.append(("vbroadcast", hash_madd_c1_vecs[hi], self.const_map[val1]))
            else:
                op1_s, c1_s, op2_s, op3_s, c3_s = hash_consts[hi]
                v1, v3 = hash_3op_vecs[hi]
                bcast_ops.append(("vbroadcast", v1, c1_s))
                bcast_ops.append(("vbroadcast", v3, c3_s))

        bcast_idx = 0
        bcast_chunk = bcast_ops[bcast_idx:bcast_idx+6]
        bcast_idx += 6
        self.instrs.append({
            "valu": bcast_chunk,
            "alu": [
                ("+", gm_idx, self.scratch["inp_indices_p"], self.const_map[0]),
                ("+", gm_val, self.scratch["inp_values_p"], self.const_map[0]),
            ]
        })
        for g in range(n_groups - 1):
            bundle = {
                "load": [("vload", group_idx[g], gm_idx), ("vload", group_val[g], gm_val)],
                "alu": [("+", gm_idx, gm_idx, vlen_const), ("+", gm_val, gm_val, vlen_const)],
            }
            if bcast_idx < len(bcast_ops):
                bundle["valu"] = bcast_ops[bcast_idx:bcast_idx+6]
                bcast_idx += 6
            self.instrs.append(bundle)
        bundle = {"load": [("vload", group_idx[n_groups-1], gm_idx), ("vload", group_val[n_groups-1], gm_val)]}
        if bcast_idx < len(bcast_ops):
            bundle["valu"] = bcast_ops[bcast_idx:bcast_idx+6]
            bcast_idx += 6
        self.instrs.append(bundle)

        # ===== MAIN LOOP + STORES IN ONE DAG =====
        ops = []
        prev_round_wrap = [None] * n_groups
        prev_round_last_hash = [None] * n_groups
        all_group_info = []

        # Round 0 optimization: all indices start at 0
        r0_load = Op("load", ("load", nv_scalar, self.scratch["forest_values_p"]),
                      deps=[], group_id=-1)
        ops.append(r0_load)

        for rnd_offset in range(rounds):
            group_info = []
            for g in range(n_groups):
                idx = group_idx[g]
                val = group_val[g]
                addr_tmp = group_addr[g]
                nv = group_nv[g]
                t1 = pool_t1[g % N_POOL]
                t2 = pool_t2[g % N_POOL]
                pt = t1

                if rnd_offset == 0:
                    # Round 0: all idx = 0, broadcast forest_values[0]
                    bcast_op = Op("valu", ("vbroadcast", nv, nv_scalar),
                                  deps=[r0_load], group_id=g)
                    ops.append(bcast_op)
                    xor_op = Op("valu", ("^", val, val, nv), deps=[bcast_op], group_id=g)
                    ops.append(xor_op)
                    mul_op = Op("valu", ("multiply_add", idx, idx, two_vec, one_vec),
                                deps=[], group_id=g)
                    ops.append(mul_op)

                else:
                    # Regular rounds: scatter load
                    addr_deps = []
                    if prev_round_wrap[g] is not None:
                        addr_deps.append(prev_round_wrap[g])
                    addr_op = Op("valu", ("+", addr_tmp, idx, fvp_vec),
                                 deps=addr_deps, group_id=g)
                    ops.append(addr_op)

                    load_ops = []
                    for lane in range(VLEN):
                        lop = Op("load", ("load_offset", nv, addr_tmp, lane),
                                 deps=[addr_op], group_id=g)
                        load_ops.append(lop)
                        ops.append(lop)

                    xor_deps = list(load_ops)
                    if prev_round_last_hash[g] is not None:
                        xor_deps.append(prev_round_last_hash[g])
                    xor_op = Op("valu", ("^", val, val, nv), deps=xor_deps, group_id=g)
                    ops.append(xor_op)

                    mul_op = Op("valu", ("multiply_add", idx, idx, two_vec, one_vec),
                                deps=[addr_op], group_id=g)
                    ops.append(mul_op)

                prev_hash = xor_op
                first_pool_writer = None
                first_pool_writer2 = None

                for hi, info in enumerate(hash_stage_info):
                    if info[0] == 'madd':
                        h_op = Op("valu", ("multiply_add", val, val,
                                           hash_madd_factor_vecs[hi], hash_madd_c1_vecs[hi]),
                                  deps=[prev_hash], group_id=g)
                        ops.append(h_op)
                        prev_hash = h_op
                    else:
                        op1_s, c1_s, op2_s, op3_s, c3_s = hash_consts[hi]
                        hv1, hv3 = hash_3op_vecs[hi]
                        h1 = Op("valu", (op1_s, t1, val, hv1), deps=[prev_hash], group_id=g)
                        h2 = Op("valu", (op3_s, t2, val, hv3), deps=[prev_hash], group_id=g)
                        h3 = Op("valu", (op2_s, val, t1, t2), deps=[h1, h2], group_id=g)
                        ops.extend([h1, h2, h3])
                        if first_pool_writer is None:
                            first_pool_writer = h1
                            first_pool_writer2 = h2
                        prev_hash = h3

                last_hash = prev_hash
                mod_op = Op("valu", ("&", pt, val, one_vec), deps=[last_hash], group_id=g)
                ops.append(mod_op)
                add_op = Op("valu", ("+", idx, idx, pt), deps=[mul_op, mod_op], group_id=g)
                ops.append(add_op)
                lt_op = Op("valu", ("<", pt, idx, n_nodes_vec), deps=[add_op], group_id=g)
                ops.append(lt_op)
                wrap_op = Op("valu", ("*", idx, idx, pt), deps=[lt_op], group_id=g)
                ops.append(wrap_op)

                prev_round_wrap[g] = wrap_op
                prev_round_last_hash[g] = last_hash

                group_info.append({
                    'first_pool': first_pool_writer,
                    'first_pool2': first_pool_writer2,
                    'last_pool_use': wrap_op,
                })
            all_group_info.append(group_info)

        # Pool deps
        pool_users = defaultdict(list)
        for rnd_offset in range(rounds):
            for g in range(n_groups):
                pool_users[g % N_POOL].append(all_group_info[rnd_offset][g])
        for pool_idx, users in pool_users.items():
            for i in range(1, len(users)):
                curr = users[i]
                prev = users[i-1]
                if curr['first_pool'] is not None and prev['last_pool_use'] is not None:
                    curr['first_pool'].deps.append(prev['last_pool_use'])
                    if curr['first_pool2'] is not None:
                        curr['first_pool2'].deps.append(prev['last_pool_use'])

        # Add store ops using sequential addressing via flow.add_imm
        # First: set up gm_idx and gm_val (ALU ops)
        # Use flow.add_imm for incrementing (doesn't need scratch constants)
        # But add_imm is flow engine, 1/cycle. With 32 groups, that's 32 cycles for addr increments.
        # Better: chain stores using ALU for address computation

        # Actually, let's use a simpler approach: chain store pairs
        # Each pair: alu(gm_idx += VLEN, gm_val += VLEN) -> store(vstore gm_idx, gm_val)
        # First pair uses a dedicated ALU op to set initial values

        # First ALU: set gm_idx = inp_indices_p, gm_val = inp_values_p
        init_store_op = Op("alu", ("+", gm_idx, self.scratch["inp_indices_p"], zero_const),
                           deps=[], group_id=-1)
        ops.append(init_store_op)
        init_store_op2 = Op("alu", ("+", gm_val, self.scratch["inp_values_p"], zero_const),
                            deps=[], group_id=-1)
        ops.append(init_store_op2)

        prev_store_pair = [init_store_op, init_store_op2]
        for g in range(n_groups):
            # Store idx: depends on wrap_op (data) and prev store pair (addr)
            s_idx = Op("store", ("vstore", gm_idx, group_idx[g]),
                       deps=[prev_round_wrap[g]] + prev_store_pair, group_id=g)
            # Store val: depends on last_hash (data) and prev store pair (addr)
            s_val = Op("store", ("vstore", gm_val, group_val[g]),
                       deps=[prev_round_last_hash[g]] + prev_store_pair, group_id=g)
            ops.append(s_idx)
            ops.append(s_val)

            if g < n_groups - 1:
                # Increment addresses for next group
                inc_idx = Op("alu", ("+", gm_idx, gm_idx, vlen_const),
                             deps=[s_idx, s_val], group_id=g)
                inc_val = Op("alu", ("+", gm_val, gm_val, vlen_const),
                             deps=[s_idx, s_val], group_id=g)
                ops.append(inc_idx)
                ops.append(inc_val)
                prev_store_pair = [inc_idx, inc_val]

        bundles = list_schedule(ops)
        self.instrs.extend(bundles)

        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734

def do_kernel_test(forest_height, rounds, batch_size, seed=123, trace=False, prints=False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem, kb.instrs, kb.debug_info(), n_cores=N_CORES,
        value_trace=value_trace, trace=trace,
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
