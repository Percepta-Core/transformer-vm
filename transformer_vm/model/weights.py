"""Analytical weight construction for the transformer.

Reads a schedule plan (plan.yaml or MILP-generated) and constructs weights
for a VanillaTransformer from the computation graph.

Each half-layer is a transform: input dims -> latent (gates + passthrough) -> output dims.
The persist/output phase is a linear projection. For each output slot:
  delta[slot] = persist_expr evaluated over gate/passthrough outputs
              - old_value (if slot was reused from a dead dim)
"""

import logging
import math
import os
from collections import defaultdict

import torch
import yaml

logger = logging.getLogger(__name__)

HARD_K = 1e10  # softmax temperature scaling to approximate hardmax (argmax) attention


def _dump_allocation(
    n_layers, std_layers, slot_of, head_map, internal_dims, alloc_path="allocation.yaml"
):
    """Write allocation.yaml showing dim-to-slot and head assignments per layer."""

    slot_to_dims = {}
    for d, s in slot_of.items():
        if d in internal_dims:
            continue
        slot_to_dims.setdefault(s, []).append(d.name)

    doc = {"slot_map": {s: sorted(names) for s, names in sorted(slot_to_dims.items())}}

    layers = []
    for li in range(n_layers):
        attn, persist1, ffn, persist2 = std_layers[li]
        layer_info = {"layer": li}

        heads = []
        if li in head_map:
            for h, info in sorted(head_map[li].items()):
                heads.append({"head": h, **info})
        layer_info["attention_heads"] = heads

        layer_info["persist1"] = [d.name for d in persist1]

        ffn_names = []
        for rg in ffn:
            ffn_names.append(rg.name)
        layer_info["ffn"] = ffn_names

        layer_info["persist2"] = [d.name for d in persist2]
        layers.append(layer_info)

    doc["layers"] = layers

    with open(alloc_path, "w") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False, width=120)
    logger.info("Wrote allocation to %s", alloc_path)


def _load_plan(path, all_dims):
    from transformer_vm.graph.core import LookUpDimension

    with open(path) as f:
        plan = yaml.safe_load(f)

    name_to_dim = {d.name: d for d in all_dims}

    def resolve(name):
        d = name_to_dim.get(name)
        if d is None:
            raise KeyError(f"plan.yaml references unknown dim '{name}'")
        return d

    std_layers = []
    alive_after = {}

    for entry in plan["layers"]:
        L = entry["layer"]

        seen_hulls = {}
        for dname in entry["attention"]:
            d = resolve(dname)
            assert isinstance(d, LookUpDimension), f"{dname} not a LookUpDimension"
            lu = d.lookup
            if lu.id not in seen_hulls:
                seen_hulls[lu.id] = lu
        attn = list(seen_hulls.values())

        persist1 = [resolve(n) for n in entry["persist1"]]
        ffn = [resolve(n) for n in entry["ffn"]]
        persist2 = [resolve(n) for n in entry["persist2"]]
        std_layers.append((attn, persist1, ffn, persist2))

        if "after_persist1" in entry:
            alive_after[4 * L + 1] = frozenset(resolve(n) for n in entry["after_persist1"]["dims"])
        if "after_persist2" in entry:
            alive_after[4 * L + 3] = frozenset(resolve(n) for n in entry["after_persist2"]["dims"])

    return plan["summary"]["layers"], std_layers, alive_after


def build_model(
    use_erase=True,
    _shared=None,
    program_graph=None,
    plan_path=None,
    max_layers=None,
    no_reuse=False,
    max_ffn=None,
):
    """Build transformer weights from a computation graph.

    Args:
        use_erase: whether to use erase-based slot reuse.
        program_graph: optional ProgramGraph. If provided, uses its graph
            data instead of importing machine.build().
        plan_path: path to a schedule plan YAML file.
        max_layers: maximum number of transformer layers.
        no_reuse: disable slot reuse in interval coloring.
        max_ffn: maximum FFN neurons per layer.
    """
    from transformer_vm.graph.core import (
        Expression,
        InputDimension,
        LookUpDimension,
        PersistDimension,
        ReGLUDimension,
        _all_dims,
    )
    from transformer_vm.model.transformer import VanillaTransformer as TinyTransformerLM

    if _shared is not None:
        (
            ALL_DIMS,
            input_tokens,
            output_tokens,
            n_layers,
            std_layers,
            alive_after,
            slot_of,
            reused_at,
            erased_at,
            D,
            internal_reglus,
            internal_lookups,
            internal_persists,
            internal_dims,
            one_expr,
            pos_expr,
            erase_q2d,
            erase_k2d,
            expr_to_tensor,
            max_heads,
            n_heads,
            d_ffn,
            max_ffn,
            all_tokens,
            tok_to_idx_map,
            vocab_size,
            input_dims,
        ) = _shared
    else:
        if program_graph is not None:
            pg = program_graph
            input_tokens = pg.input_tokens
            output_tokens = pg.output_tokens
            ALL_DIMS = pg.all_dims
            _one = pg.one
            _position = pg.position
            _inv_log_pos = pg.inv_log_pos
            _position_sq = pg.position_sq
        else:
            from transformer_vm.graph.core import inv_log_pos, one, position, position_sq
            from transformer_vm.wasm.interpreter import build as build_graph

            input_tokens, output_tokens = build_graph()
            ALL_DIMS = list(_all_dims)
            _one = one
            _position = position
            _inv_log_pos = inv_log_pos
            _position_sq = position_sq

        # ── Load schedule ─────────────────────────────────────────
        if plan_path:
            n_layers, std_layers, alive_after = _load_plan(plan_path, ALL_DIMS)
        else:
            from transformer_vm.scheduler.milp import milp_schedule

            sched = milp_schedule(
                input_tokens,
                output_tokens,
                max_layers=max_layers,
                max_ffn=max_ffn,
                program_graph=program_graph,
            )
            std_layers = sched["std_layers"]
            n_layers = sched["num_layers"]
            alive_after = sched.get("alive_after", {})

        # ── Slot assignment (interval coloring) ───────────────────
        input_dims = [d for d in ALL_DIMS if isinstance(d, InputDimension)]
        FIXED = {_position: 0, _inv_log_pos: 1, _position_sq: 2}
        protected_slots = set(FIXED.values())
        slot_of = dict(FIXED)
        next_slot = 3
        for d in input_dims:
            if d not in slot_of:
                slot_of[d] = next_slot
                next_slot += 1

        free_slots = []
        pending_free = []  # freed this half-layer, available next
        reused_at = {}  # (L, sp) -> slots written (already 0)
        erased_at = {}  # (L, sp) -> slots being erased to 0

        cur = frozenset(input_dims)
        for L in range(n_layers):
            for sp in (1, 3):
                c = 4 * L + sp
                nxt = alive_after.get(c, cur)
                dying = cur - nxt
                born = nxt - cur

                if not no_reuse:
                    free_slots.extend(pending_free)
                    free_slots.sort()
                    pending_free = []

                erased = set()
                if not no_reuse:
                    for d in dying:
                        s = slot_of[d]
                        if s in protected_slots:
                            continue
                        pending_free.append(s)
                        erased.add(s)

                reused = set()
                for d in sorted(born, key=lambda x: x.id):
                    if free_slots:
                        s = free_slots.pop(0)
                        slot_of[d] = s
                        reused.add(s)
                    else:
                        slot_of[d] = next_slot
                        next_slot += 1

                reused_at[(L, sp)] = reused
                erased_at[(L, sp)] = erased
                cur = nxt

        # Assign slots to output-referenced dims not yet covered
        output_dims_need = set()
        for expr in output_tokens.values():
            if isinstance(expr, Expression):
                for d in expr.terms:
                    if d not in slot_of and d not in input_dims:
                        output_dims_need.add(d)
        for d in sorted(output_dims_need, key=lambda x: x.id):
            if free_slots:
                s = free_slots.pop(0)
                slot_of[d] = s
            else:
                slot_of[d] = next_slot
                next_slot += 1

        D = next_slot
        D += D % 2

        # ── Internal dims (never in any alive set) ────────────────
        all_ever_alive = set()
        for s in alive_after.values():
            all_ever_alive |= s
        output_dims_set = set()
        for expr in output_tokens.values():
            if isinstance(expr, Expression):
                output_dims_set |= set(expr.terms.keys())
        non_internal = all_ever_alive | set(input_dims) | output_dims_set
        internal_reglus = {
            d for d in ALL_DIMS if isinstance(d, ReGLUDimension) and d not in non_internal
        }
        internal_lookups = {
            d for d in ALL_DIMS if isinstance(d, LookUpDimension) and d not in non_internal
        }
        internal_persists = {
            d for d in ALL_DIMS if isinstance(d, PersistDimension) and d not in non_internal
        }
        internal_dims = internal_reglus | internal_lookups | internal_persists

        # ── Expressions for passthrough/erase Q/K ─────────────────
        one_expr = Expression({_one: 1})
        pos_expr = Expression({_position: 1})
        erase_q2d = [pos_expr, one_expr]
        erase_k2d = [pos_expr * 2, one_expr]

    def expr_to_tensor(expr):
        w = torch.zeros(D, dtype=torch.float64)
        for dim, coeff in expr.terms.items():
            if dim in internal_dims:
                continue
            if dim in slot_of:
                w[slot_of[dim]] += coeff
        return w

    if _shared is None:
        # ── Count resources (n_heads, d_ffn) ──────────────────────
        max_heads = 0
        max_ffn = 1

        cur = frozenset(input_dims)
        for L in range(n_layers):
            attn, persist1, ffn, persist2 = std_layers[L]

            c1 = 4 * L + 1
            nxt1 = alive_after.get(c1, cur)
            erased1 = erased_at[(L, 1)]

            lookup_dims_set = set()
            for lu in attn:
                lookup_dims_set |= set(lu.dims)
            n_lu_heads = sum((len(lu.value_exprs) + 1) // 2 for lu in attn)

            pt_srcs = set(erased1)
            for pd in persist1:
                for d in pd.expr.terms:
                    if d not in slot_of or d in internal_dims:
                        continue
                    if not (isinstance(d, LookUpDimension) and d in lookup_dims_set):
                        pt_srcs.add(slot_of[d])

            max_heads = max(max_heads, n_lu_heads + (len(pt_srcs) + 1) // 2)
            cur = nxt1

            c3 = 4 * L + 3
            nxt3 = alive_after.get(c3, cur)
            erased3 = erased_at[(L, 3)]

            same_rg = set(ffn)
            pt_srcs_ffn = set(erased3)
            for pd in persist2:
                for d in pd.expr.terms:
                    if d not in slot_of or d in internal_dims:
                        continue
                    if d not in same_rg:
                        pt_srcs_ffn.add(slot_of[d])

            max_ffn = max(max_ffn, len(ffn) + len(pt_srcs_ffn))
            cur = nxt3

        if 2 * max_heads > D:
            D = 2 * max_heads
            D += D % 2
        n_heads = D // 2
        d_ffn = max_ffn

        # ── Vocabulary ────────────────────────────────────────────
        all_tokens = sorted(set(input_tokens.keys()) | set(output_tokens.keys()))
        tok_to_idx_map = {t: i for i, t in enumerate(all_tokens)}
        vocab_size = len(all_tokens)

    # ── Create model ──────────────────────────────────────────────
    model = TinyTransformerLM(
        vocab=vocab_size,
        d_model=D,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        stop_token_id=tok_to_idx_map.get("halt", 0),
    )

    # ── Populate weights ──────────────────────────────────────────
    sqrt_dh = math.sqrt(2.0)
    n_hulls = n_pt_h = n_pt_g = 0
    head_map = {}  # layer_idx -> {head_idx -> info}

    with torch.no_grad():
        # Embedding
        model.tok.weight.zero_()
        for tok_name, expr in input_tokens.items():
            idx = tok_to_idx_map[tok_name]
            model.tok.weight[idx] = expr_to_tensor(expr)
            model.tok.weight[idx, 0] = 0.0
            model.tok.weight[idx, 1] = 0.0
            model.tok.weight[idx, 2] = 0.0

        # Output head
        model.head.weight.zero_()
        for tok_name, expr in output_tokens.items():
            idx = tok_to_idx_map[tok_name]
            model.head.weight[idx] = expr_to_tensor(expr)

        # ── Per-layer weights ─────────────────────────────────────
        cur = frozenset(input_dims)
        all_tiebreak = []

        for layer_idx in range(n_layers):
            attn, persist1, ffn, persist2 = std_layers[layer_idx]

            ip = model.attn[layer_idx].in_proj_weight.data
            op_w = model.attn[layer_idx].out_proj.weight.data
            ip.zero_()
            op_w.zero_()

            fi = model.ff_in[layer_idx].weight.data
            fo = model.ff_out[layer_idx].weight.data
            fi.zero_()
            fo.zero_()

            # ── ATTENTION HALF-LAYER ──────────────────────────────
            c1 = 4 * layer_idx + 1
            nxt1 = alive_after.get(c1, cur)
            head_idx = 0
            lookup_dim_to_head = {}
            layer_tiebreak = []

            # 1) LookUp heads
            layer_heads = {}
            for lu in attn:
                nv = len(lu.value_exprs)
                for p in range((nv + 1) // 2):
                    h = head_idx
                    head_idx += 1
                    n_hulls += 1
                    layer_tiebreak.append(1 if lu.tie_break == "latest" else 0)
                    ip[h * 2] = expr_to_tensor(lu.query_exprs_2d[0]) * HARD_K * sqrt_dh
                    ip[h * 2 + 1] = expr_to_tensor(lu.query_exprs_2d[1]) * HARD_K * sqrt_dh
                    ip[D + h * 2] = expr_to_tensor(lu.key_exprs_2d[0])
                    ip[D + h * 2 + 1] = expr_to_tensor(lu.key_exprs_2d[1])
                    ip[2 * D + h * 2] = expr_to_tensor(lu.value_exprs[p * 2])
                    if p * 2 + 1 < nv:
                        ip[2 * D + h * 2 + 1] = expr_to_tensor(lu.value_exprs[p * 2 + 1])

                    d0 = lu.dims[p * 2]
                    lookup_dim_to_head[d0] = (h, 0)
                    if d0 not in internal_lookups:
                        op_w[slot_of[d0], h * 2] = 1.0
                    if p * 2 + 1 < nv:
                        d1 = lu.dims[p * 2 + 1]
                        lookup_dim_to_head[d1] = (h, 1)
                        if d1 not in internal_lookups:
                            op_w[slot_of[d1], h * 2 + 1] = 1.0

                    vals = [d0.name]
                    if p * 2 + 1 < nv:
                        vals.append(lu.dims[p * 2 + 1].name)
                    q_terms = {d.name: c for d, c in lu.query_exprs_2d[0].terms.items()}
                    q_terms.update(
                        {"_qy_" + d.name: c for d, c in lu.query_exprs_2d[1].terms.items()}
                    )
                    k_terms = {d.name: c for d, c in lu.key_exprs_2d[0].terms.items()}
                    k_terms.update(
                        {"_ky_" + d.name: c for d, c in lu.key_exprs_2d[1].terms.items()}
                    )
                    layer_heads[h] = {
                        "type": "lookup",
                        "lookup": lu.name or f"lookup_{lu.id}",
                        "values": vals,
                        "tie_break": lu.tie_break,
                    }
            head_map[layer_idx] = layer_heads

            # 2) Passthrough contributions: src_slot -> {dst_slot: coeff}
            #    Persist terms referencing non-lookup dims need passthrough.
            pt = defaultdict(lambda: defaultdict(float))

            for pd in persist1:
                if pd not in slot_of:
                    continue
                for d, c in pd.expr.terms.items():
                    if d in lookup_dim_to_head:
                        h, comp = lookup_dim_to_head[d]
                        op_w[slot_of[pd], h * 2 + comp] += c
                    elif d in slot_of:
                        pt[slot_of[d]][slot_of[pd]] += c

            if use_erase:
                for s in erased_at[(layer_idx, 1)]:
                    pt[s][s] -= 1.0

            # 3) Pack passthroughs 2 per head (each reads one src_slot)
            pt_items = list(pt.items())
            for pair_idx in range(0, len(pt_items), 2):
                h = head_idx
                head_idx += 1
                ip[h * 2] = expr_to_tensor(erase_q2d[0]) * HARD_K * sqrt_dh
                ip[h * 2 + 1] = expr_to_tensor(erase_q2d[1]) * HARD_K * sqrt_dh
                ip[D + h * 2] = expr_to_tensor(erase_k2d[0])
                ip[D + h * 2 + 1] = expr_to_tensor(erase_k2d[1])

                src1, dsts1 = pt_items[pair_idx]
                ip[2 * D + h * 2, src1] = 1.0
                for dst, coeff in dsts1.items():
                    op_w[dst, h * 2] += coeff

                pt_info = {"type": "passthrough", "slot_v0": src1}
                if pair_idx + 1 < len(pt_items):
                    src2, dsts2 = pt_items[pair_idx + 1]
                    ip[2 * D + h * 2 + 1, src2] = 1.0
                    for dst, coeff in dsts2.items():
                        op_w[dst, h * 2 + 1] += coeff
                    pt_info["slot_v1"] = src2
                layer_heads[h] = pt_info

            n_pt_h += len(pt_items)
            while len(layer_tiebreak) < head_idx:
                layer_tiebreak.append(0)
            assert head_idx <= n_heads, f"L{layer_idx} attn: {head_idx} heads > {n_heads}"
            while len(layer_tiebreak) < n_heads:
                layer_tiebreak.append(0)
            all_tiebreak.append(layer_tiebreak)

            cur = nxt1

            # ── FFN HALF-LAYER ────────────────────────────────────
            c3 = 4 * layer_idx + 3
            nxt3 = alive_after.get(c3, cur)
            reglu_to_gate = {}
            j = 0

            # 1) ReGLU neurons
            for rg in ffn:
                fi[j] = expr_to_tensor(rg.b_expr)
                fi[d_ffn + j] = expr_to_tensor(rg.a_expr)
                reglu_to_gate[rg] = j
                if rg not in internal_reglus:
                    fo[slot_of[rg], j] = 1.0
                j += 1

            # 2) Passthrough contributions: src_slot -> {dst_slot: coeff}
            pt_ffn = defaultdict(lambda: defaultdict(float))

            for pd in persist2:
                if pd not in slot_of:
                    continue
                for d, c in pd.expr.terms.items():
                    if d in reglu_to_gate:
                        fo[slot_of[pd], reglu_to_gate[d]] += c
                    elif d in slot_of:
                        pt_ffn[slot_of[d]][slot_of[pd]] += c

            if use_erase:
                for s in erased_at[(layer_idx, 3)]:
                    pt_ffn[s][s] -= 1.0

            # 3) One neuron per source slot
            for src, dsts in pt_ffn.items():
                fi[j] = expr_to_tensor(one_expr)
                fi[d_ffn + j, src] = 1.0
                for dst, coeff in dsts.items():
                    fo[dst, j] += coeff
                j += 1
                n_pt_g += 1

            assert j <= d_ffn, f"L{layer_idx} ffn: {j} neurons > {d_ffn}"

            cur = nxt3

    model.attn_erase = [sorted(erased_at[(li, 1)]) for li in range(n_layers)]
    model.ffn_erase = [sorted(erased_at[(li, 3)]) for li in range(n_layers)]
    model.head_tiebreak = all_tiebreak

    # Head type: 0=lookup (hull), 1=passthrough (V[t]), 2=gather (V[round(q)])
    from transformer_vm.graph.core import InputDimension as _ID
    from transformer_vm.graph.core import _all_lookups
    all_head_type = []
    for li in range(n_layers):
        layer_ht = [0] * n_heads
        if li in head_map:
            for h, info in head_map[li].items():
                if info.get("type") == "passthrough":
                    layer_ht[h] = 1
                elif info.get("type") == "lookup":
                    # Check if the lookup keys on position (1D key = position)
                    lu_name = info.get("lookup", "")
                    # Find the LookUp object by matching name
                    for lu in pg.all_lookups if program_graph else _all_lookups:
                        if (lu.name or f"lookup_{lu.id}") == lu_name:
                            kx = lu.key_exprs_2d[0]
                            # Position-keyed if kx = 2*position (one term, the position dim)
                            if len(kx.terms) == 1:
                                dim, coeff = next(iter(kx.terms.items()))
                                if isinstance(dim, _ID) and dim.name == "position" and coeff == 2:
                                    layer_ht[h] = 2
                            break
        all_head_type.append(layer_ht)
    model.head_type = all_head_type
    n_pt = sum(v == 1 for layer in all_head_type for v in layer)
    n_ga = sum(v == 2 for layer in all_head_type for v in layer)
    n_hu = sum(v == 0 for layer in all_head_type for v in layer)
    logger.info("  head_type: %d passthrough, %d gather, %d hull", n_pt, n_ga, n_hu)

    n_persist = sum(len(p1) + len(p2) for _, p1, _, p2 in std_layers)
    logger.info(
        "Built model: d_model=%d, n_layers=%d, n_heads=%d, d_ffn=%d, erase=%s",
        D,
        n_layers,
        n_heads,
        d_ffn,
        use_erase,
    )
    logger.info(
        "  vocab=%d, hulls=%d, persist=%d, pt_h=%d, pt_g=%d",
        vocab_size,
        n_hulls,
        n_persist,
        n_pt_h,
        n_pt_g,
    )

    _dump_allocation(n_layers, std_layers, slot_of, head_map, internal_dims)

    shared = (
        ALL_DIMS,
        input_tokens,
        output_tokens,
        n_layers,
        std_layers,
        alive_after,
        slot_of,
        reused_at,
        erased_at,
        D,
        internal_reglus,
        internal_lookups,
        internal_persists,
        internal_dims,
        one_expr,
        pos_expr,
        erase_q2d,
        erase_k2d,
        expr_to_tensor,
        max_heads,
        n_heads,
        d_ffn,
        max_ffn,
        all_tokens,
        tok_to_idx_map,
        vocab_size,
        input_dims,
    )

    return model, all_tokens, tok_to_idx_map, shared


def build_model_pair(
    program_graph=None, plan_path=None, max_layers=None, no_reuse=False, max_ffn=None
):
    """Build both erase and mask models from the same graph/plan."""
    model_e, tok, t2i, shared = build_model(
        use_erase=True,
        program_graph=program_graph,
        plan_path=plan_path,
        max_layers=max_layers,
        no_reuse=no_reuse,
        max_ffn=max_ffn,
    )
    model_m, _, _, _ = build_model(use_erase=False, _shared=shared)
    return model_e, model_m, tok, t2i


def flops_per_token(model):
    D = model.tok.weight.shape[1]
    n_layers = len(model.attn)
    d_ffn = model.ff_in[0].weight.shape[0] // 2
    vocab = model.head.weight.shape[0]
    per_layer = 4 * (2 * D * D) + 2 * (2 * d_ffn * D) + 2 * (D * d_ffn)
    return n_layers * per_layer + 2 * vocab * D


def save_weights(model, all_tokens, path):
    """Save model weights as a flat binary file for the C++ inference engine."""
    import struct

    n_layers = len(model.attn)
    with open(path, "wb") as f:
        f.write(
            struct.pack(
                "<6i",
                len(all_tokens),
                model.tok.weight.shape[1],
                n_layers,
                model.attn[0].num_heads,
                model.ff_in[0].weight.shape[0] // 2,
                model.stop_token_id,
            )
        )
        for t in all_tokens:
            b = t.encode()
            f.write(struct.pack("<I", len(b)))
            f.write(b)

        def W(t):
            f.write(t.detach().contiguous().cpu().numpy().tobytes())

        W(model.tok.weight)
        for li in range(n_layers):
            W(model.attn[li].in_proj_weight)
            W(model.attn[li].out_proj.weight)
            W(model.ff_in[li].weight)
            W(model.ff_out[li].weight)
        W(model.head.weight)

        has_erase = hasattr(model, "attn_erase")
        f.write(struct.pack("<i", 1 if has_erase else 0))
        if has_erase:
            for li in range(n_layers):
                ae = model.attn_erase[li]
                f.write(struct.pack("<i", len(ae)))
                for s in ae:
                    f.write(struct.pack("<i", s))
                fe = model.ffn_erase[li]
                f.write(struct.pack("<i", len(fe)))
                for s in fe:
                    f.write(struct.pack("<i", s))

        has_tiebreak = hasattr(model, "head_tiebreak")
        f.write(struct.pack("<i", 1 if has_tiebreak else 0))
        if has_tiebreak:
            H = model.attn[0].num_heads
            for li in range(n_layers):
                for h in range(H):
                    f.write(struct.pack("<i", model.head_tiebreak[li][h]))

        has_head_type = hasattr(model, "head_type")
        f.write(struct.pack("<i", 1 if has_head_type else 0))
        if has_head_type:
            H = model.attn[0].num_heads
            for li in range(n_layers):
                for h in range(H):
                    f.write(struct.pack("<i", model.head_type[li][h]))

    logger.info("Saved weights to %s (%s bytes)", path, f"{os.path.getsize(path):,}")


def load_weights(path):
    """Load model weights from a binary file produced by save_weights().

    Returns (model, all_tokens, tok_to_idx_map).
    """
    import struct

    import numpy as np

    from transformer_vm.model.transformer import VanillaTransformer as TinyTransformerLM

    with open(path, "rb") as f:
        vocab, d_model, n_layers, n_heads, d_ffn, stop_token_id = struct.unpack("<6i", f.read(24))

        all_tokens = []
        for _ in range(vocab):
            slen = struct.unpack("<I", f.read(4))[0]
            all_tokens.append(f.read(slen).decode())
        tok_to_idx_map = {t: i for i, t in enumerate(all_tokens)}

        model = TinyTransformerLM(
            vocab=vocab,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ffn=d_ffn,
            stop_token_id=stop_token_id,
        )

        def R(shape):
            n = 1
            for s in shape:
                n *= s
            data = np.frombuffer(f.read(n * 8), dtype=np.float64)
            return torch.from_numpy(data.copy()).reshape(shape)

        with torch.no_grad():
            model.tok.weight.copy_(R((vocab, d_model)))
            for li in range(n_layers):
                model.attn[li].in_proj_weight.copy_(R((3 * d_model, d_model)))
                model.attn[li].out_proj.weight.copy_(R((d_model, d_model)))
                model.ff_in[li].weight.copy_(R((2 * d_ffn, d_model)))
                model.ff_out[li].weight.copy_(R((d_model, d_ffn)))
            model.head.weight.copy_(R((vocab, d_model)))

        has_erase = struct.unpack("<i", f.read(4))[0]
        if has_erase:
            model.attn_erase = []
            model.ffn_erase = []
            for _ in range(n_layers):
                ae_len = struct.unpack("<i", f.read(4))[0]
                ae = [struct.unpack("<i", f.read(4))[0] for _ in range(ae_len)]
                model.attn_erase.append(ae)
                fe_len = struct.unpack("<i", f.read(4))[0]
                fe = [struct.unpack("<i", f.read(4))[0] for _ in range(fe_len)]
                model.ffn_erase.append(fe)

        has_tiebreak = struct.unpack("<i", f.read(4))[0]
        if has_tiebreak:
            model.head_tiebreak = []
            for _ in range(n_layers):
                layer_tb = [struct.unpack("<i", f.read(4))[0] for _ in range(n_heads)]
                model.head_tiebreak.append(layer_tb)

        # Head type: 1 = passthrough, 0 = lookup
        remaining = f.read(4)
        if remaining and len(remaining) == 4:
            has_head_type = struct.unpack("<i", remaining)[0]
            if has_head_type:
                model.head_type = []
                for _ in range(n_layers):
                    layer_ht = [struct.unpack("<i", f.read(4))[0] for _ in range(n_heads)]
                    model.head_type.append(layer_ht)

    logger.info(
        "Loaded weights from %s (vocab=%d, d_model=%d, n_layers=%d, n_heads=%d, d_ffn=%d)",
        path,
        vocab,
        d_model,
        n_layers,
        n_heads,
        d_ffn,
    )
    return model, all_tokens, tok_to_idx_map
