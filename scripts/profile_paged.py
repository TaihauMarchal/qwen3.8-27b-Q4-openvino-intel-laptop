#!/usr/bin/env python3
"""Decode-only per-op GPU profile of the Qwen3.8 language model in the *paged* (continuous batching) form.

The model gets the same graph rewrite GenAI CB applies (SDPAToPagedAttention -> PagedAttentionExtension,
PagedCausalConv1D, PagedGatedDeltaNet); KV cache and linear-attention state tables are remote tensors owned by this
script (as GenAI's cache managers do), so two infer requests can share them:
  request A : prefill only (chunks of --chunk tokens, non-speculative linear-attention plan)
  request B : decode only  (K+1 tokens per pass, speculative verifier plan: committed row + K+1 scratch rows)
GPU PERF_COUNT is a running average per network (= per infer request), so request B's profile is a pure decode
profile at the requested context length.

usage: profile_paged.py MODEL_DIR --ctx 8640 [--k 3] [--steps 24] [--chunk 256] [--json OUT]
"""
import argparse, collections, json, time
import numpy as np
import openvino as ov
from openvino._offline_transformations import paged_attention_transformation


def add_gather_before_lm_head(m):
    """Same idea as GenAI's apply_gather_before_matmul_transformation: logits only for sampled token rows."""
    import openvino.opset13 as ops
    res = [r for r in m.get_results() if "logits" in r.output(0).get_names() or "logits" in r.input(0).get_source_output().get_names()]
    if not res:
        return False
    node = res[0].input(0).get_source_output().get_node()
    for _ in range(6):
        if node.get_type_name() == "MatMul":
            break
        node = node.input(0).get_source_output().get_node()
    if node.get_type_name() != "MatMul":
        return False
    src = node.input(0).get_source_output()
    idx = ops.parameter([-1], ov.Type.i64, name="sampled_tokens_indices")
    idx.output(0).set_names({"sampled_tokens_indices"})
    g = ops.gather(src, idx, ops.constant(0, ov.Type.i64))
    node.input(0).replace_source_output(g.output(0))
    m.add_parameters([idx])
    m.validate_nodes_and_infer_types()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, default=8640)
    ap.add_argument("--k", type=int, default=3, help="draft tokens per pass (decode pass = K+1 tokens); 0 = plain 1-token decode")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--json")
    ap.add_argument("--sweep-k", default="", help="after the main run: comma list of extra K values (K+1 tokens/pass, 0 = plain decode), wall medians only")
    ap.add_argument("--sweep-steps", type=int, default=12)
    ap.add_argument("--dump-logits", help="save logits of every decode pass (npz) to compare two plugins on identical inputs")
    ap.add_argument("--props", default="{}")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--nodes", default="PagedAttentionExtension", help="node types printed individually (first 3)")
    a = ap.parse_args()

    core = ov.Core()
    print("openvino", ov.__file__, ov.get_version(), flush=True)
    emb = core.compile_model(f"{a.model}/openvino_text_embeddings_model.xml", "CPU")
    t0 = time.time()
    m = core.read_model(f"{a.model}/openvino_language_model.xml")
    paged_attention_transformation(m, False, False, True)
    gather_ok = add_gather_before_lm_head(m)
    print("gather before lm_head:", gather_ok, flush=True)
    t_read = time.time() - t0
    props = {"PERF_COUNT": True}
    props.update(json.loads(a.props))
    t0 = time.time()
    cm = core.compile_model(m, "GPU", props)
    t_compile = time.time() - t0
    print(f"read+transform {t_read:.1f}s compile {t_compile:.1f}s", flush=True)
    ctx = core.get_default_context("GPU")

    block = 16
    sweep = [int(x) for x in a.sweep_k.split(",") if x != ""]
    max_tokens = a.ctx + (a.k + 1) * (a.steps + 4) + 64 + sum((k + 1) * (a.sweep_steps + 2) for k in sweep)
    n_kv_blocks = (max_tokens + block - 1) // block
    n_la_rows = max([a.k] + sweep) + 2
    remote = {}
    kv_bytes = la_bytes = 0
    for inp in cm.inputs:
        name = inp.get_any_name()
        ps = inp.get_partial_shape()
        et = inp.get_element_type()
        if name.startswith("key_cache.") or name.startswith("value_cache."):
            shape = [n_kv_blocks] + [ps[i].get_length() for i in range(1, 4)]
            remote[name] = ctx.create_tensor(et, ov.Shape(shape), {})
            kv_bytes += int(np.prod(shape)) * et.size
            if name in ("key_cache.0", "value_cache.0"):
                print(" ", name, shape, et, flush=True)
        elif "_state_table." in name:
            shape = [n_la_rows] + [ps[i].get_length() for i in range(1, ps.rank.get_length())]
            t = ctx.create_tensor(et, ov.Shape(shape), {})
            z = ov.Tensor(et, ov.Shape(shape))
            np.asarray(z.data).fill(0)
            t.copy_from(z)
            remote[name] = t
            la_bytes += int(np.prod(shape)) * et.size
            if name.endswith(".0"):
                print(" ", name, shape, et, flush=True)
    print(f"KV cache {kv_bytes/2**20:.0f} MiB ({n_kv_blocks} blocks), LA state {la_bytes/2**20:.0f} MiB ({n_la_rows} rows)", flush=True)

    rng = np.random.default_rng(0)
    ids = rng.integers(1000, 50000, size=(1, max_tokens)).astype(np.int64)
    in_names = {i.get_any_name() for i in cm.inputs}

    def mk_request():
        r = cm.create_infer_request()
        for n, t in remote.items():
            r.set_tensor(n, t)
        return r

    def run(req, past, T, la_blocks, la_interval, sampled=None):
        e = emb({"input": ids[:, past:past + T]})[0].reshape(T, -1).astype(np.float32)
        pos = np.arange(past, past + T, dtype=np.int64)[None, :].repeat(4, 0)
        nb = (past + T + block - 1) // block
        feed = {"inputs_embeds": e, "position_ids": pos,
                "max_context_len": np.array(past + T, np.int32),
                "past_lens": np.array([past], np.int32),
                "subsequence_begins": np.array([0, T], np.int32),
                "block_indices_begins": np.array([0, nb], np.int32),
                "block_indices": np.arange(nb, dtype=np.int32),
                "score_aggregation_window": np.array([0], np.int32),
                "la.block_indices": np.array(la_blocks, np.int32),
                "la.block_indices_begins": np.array([0, len(la_blocks)], np.int32),
                "la.past_lens": np.array([past], np.int32),
                "la.cache_interval": np.array([la_interval], np.int32),
                "sampled_tokens_indices": np.array(sampled if sampled is not None else [T - 1], np.int64)}
        for n, v in feed.items():
            if n in in_names:
                req.set_tensor(n, ov.Tensor(v))
        t0 = time.perf_counter()
        req.infer()
        dt = time.perf_counter() - t0
        return dt

    # ---- prefill on request A (never used for the decode profile)
    reqA = mk_request()
    t0 = time.time()
    past = 0
    pre = []
    while past < a.ctx:
        T = min(a.chunk, a.ctx - past)
        pre.append(run(reqA, past, T, [0, 0], 0))
        past += T
    print(f"prefill {a.ctx} tok in chunks of {a.chunk}: {time.time()-t0:.2f}s, last chunks ms {[round(x*1000) for x in pre[-3:]]}", flush=True)
    snap_prefill = {p.node_name: p.real_time.total_seconds() * 1e6 for p in reqA.profiling_info}
    pre_tot = sum(snap_prefill.values())
    print(f"prefill mean per-chunk GPU kernel time: {pre_tot/1000:.1f} ms over {len(pre)} chunks", flush=True)
    pre_by = collections.Counter()
    for p in reqA.profiling_info:
        pre_by[(p.node_type, p.exec_type)] += p.real_time.total_seconds() * 1e3
    for (k, v) in pre_by.most_common(8):
        print(f"  prefill {v:8.2f} ms/chunk  {k[0]} | {k[1]}", flush=True)

    # ---- decode on request B
    reqB = reqA
    times = []
    dumped = []
    committed = 0
    for s in range(a.steps):
        if a.k > 0:
            scratch = [r for r in range(n_la_rows) if r != committed][:a.k + 1]
            dt = run(reqB, past, a.k + 1, [committed] + scratch, 1, list(range(a.k + 1)))
            acc = 2 + (s % 2)                      # ~2.5 accepted tokens per pass
            committed = scratch[acc - 1]           # promote the winning row
            past += acc
        else:
            dt = run(reqB, past, 1, [0, 0], 0)
            past += 1
        times.append(dt)
        if a.dump_logits:
            dumped.append(np.array(reqB.get_tensor("logits").data, dtype=np.float32).reshape(-1, reqB.get_tensor("logits").data.shape[-1]))
    if a.dump_logits:
        np.savez_compressed(a.dump_logits, **{f"pass{i}": x for i, x in enumerate(dumped)})
    ms = [round(t * 1000, 1) for t in times]
    med = float(np.median(times[2:])) * 1000
    print(f"decode ctx~{a.ctx} K={a.k}: ms/pass {ms}", flush=True)
    print(f"median (skip 2) {med:.1f} ms", flush=True)

    # GPU PERF_COUNT is an arithmetic mean over every infer() of the (shared) network since compile:
    # decode-only mean = (mean_after_decode * (N1+N2) - mean_after_prefill * N1) / N2
    N1, N2 = len(pre), len(times)
    by_exec = collections.Counter(); cnt = collections.Counter(); tot = 0.0
    nodes = []
    for p in reqB.profiling_info:
        us2 = p.real_time.total_seconds() * 1e6
        us1 = snap_prefill.get(p.node_name, 0.0)
        us = (us2 * (N1 + N2) - us1 * N1) / N2
        if us2 <= 0:
            continue
        tot += us
        by_exec[(p.node_type, p.exec_type)] += us
        cnt[(p.node_type, p.exec_type)] += 1
        nodes.append((p.node_type, p.exec_type, p.node_name, us))
    print(f"decode-only sum of per-primitive GPU kernel time: {tot/1000:.2f} ms")
    for (k, v) in by_exec.most_common(a.top):
        print(f"  {v/1000:8.2f} ms {100*v/tot:5.1f}%  n={cnt[k]:4d}  {k[0]} | {k[1]}")
    show = set(x for x in a.nodes.split(",") if x)
    seen = collections.Counter()
    for t, e, n, us in nodes:
        if t in show and seen[t] < 3:
            seen[t] += 1
            print(f"    node {us:9.1f} us  {t} {e} {n[-80:]}")
    sweep_res = {}
    for k in sweep:
        ts = []
        for st in range(a.sweep_steps):
            if k > 0:
                scratch = [r for r in range(n_la_rows) if r != committed][:k + 1]
                ts.append(run(reqB, past, k + 1, [committed] + scratch, 1, list(range(k + 1))))
                acc = min(k, 2 + (st % 2))
                committed = scratch[acc - 1]
                past += acc
            else:
                ts.append(run(reqB, past, 1, [committed, committed], 0))
                past += 1
        sweep_res[k] = float(np.median(ts[2:])) * 1000
        print(f"sweep K={k} ({k+1} tokens/pass) ctx~{past}: median {sweep_res[k]:.1f} ms/pass  {[round(t*1000) for t in ts]}", flush=True)
    if a.json:
        json.dump({"ov": ov.get_version(), "ctx": a.ctx, "k": a.k, "read_s": t_read, "compile_s": t_compile,
                   "prefill_ms": [round(x * 1000, 1) for x in pre], "decode_ms": ms, "median_ms": med, "sweep": sweep_res,
                   "prefill_by_exec": [[k[0], k[1], v] for k, v in pre_by.most_common()],
                   "by_exec": [[k[0], k[1], cnt[k], v / 1000] for k, v in by_exec.most_common()],
                   "nodes": [[t, e, n, us] for t, e, n, us in nodes]}, open(a.json, "w"), indent=0)


if __name__ == "__main__":
    main()
