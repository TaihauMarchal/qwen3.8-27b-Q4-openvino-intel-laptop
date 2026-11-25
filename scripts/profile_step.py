#!/usr/bin/env python3
"""Profile one prefill + N decode steps of a stateful Qwen3.5/3.8 OV language model on GPU.

Usage: python profile_step.py MODEL_DIR [--prompt-len 2048] [--steps 8] [--dump-runtime FILE]
Prints per-layer-type and per-exec-type totals for the LAST decode step (PERF_COUNT),
and wall time per decode step. The plugin used is whatever 'openvino' package is imported
(stock wheel, or a custom-built GPU plugin put first on PYTHONPATH / LD_LIBRARY_PATH).
"""
import argparse, collections, time, numpy as np, openvino as ov

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=8); ap.add_argument("--dump-runtime")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--show", default="", help="comma list of node types whose individual nodes are printed (layer 0/1 only)")
    a = ap.parse_args()
    core = ov.Core()
    print("openvino", ov.__file__, ov.get_version())
    emb = core.compile_model(f"{a.model}/openvino_text_embeddings_model.xml", "CPU")
    m = core.read_model(f"{a.model}/openvino_language_model.xml")
    t0 = time.time()
    cm = core.compile_model(m, "GPU", {"PERF_COUNT": "YES"})
    print(f"compile {time.time()-t0:.1f}s")
    if a.dump_runtime:
        rt = cm.get_runtime_model()
        with open(a.dump_runtime, "w") as f:
            for op in rt.get_ordered_ops():
                info = op.get_rt_info()
                g = lambda k: str(info[k].astype(str)) if k in info else ""
                f.write(f"{op.get_friendly_name()}\t{g('layerType')}\t{g('runtimePrecision')}\t{g('outputLayouts')}\t{g('primitiveType')}\t{g('originalLayersNames')[-300:]}\t"
                        f"{[str(i.get_partial_shape()) for i in op.inputs()]}\t{[str(o.get_partial_shape()) for o in op.outputs()]}\n")
    req = cm.create_infer_request()
    rng = np.random.default_rng(0)
    ids = rng.integers(1000, 50000, size=(1, max(a.prompt_len, 8))).astype(np.int64)
    def step(tok_ids, past):
        T = tok_ids.shape[1]
        e = emb({"input": tok_ids})[0]
        pos = np.arange(past, past + T, dtype=np.int64)[None, None, :].repeat(4, 0)
        req.infer({"inputs_embeds": e, "attention_mask": np.ones((1, past + T), np.int64),
                   "position_ids": pos, "beam_idx": np.zeros(1, np.int32)})
        return req.get_tensor("logits").data[0, -1].copy()
    req.reset_state()
    if a.prompt_len > 0:
        t0 = time.time()
        for c0 in range(0, a.prompt_len, a.chunk):  # chunked: full-sequence logits of a 2k prompt exhaust GPU memory on the 27B
            lg = step(ids[:, c0:c0 + a.chunk], c0)
        print(f"prefill {a.prompt_len} tok (chunks of {a.chunk}): {time.time()-t0:.2f}s")
    else:
        # NOTE: GPU PERF_COUNT values are running averages over ALL infer() calls since compile_model,
        # so a prefill would pollute the decode breakdown. --prompt-len 0 => decode-only profile.
        # (start the state with an 8-token chunk)
        lg = step(ids[:, :8], 0); print("decode-only profile (8-token start, no real prefill)")
    past = max(a.prompt_len, 8); times = []
    for i in range(a.steps):
        tok = np.array([[int(lg.argmax())]], np.int64)
        t0 = time.time(); lg = step(tok, past); times.append(time.time() - t0); past += 1
    print("decode ms/step:", [round(t * 1000, 1) for t in times], "median", round(1000 * float(np.median(times)), 1))
    by_type = collections.Counter(); by_exec = collections.Counter(); cnt = collections.Counter()
    tot = 0
    for p in req.profiling_info:
        us = p.real_time.total_seconds() * 1e6
        if us <= 0: continue
        tot += us; by_type[p.node_type] += us; by_exec[(p.node_type, p.exec_type)] += us; cnt[(p.node_type, p.exec_type)] += 1
    print(f"sum of kernel times last step: {tot/1000:.2f} ms")
    show = set(x for x in a.show.split(",") if x)
    for p in req.profiling_info:
        if p.node_type in show and (".layers.0." in p.node_name or ".layers.1." in p.node_name or "layers" not in p.node_name):
            print(f"    node {p.real_time.total_seconds()*1e6:8.1f} us  {p.node_type} {p.exec_type} {p.node_name[-90:]}")
    for (k, v) in by_exec.most_common(a.top):
        print(f"  {v/1000:8.2f} ms {100*v/tot:5.1f}%  n={cnt[k]:4d}  {k[0]} | {k[1]}")
if __name__ == "__main__":
    main()
