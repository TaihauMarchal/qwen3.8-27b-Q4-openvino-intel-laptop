#!/usr/bin/env python3
"""Decode benchmark of a Qwen3.5/3.8 OpenVINO model through the continuous-batching VLMPipeline,
optionally with the MTP draft head (speculative decoding), greedy.

usage: bench_cb.py MODEL_DIR --tag NAME [--nat 3,4] [--no-mtp] [--prefix-cache 0|1] [--cache-gb 2]
                   [--mbt 1024] [--max-new 200] [--prompts short,code,doc] [--doc FILE] [--long-words 0]
                   [--props JSON] [--draft-props JSON]

Writes logs/<tag>.json next to this script: per prompt TTFT, decode tokens/s, tokens per verify pass,
ms per pass, and the generated text (diff two runs to check that an optimisation does not change outputs).
"""
import argparse, json, os, statistics as st, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = {
    "short": "Explain in a few paragraphs the difference between WireGuard and OpenVPN, and when to choose one or "
             "the other for a small company network.",
    "code": "Write a Python function `parse_log(path)` that reads an nginx access log, counts requests per HTTP status "
            "code and per client IP, and returns the 10 most active IPs. Add docstrings and a short usage example.",
    "mail": "Write a professional email to an IT team announcing a Wi-Fi network maintenance on Saturday from 10 pm to "
            "2 am, with the expected impact, the services that are not affected and a contact in case of problems.",
}
DOC = ("A small company runs its services on a few Linux servers behind a VPN. Backups are taken every night and "
       "copied off-site once a week. Every change to the firewall must be reviewed by a second person. Incidents are "
       "written down in a shared journal with the date, the symptom, the root cause and the fix. ") * 20


def long_prompt(words):
    base = DOC.split()
    out = []
    while len(out) < words:
        out.extend(base)
    return "Here is a long document:\n\n" + " ".join(out[:words]) + "\n\nWrite a detailed summary."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--props", default="{}", help="JSON compile properties for the main model")
    ap.add_argument("--draft-props", default=None, help="JSON compile properties for the draft (default: --props)")
    ap.add_argument("--nat", default="3", help="comma list of num_assistant_tokens values")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--prefix-cache", type=int, default=0)
    ap.add_argument("--cache-gb", type=int, default=2)
    ap.add_argument("--mbt", type=int, default=1024, help="SchedulerConfig.max_num_batched_tokens")
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--prompts", default="short,code,doc")
    ap.add_argument("--doc", help="text file used for the 'doc' prompt (default: a built-in synthetic document)")
    ap.add_argument("--no-streamer", action="store_true")
    ap.add_argument("--long-words", type=int, default=0, help="also measure TTFT on a synthetic document of N words")
    ap.add_argument("--long-new", type=int, default=32)
    ap.add_argument("--warm", type=int, default=1)
    a = ap.parse_args()

    import openvino as ov
    import openvino_genai as g
    info = {"openvino": ov.get_version(), "genai": getattr(g, "__version__", "?"), "genai_file": g.__file__}
    print(json.dumps(info), flush=True)
    doc = Path(a.doc).read_text() if a.doc else DOC
    P["doc"] = "Here is a document:\n\n" + doc + "\n\nSummarise this document and list its main rules."

    props = json.loads(a.props)
    dprops = json.loads(a.draft_props) if a.draft_props is not None else props
    sc = g.SchedulerConfig()
    sc.enable_prefix_caching = bool(a.prefix_cache)
    sc.cache_size = a.cache_gb
    sc.max_num_seqs = a.max_num_seqs
    sc.max_num_batched_tokens = a.mbt
    kw = dict(scheduler_config=sc, **props)
    if not a.no_mtp:
        kw["draft_model"] = g.draft_model(a.model, "GPU", **dprops)   # MTP head shipped inside the model folder
    t0 = time.time()
    pipe = g.VLMPipeline(a.model, "GPU", **kw)
    load_s = time.time() - t0
    print(f"load {load_s:.1f}s", flush=True)
    tok = pipe.get_tokenizer()

    def run(text, nat, max_new):
        prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                         extra_context={"enable_thinking": False})
        cfg = pipe.get_generation_config()
        cfg.max_new_tokens = max_new
        cfg.apply_chat_template = False
        cfg.do_sample = False
        cfg.presence_penalty = 0.5
        if not a.no_mtp:
            cfg.num_assistant_tokens = nat
        streamer = None if a.no_streamer else (lambda c: g.StreamingStatus.RUNNING)
        t0 = time.perf_counter()
        r = pipe.generate(prompt, images=[], generation_config=cfg, streamer=streamer)
        wall = time.perf_counter() - t0
        pm = r.perf_metrics
        raw = pm.raw_metrics
        n = pm.get_num_generated_tokens()
        ttft = pm.get_ttft().mean
        dec_ms = pm.get_generate_duration().mean - ttft
        inf = [d / 1000 for d in raw.token_infer_durations]
        bs = list(raw.m_batch_sizes)
        passes = len(bs)
        return {"in_tok": pm.get_num_input_tokens(), "gen": n, "ttft_ms": round(ttft), "wall_s": round(wall, 2),
                "decode_tps": round((n - 1) / (dec_ms / 1000), 3) if dec_ms > 0 else None,
                "passes": passes, "tok_per_pass": round(sum(bs[1:]) / max(passes - 1, 1), 3),
                "ms_per_pass": round(dec_ms / max(passes - 1, 1), 1),
                "infer_ms_per_pass_med": round(st.median(inf[1:]), 1) if len(inf) > 2 else None,
                "text": r.texts[0]}

    res = {"tag": a.tag, "info": info, "args": vars(a), "load_s": round(load_s, 1), "runs": []}
    nats = [int(x) for x in a.nat.split(",")]
    for _ in range(a.warm):                     # warm-up: kernel compilation, first allocations
        run(P["short"], nats[0], 24)
    for nat in nats:
        for name in [x for x in a.prompts.split(",") if x]:
            o = run(P[name], nat, a.max_new)
            o.update(prompt=name, nat=nat)
            print(json.dumps({k: v for k, v in o.items() if k != "text"}), flush=True)
            res["runs"].append(o)
    if a.long_words:
        o = run(long_prompt(a.long_words), nats[0], a.long_new)
        o.update(prompt=f"long{a.long_words}", nat=nats[0])
        print(json.dumps({k: v for k, v in o.items() if k != "text"}), flush=True)
        res["runs"].append(o)
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / f"{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("saved", HERE / "logs" / f"{a.tag}.json")


if __name__ == "__main__":   # required: openvino_genai may spawn a child that re-imports this file
    main()
