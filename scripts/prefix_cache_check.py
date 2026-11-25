#!/usr/bin/env python3
"""Append-only multi-turn loop (the shape of an agent / tool-calling conversation) with prefix caching
on or off, MTP speculative decoding on, greedy.

Run it twice (--prefix-cache 0 and 1) and compare:
  * TTFT of turns 2..N: with a working prefix cache only the new tokens are prefilled;
  * the generated texts: they must be byte-identical with and without the cache (greedy).

usage: prefix_cache_check.py MODEL_DIR --tag NAME --prefix-cache 0|1 [--entries 40] [--turns 3] [--nat 3]
       [--max-new 80] [--interval-mult 0] [--cache-gb 2]
"""
import argparse, json, random, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITES = ["north warehouse", "head office", "data room", "south shop", "airport kiosk", "lab"]
PEOPLE = ["Alex", "Sam", "Robin", "Charlie", "Jordan", "Morgan"]
EVENTS = ["switch rebooted after a power cut", "printer queue stuck", "VPN tunnel down", "disk 90 % full",
          "TLS certificate close to expiry", "backup job failed", "Wi-Fi access point offline"]


def journal(n, seed=0):
    """Synthetic incident journal with planted facts the follow-up questions refer to."""
    rnd = random.Random(seed)
    lines = []
    for i in range(n):
        lines.append(f"{i + 1:03d}. Site: {rnd.choice(SITES)}. Technician: {rnd.choice(PEOPLE)}. "
                     f"Event: {rnd.choice(EVENTS)}. Fix applied and verified, ticket closed after "
                     f"{rnd.randint(5, 90)} minutes.")
    lines.insert(n // 3, "PLANTED: the VPN tunnel of the north warehouse was restored by Robin with "
                         "`systemctl restart wg-quick@wg0`.")
    lines.insert(2 * n // 3, "PLANTED: the TLS certificate of the internal registry was renewed by Morgan; "
                             "it now expires on 17 August 2027.")
    return "\n".join(lines)


FOLLOW = [
    "Tool result `wg show`: interface wg0, latest handshake 12 seconds ago. According to the journal, which "
    "command restored this tunnel, and on which site?",
    "Tool result `openssl s_client`: notAfter=Aug 17 2027. Is this consistent with the journal? Answer in one "
    "sentence, then list the technicians you mentioned.",
    "Summarise the three most frequent kinds of events in the journal.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prefix-cache", type=int, default=1)
    ap.add_argument("--entries", type=int, default=40)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--nat", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--interval-mult", type=int, default=0,
                    help="SchedulerConfig.cache_interval_multiplier when the GenAI build has it (0 = default)")
    ap.add_argument("--cache-gb", type=int, default=2)
    a = ap.parse_args()

    import openvino as ov
    import openvino_genai as g
    info = {"openvino": ov.get_version(), "genai": getattr(g, "__version__", "?"), "prefix_cache": a.prefix_cache}
    print(json.dumps(info), flush=True)
    sc = g.SchedulerConfig()
    sc.enable_prefix_caching = bool(a.prefix_cache)
    sc.cache_size = a.cache_gb
    if a.interval_mult:
        sc.cache_interval_multiplier = a.interval_mult
    sc.max_num_seqs = 1
    sc.max_num_batched_tokens = 1024
    t0 = time.time()
    pipe = g.VLMPipeline(a.model, "GPU", scheduler_config=sc, draft_model=g.draft_model(a.model, "GPU"))
    print(f"load {time.time() - t0:.1f}s", flush=True)
    tok = pipe.get_tokenizer()
    msgs = [{"role": "user", "content": "Here is an incident journal:\n\n" + journal(a.entries) +
             "\n\nWho restored the VPN tunnel of the north warehouse, and when does the registry certificate expire?"}]
    res = {"tag": a.tag, "info": info, "turns": []}
    for turn in range(a.turns):
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, extra_context={"enable_thinking": False})
        cfg = pipe.get_generation_config()
        cfg.max_new_tokens = a.max_new
        cfg.apply_chat_template = False
        cfg.do_sample = False
        cfg.presence_penalty = 0.5
        cfg.num_assistant_tokens = a.nat
        r = pipe.generate(prompt, images=[], generation_config=cfg)
        pm = r.perf_metrics
        n = pm.get_num_generated_tokens()
        ttft = pm.get_ttft().mean
        dec = pm.get_generate_duration().mean - ttft
        o = {"turn": turn + 1, "in_tok": pm.get_num_input_tokens(), "gen": n, "ttft_ms": round(ttft),
             "decode_tps": round((n - 1) / (dec / 1000), 3) if dec > 0 else None, "text": r.texts[0]}
        print(json.dumps({k: v for k, v in o.items() if k != "text"}), flush=True)
        res["turns"].append(o)
        msgs.append({"role": "assistant", "content": r.texts[0]})
        msgs.append({"role": "user", "content": FOLLOW[turn % len(FOLLOW)]})
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / f"{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("saved", HERE / "logs" / f"{a.tag}.json")


if __name__ == "__main__":
    main()
