#!/usr/bin/env python3
"""Build a static token subset for the MTP draft head's lm_head (CPU only).

Why: on Qwen3.8 the vocabulary has 248,320 tokens, so the draft head's lm_head matmul and the draft
sampling are a visible part of every draft step. Restricting the draft to the ~65k tokens that the model
actually emits in your workload keeps acceptance unchanged in practice (the main model still verifies
every token over the full vocabulary, so outputs are exact) and makes each draft step cheaper.

Ranking = token frequency in your text corpora (+ optional word-frequency lists, + the model's own past
generations, weighted x2). Always kept: byte tokens, added/special tokens, every short letter-free token
(digits, punctuation, markdown, whitespace runs).

usage: build_draft_vocab.py MODEL_DIR OUT.txt [--size 65536] [--text FILE ...] [--wordfreq FILE[:WEIGHT] ...]
                            [--generations JSON ...]
  --text         plain-text files tokenised as is (docs, code, mails in your languages)
  --wordfreq     "word count" per line (e.g. hermitdave/FrequencyWords lists); default weight 1.0
  --generations  JSON files written by bench_cb.py (their "runs[].text" fields)
Output: one token id per line, sorted. Coverage check: run the same script logic on held-out generations.
"""
import argparse, collections, json
from pathlib import Path
import numpy as np
import openvino_genai as g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=65536)
    ap.add_argument("--text", nargs="*", default=[])
    ap.add_argument("--wordfreq", nargs="*", default=[])
    ap.add_argument("--generations", nargs="*", default=[])
    a = ap.parse_args()

    tok = g.Tokenizer(a.model)
    enc = lambda s: np.array(tok.encode(s, add_special_tokens=False).input_ids.data).ravel().tolist()
    score = collections.Counter()

    cnt = collections.Counter()
    for f in a.text:
        t = Path(f).read_text(errors="ignore")
        for j in range(0, len(t), 20000):
            cnt.update(enc(t[j:j + 20000]))
    tot = max(sum(cnt.values()), 1)
    for k, v in cnt.items():
        score[k] += v / tot

    for spec in a.wordfreq:
        path, _, w = spec.partition(":")
        wgt = float(w) if w else 1.0
        wt = collections.Counter()
        for line in open(path, encoding="utf-8"):
            parts = line.split()
            if len(parts) != 2:
                continue
            word, c = parts[0], int(parts[1])
            # mid-sentence, glued, sentence start, line start
            for form, f in ((" " + word, 0.6), (word, 0.1), (" " + word.capitalize(), 0.2), (word.capitalize(), 0.1)):
                for t in enc(form):
                    wt[t] += c * f
        wtot = max(sum(wt.values()), 1)
        for k, v in wt.items():
            score[k] += wgt * v / wtot

    own = collections.Counter()
    for f in a.generations:
        for r in json.load(open(f)).get("runs", []):
            own.update(enc(r.get("text", "")))
    otot = max(sum(own.values()), 1)
    for k, v in own.items():
        score[k] += 2.0 * v / otot

    tj = json.load(open(Path(a.model) / "tokenizer.json"))
    added = [t["id"] for t in tj.get("added_tokens", [])]
    vocab_size = max(max(tj["model"]["vocab"].values()), max(added, default=0)) + 1
    keep = set(range(256)) | set(added) | set(own)
    for i in range(vocab_size):
        try:
            s = tok.decode([i], skip_special_tokens=False)
        except Exception:
            continue
        if s and len(s) <= 6 and not any(ch.isalpha() for ch in s) and all(ord(ch) < 0x3000 for ch in s):
            keep.add(i)
    forced = len(keep)
    for t, _ in score.most_common():
        if len(keep) >= a.size:
            break
        keep.add(t)
    Path(a.out).write_text("\n".join(map(str, sorted(keep))) + "\n")
    print(f"wrote {a.out}: {len(keep)} tokens ({forced} forced) out of {vocab_size}")


if __name__ == "__main__":
    main()
