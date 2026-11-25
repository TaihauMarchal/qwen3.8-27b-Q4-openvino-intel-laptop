# Qwen3.8-27B on a laptop iGPU with OpenVINO: from 4.6 to 11–13 tokens/s

A 27-billion-parameter model, INT4, running **locally on a 14" laptop** (Intel Core Ultra 9 285H, integrated
Arc 140T GPU, 32 GB shared LPDDR5X), with OpenVINO and OpenVINO GenAI. No discrete GPU, no cloud.

| | Decode speed (Qwen3.8-27B INT4, Arc 140T) |
|---|---|
| Stock OpenVINO 2026.4.1, no speculative decoding | **4.6 tokens/s** |
| Final stack, 8 mixed prompts (prose + tool calls), greedy / T=0.6 | **11.2–11.3 tokens/s** |
| Final stack, 15-round tool-calling agent loop, T=0.6 | **13.2 tokens/s** (individual rounds 11–17) |
| Final stack, 13k-token context | **11.2 tokens/s** (stock speculative: 9.4) |

Time to first token on the 2nd, 3rd… turn of a conversation went from **16–19 s to ~2 s** (and below 1 s on
long agent loops) thanks to prefix caching combined with speculative decoding.

This repository documents how: the measurements, what each step brought, what came from upstream OpenVINO
and what I built myself, the scripts to reproduce the measurements, and the research patches.

> Built with the help of Claude Code (an AI coding agent). All measurements were run and checked by me on
> my own machine; numbers below come from the benchmark logs, not from estimates.

---

## 1. Hardware and the bandwidth ceiling

- ASUS Zenbook 14 (UX3405CA), Intel Core Ultra 9 285H, Arc 140T iGPU (Xe-LPG+, arch 12.74), 32 GB LPDDR5X-8400
  running at 7467 MT/s (BIOS setting), shared between CPU and GPU.
- Ubuntu 26.04, kernel 7.0, i915 driver, Intel compute runtime (NEO) 26.05.37020.3, IGC 2.28.4.
- Model: [OpenVINO/Qwen3.8-27B-int4-ov](https://huggingface.co/OpenVINO/Qwen3.8-27B-int4-ov) — hybrid architecture
  (48 Gated DeltaNet linear-attention layers + 16 full-attention layers), built-in MTP head.

Decoding one token of a dense 27B model means streaming **all its weights** from memory:
13.45 GB of INT4 weights (u4, group 128, asymmetric) + a 1.27 GB u8 `lm_head` (248,320-token vocabulary).

| | GB/s |
|---|---|
| Theoretical (128-bit bus × 7467 MT/s) | 119 |
| Measured sequential read from the iGPU (OpenCL micro-benchmark, USM device memory) | 76–80 |
| Stock fully-connected kernels during decode (13.9 GB in 187 ms) | 74.5 (≈ 95 % of the measured ceiling) |

So the stock kernels are already near the hardware limit, and **one token per weight pass cannot go much
beyond ~5.5 tokens/s on this machine**. Profile of one stock decode step (decode-only PERF_COUNT, 2k context):

| Op | Share of GPU time |
|---|---|
| FullyConnectedCompressed (oneDNN, INT4 weights) | 89.4 % |
| GatedDeltaNet (reference kernel) | 4.6 % |
| Attention (SDPA), RMSNorm, transposes, slices… | ~6 % |

The only big lever is therefore **speculative decoding**: verify several tokens per weight pass.
A 4-token verify pass costs about 1.2–1.4× a 1-token pass, and the MTP head proposes tokens that the main
model accepts with probability 0.79 / 0.48 / 0.29 / 0.19 / 0.12 for draft positions 1..5 (measured on 339
passes). Everything below is about making those passes cheaper, longer and usable in real conversations.

## 2. Step by step

All numbers: Qwen3.8-27B INT4 on the Arc 140T, continuous-batching `VLMPipeline`, `max_num_seqs=1`, ~200
generated tokens per prompt, greedy unless stated. Machine speed drifts ±3–5 % between sessions, so each
comparison below was run back to back in the same session.

| # | Change | Decode t/s | ms / pass | tokens / pass | Origin |
|---|---|---|---|---|---|
| 0 | Stock OpenVINO 2026.4.1, no speculative decoding | 4.60 | 217 (1 token) | 1 | upstream |
| 1 | MTP speculative decoding, 3 draft tokens | 8.07 | 360 | 2.91 | upstream feature, my integration |
| 2 | Draft head restricted to a 65k-token vocabulary subset + text embeddings on CPU + draft KV-cache sizing | 10.17 (+26 %) | 285 | 2.90 | **mine** |
| 3 | Adaptive draft length (up to 5 tokens, stop when the draft's confidence product < 0.5) | 11.16 (+7.8 % on its prompt set) | — | — | **mine** |
| 4 | GPU plugin: backported upstream kernels + my GatedDeltaNet / permute kernels + parallel weight loading | 11.2 short context; **8.6k ctx 10.3 → 11.4, 13k ctx 9.4 → 11.2** | 334 at 13k (was 387) | 3.3 | upstream backport + **mine** |
| 5 | Prefix caching **together with** MTP on the hybrid model (turn 2+ TTFT 16–19 s → 1.8–2.2 s) | — | — | — | **mine** on 2026.4.1; solved independently upstream in openvino.genai#4499 |

Optional, off by default: quantizing the activations of the multi-token verify passes to int8
(`OV_GPU_DQ_THRESHOLD=1`, see §4) gives another −4 to −7 % ms/pass (11.7–11.9 t/s greedy) but changes the
generated text slightly.

### What came from upstream (thank you, OpenVINO team)

- MTP speculative decoding for Qwen3.5/3.8 in OpenVINO GenAI.
- `[GPU] pa optimization for speculative verify stage` —
  [openvino#38493](https://github.com/openvinotoolkit/openvino/pull/38493) and its tuning
  [openvino#39318](https://github.com/openvinotoolkit/openvino/pull/39318): a GQA multi-token PagedAttention path.
  On this iGPU it is the fallback (non-DPAS) variant that runs. Backported onto 2026.4.1 it cut a 4-token verify
  pass at 8.6k context from 388.5 to 338.6 ms.
- Paged Gated DeltaNet kernel rework, [openvino#38793](https://github.com/openvinotoolkit/openvino/pull/38793):
  −7 to −9 % TTFT (GDN time per 256-token prefill chunk 119 → 42 ms).
- Constant prefetch during compilation, [openvino#38993](https://github.com/openvinotoolkit/openvino/pull/38993):
  I reimplemented the idea locally in the GPU plugin as a parallel constant copy — cold load of the language
  model 35.9 → 22.5 s.
- Prefix caching for hybrid models in speculative decoding,
  [openvino.genai#4499](https://github.com/openvinotoolkit/openvino.genai/pull/4499) — merged upstream
  (2026-10-12), after the 2026.4.1 release branch I was working on (2026-09-21), which still rejected this
  combination; my own implementation (below) solved the same problem on 2026.4.1.
- No padding of 2–4-row INT4 fully-connected layers to 16 rows,
  [openvino#38795](https://github.com/openvinotoolkit/openvino/pull/38795) — the same fix I had made locally.

### What I built

- **Integration and backport** of the above onto the 2026.4.1 release (11 files, merged without conflicts),
  validated with byte-identical outputs where expected.
- **Prefix caching + MTP on a hybrid model** (GenAI): snapshots of the linear-attention state at block
  boundaries (LRU), KV prefix cache, MTP draft alignment, a fix of the block hash when prefill is chunked.
  Greedy outputs byte-identical with and without the cache on the 4B (3 turns) and 27B (4 turns).
  27B, 4 turns: TTFT of turns 2–4 **16.3 / 17.7 / 18.6 s → 1.84 / 2.15 / 2.05 s**.
  Long 8-turn agent scenario (9.1k → 11k tokens): first TTFT 26.6 s, then **0.77–0.86 s** per turn.
- **Draft vocabulary subset**: the MTP head scores 248,320 tokens per draft step; restricting it to the 65,536
  tokens this workload actually uses covers 99.87 % of held-out generated tokens, keeps tokens/pass unchanged,
  and the main model still verifies over the full vocabulary (outputs stay exact).
  Script: [`scripts/build_draft_vocab.py`](scripts/build_draft_vocab.py).
  Upstream proposal: _link to the openvino.genai issue / PR — TODO_.
- **Adaptive draft length**: draft up to 5 tokens but stop as soon as the product of the draft's top-1
  probabilities drops below 0.5. Fixed 3 tokens was the best fixed setting; adaptive adds +7.8 % overall:
  tool-call outputs (very predictable) 10.45 → 12.94 t/s (+24 %), prose 10.29 → 10.08 t/s (−2 %).
- **Optimized GatedDeltaNet GPU kernel** (non-paged / stateful path): GDN time per decode step
  **9.62 → 5.29 ms** (48 layers), full decode step median 254.2 → 245.2 ms (−3.5 %, research build on 2026.4.1,
  together with the permute kernel below).
  Upstream PR: [openvinotoolkit/openvino#40286](https://github.com/openvinotoolkit/openvino/pull/40286) — rewritten
  on master and limited to single-token decode, where it wins: **1.4× on the kernel** (0.128 → 0.092 ms per call,
  Qwen3.8 linear-attention shape), under 1 % of a 27B decode step.
- **Dynamic-shape f↔y permute kernel** replacing the reference permute: transposes in a 512-token prefill
  chunk 78.2 → 37.2 ms.
- **Non-finite-logits guard**: one generation out of 111 in one configuration produced NaN logits (cause not
  established). GenAI now checks every logits row, stops the request with a clear error and flushes the
  prefix caches; the serving worker retries once. Overhead: +0.2 ms per ~313 ms pass.
- **Serving hardening**: stall watchdog, restart on OpenCL errors, memory returned to the OS after each
  request, memory-headroom rules (see lessons).

## 3. Results in detail

Final stack vs. the same GenAI build with the stock GPU plugin (A = stock plugin, B = final plugin), 27B,
`num_assistant_tokens` 5 (adaptive), `max_num_batched_tokens` 1024:

| Workload | A t/s | B t/s | A ms/pass | B ms/pass |
|---|---|---|---|---|
| 8 prompts, greedy | 11.12 | 11.17–11.21 | 289 | 296 |
| 8 prompts, T=0.6 | 11.50–11.53 | 11.24–11.28 | 289 | 296 |
| 1.5k-token context | 12.57 | 12.10 | 304 | 306 |
| 8.6k-token context | 10.28 | 11.43 | 350 | 319 |
| 13k-token context | 9.36 | 11.20 | 387 | 334 |
| Tool-calling loop (15 rounds, 3 conversations, T=0.6) | 12.27 | 13.21 | 333 | 322 |

The plugin changes matter at long context (where attention cost grows) and in agent loops, not on short
prompts. Long-context answers were checked with planted facts (6/6 correct at 1.5k, 8.6k and 13k tokens,
greedy and sampled) and 24/24 tool calls were well-formed.

Load time of the pipeline: 40.1 → 31.7 s (mean of the benchmark runs).

## 4. Reproduce

Scripts (Python, OpenVINO / OpenVINO GenAI wheels):

| Script | What it measures |
|---|---|
| [`scripts/bench_cb.py`](scripts/bench_cb.py) | decode t/s, tokens/pass, ms/pass, TTFT through `VLMPipeline` with or without MTP; saves texts to diff runs |
| [`scripts/prefix_cache_check.py`](scripts/prefix_cache_check.py) | multi-turn append-only loop; TTFT per turn and cache on/off output equality |
| [`scripts/profile_step.py`](scripts/profile_step.py) | per-op GPU profile of prefill + decode steps (stateful model) |
| [`scripts/profile_paged.py`](scripts/profile_paged.py) | per-op GPU profile of K+1-token verify passes in the paged (continuous-batching) form, at a chosen context length |
| [`scripts/build_draft_vocab.py`](scripts/build_draft_vocab.py) | builds the draft vocabulary subset |

```bash
# stock wheels
python -m venv ov && . ov/bin/activate
pip install openvino==2026.4.1 openvino-genai==2026.4.1 openvino-tokenizers==2026.4.1
hf download OpenVINO/Qwen3.8-27B-int4-ov --local-dir qwen3.8-27b-int4-ov

python scripts/bench_cb.py qwen3.8-27b-int4-ov --tag nomtp --no-mtp
python scripts/bench_cb.py qwen3.8-27b-int4-ov --tag mtp3 --nat 3 --long-words 6000
python scripts/prefix_cache_check.py qwen3.8-27b-int4-ov --tag p1 --prefix-cache 1
```

With a current OpenVINO GenAI (which includes openvino.genai#4499) prefix caching + MTP works out of the box.
The draft vocabulary subset, adaptive draft length and the custom kernels need the research patches below
until they land upstream. Settings I use with them:

```
OV_MTP_DRAFT_VOCAB=vocab_65536.txt     # draft lm_head subset (build_draft_vocab.py)
OV_MTP_DRAFT_MIN_CONF=0.5              # adaptive draft length, num_assistant_tokens = 5 as the cap
OV_GENAI_TEXT_EMBEDDINGS_DEVICE=CPU    # embedding lookups on CPU: 17.1 -> 1.2-3.3 ms per pass, frees GPU memory
OV_MTP_DRAFT_KV_BLOCKS=auto            # size the draft KV cache from the main one (with the two above: +3.8 GB free)
OV_GENAI_LA_GEN_SNAPSHOT_GRID=32       # linear-attention snapshots during generation
OV_GENAI_LA_SHARED_PREFIX_MIN=512      # keep one snapshot for a shared system-prompt prefix
SchedulerConfig: max_num_seqs=1, max_num_batched_tokens=1024 (−15 % TTFT vs 256), cache_size=2 GB
# opt-in, changes outputs: OV_GPU_DQ_THRESHOLD=1  (int8 activations on multi-token verify passes)
```

### Research patches

In [`patches/`](patches/) — working code, validated as described above, **not** upstream-quality (environment
variables instead of properties, debug hooks). Focused upstream PRs are being prepared from them.

| Patch | Base | Content |
|---|---|---|
| `openvino.genai-mtp-prefix-cache.patch` | openvino.genai `ab01d17` (2026.4.1.0) | prefix cache + MTP on hybrid models, draft vocab subset, adaptive draft, NaN guard, profiling hooks, unit tests |
| `openvino-backport-long-context.patch` | openvino `e213a147` (2026.4.1) | backport of openvino#38493, #39318, #38793 + parallel constant copy |
| `openvino-qwen38-gpu-kernels.patch` | same, on top of the backport | GatedDeltaNet opt kernel, f↔y dynamic permute, FC no-padding (≤ 8 rows), DQ threshold override |

## 5. Lessons learned

- **Measure the ceiling first.** A 30-line OpenCL streaming benchmark told me the fully-connected kernels
  were already at 95 % of achievable bandwidth, which redirected the effort to speculative decoding instead
  of rewriting matmul kernels (I dropped my custom INT4 decode matmul: at most ~5 % left to gain).
- **iGPU memory is system memory.** Below ~5 GB of `MemAvailable`, GPU buffers get swapped and the i915 driver
  reports GPU hangs (also with the stock stack). With ≥ 7 GB free I have not seen a hang since. Keep the
  27B + draft + caches under ~23 GB, and move what you can to CPU (text embeddings) or size it (draft KV).
- **Thermals are a performance parameter.** A parallel build at -j4 reached 99 °C; overlapping heavy builds
  and GPU benchmarks once hard-reset the machine. One heavy job at a time, wait for the CPU to cool before a
  benchmark, compare variants back to back.
- **GPU `PERF_COUNT` is a running average per compiled network**: a prefill pollutes a decode profile
  (profile decode on a separate infer request), and multi-stage kernels may only report their last stage —
  trust wall time.
- **Check that your kernel actually runs.** My first custom matmul "passed" validation because it was never
  selected: below 64 rows the plugin skips dynamic quantization and feeds f16 activations, so an i8 kernel
  was rejected at runtime. Always look for a numerical difference (proof of execution) before timing.
- **Exactness checks catch everything.** Every optimization was validated by diffing greedy outputs
  (byte-identical where the math is unchanged) and, where numerics change, by planted-fact long-context tests.
- Things that did **not** help here: transparent huge pages for GPU buffers (no measurable effect), a newer
  compute runtime, more than 3 fixed draft tokens (+0.07 tokens/pass for a longer pass).

## Links

- OpenVINO: https://github.com/openvinotoolkit/openvino — OpenVINO GenAI: https://github.com/openvinotoolkit/openvino.genai
- My upstream contributions: [GatedDeltaNet kernel PR #40286](https://github.com/openvinotoolkit/openvino/pull/40286) · _draft vocabulary subset issue/PR — TODO_
- Discussion: _openvino.genai Show and tell — TODO_

## License

Scripts and documentation: Apache License 2.0 (same as OpenVINO). The patches are derived from OpenVINO /
OpenVINO GenAI (Apache 2.0); the backport patch contains upstream code by the OpenVINO authors.

---

### Résumé en français

Un modèle de 27 milliards de paramètres (Qwen3.8-27B, INT4) tourne en local sur un ultraportable
(Core Ultra 9 285H, iGPU Arc 140T, 32 Go partagés) avec OpenVINO. Le décodage passe de **4,6 jetons/s** en
configuration d'origine à **11–13 jetons/s** mesurés (11–17 sur les tours individuels d'un agent outillé), et le temps avant le premier jeton des tours suivants d'une
conversation de **16–19 s à ~2 s**. La bande passante mémoire limite le décodage classique à ~5,5 jetons/s :
le gain vient du décodage spéculatif MTP (fonction d'OpenVINO), de son couplage au cache de préfixe sur un
modèle hybride, d'un sous-vocabulaire pour la tête de brouillon, d'une longueur de brouillon adaptative et de
noyaux GPU (rétroportés de l'amont et écrits par moi). Les mesures, les scripts et les correctifs sont dans
ce dépôt ; les contributions amont sont en préparation.
