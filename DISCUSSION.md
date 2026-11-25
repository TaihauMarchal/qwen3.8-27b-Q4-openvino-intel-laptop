<!-- Posted as https://github.com/openvinotoolkit/openvino/discussions/40307 (category "Show and tell"). -->

**Title:** Qwen3.8-27B INT4 at 11–13 tokens/s on a laptop iGPU (Arc 140T) with MTP + prefix caching

Hi everyone,

I wanted to share some first-hand numbers from running **Qwen3.8-27B INT4**
([OpenVINO/Qwen3.8-27B-int4-ov](https://huggingface.co/OpenVINO/Qwen3.8-27B-int4-ov)) locally on a 14" laptop:
Core Ultra 9 285H, integrated **Arc 140T**, 32 GB shared LPDDR5X, Ubuntu 26.04.

| | Decode t/s |
|---|---|
| OpenVINO 2026.4.1, no speculative decoding | 4.6 |
| + MTP speculative decoding (3 draft tokens) | 8.1 |
| Final stack, mixed prompts | 11.2–11.3 |
| Final stack, 15-round tool-calling loop (T=0.6) | 13.2 (rounds 11–17) |
| Final stack, 13k-token context | 11.2 (MTP with stock plugin: 9.4) |

Prefix caching together with MTP on this hybrid model brings the TTFT of follow-up turns from 16–19 s down to
~2 s (below 1 s in long agent loops) — great to see this now supported upstream with #4499.

What helped, beyond the upstream MTP support:

- the GQA multi-token PagedAttention path for verify passes (openvino#38493 / #39318) and the paged
  Gated DeltaNet rework (openvino#38793), backported onto 2026.4.1 — a big win at long context;
- restricting the **MTP draft head to a 65k-token vocabulary subset** (+26 % together with running text
  embeddings on CPU; outputs stay exact since the main model verifies over the full vocabulary) —
  proposed here: https://github.com/openvinotoolkit/openvino.genai/issues/5027 (PR https://github.com/openvinotoolkit/openvino.genai/pull/5029);
- an **adaptive draft length** (stop drafting when the draft's confidence product < 0.5): +7.8 % overall,
  +24 % on tool-call outputs;
- an optimized non-paged GatedDeltaNet GPU kernel — https://github.com/openvinotoolkit/openvino/pull/40286.

Decode on this iGPU is memory-bound (the stock INT4 FC kernels reach ~95 % of the 76–80 GB/s I can measure),
so one token per weight pass tops out around 5.5 t/s; speculative decoding is what makes a 27B usable here.

Full write-up, measurements, scripts and lessons learned (memory headroom vs i915 GPU hangs, thermals,
profiling pitfalls): https://github.com/TaihauMarchal/qwen3.8-27b-Q4-openvino-intel-laptop

Thanks to the OpenVINO and GenAI teams for MTP and the hybrid-model work — happy to share logs or run tests
on this hardware if useful.

_AI assistance: I used Claude Code (an AI coding agent) for parts of the implementation and the write-up;
all numbers above were measured on my own machine._
