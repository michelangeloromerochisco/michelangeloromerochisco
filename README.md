# Michelangelo Romero Chisco

Building open ternary language models from Bogotá, Colombia

---

## Projects

### 🌸 [Orchid 1.0](https://huggingface.co/MicheRomChis/orchid-1.0) — First competitive LLM built in Colombia
2B ternary-weight model ({−1, 0, +1} at 1.58 bits/weight) fine-tuned and aligned with ORPO entirely on a 4 GB laptop GPU. Trained end-to-end with no cloud compute.

| Benchmark | Score | vs BitNet base |
|-----------|------:|---------------:|
| ARC-Challenge | **56.0%** | +6.1 pp |
| WinoGrande | **74.0%** | — |
| Internal benchmark | **#3 of 12** (87.9%) | above all tested open-weight models |

### ⚙️ [ternative](https://github.com/michelangeloromerochisco/ternative) — Inference engine for ternary LLMs
C++/CUDA inference engine built to solve the **ternary merge problem**: LoRA deltas (≈10⁻⁵) are silently erased when merged into ternary weights (≈1.2 scale) and re-quantized. ternative keeps the adapter separate and applies it at full F32 precision at load time.

- OpenAI-compatible HTTP server
- GPU-resident decode: ~6–7 tok/s on RTX 3050 (4 GB)
- CPU fallback: ~6 tok/s with AVX2
- All 30 transformer layers fit in 4 GB VRAM (F16 + INT8 mixed)

### 🔜 Terse — Ternary sparse transformer family *(in development)*
Clean-room ternary sparse transformer: Mini (1.5B/4.5B params), Medium (9B/27B), Pro (27B/81B). MoE routing, hybrid KDA+MLA attention, recurrent depth, vision. Targets the same consumer hardware as Orchid.

---

## Stack

`C++17` · `CUDA` · `Python` · `PyTorch` · `PEFT` · `TRL` · `GGUF`

---

[![HuggingFace](https://img.shields.io/badge/🤗-MicheRomChis-yellow)](https://huggingface.co/MicheRomChis)
[![Paper](https://img.shields.io/badge/Paper-Orchid%201.0-red)](https://huggingface.co/MicheRomChis/orchid-1.0/blob/main/orchid-1-0-technical-paper.pdf)
[![ternative](https://img.shields.io/badge/GitHub-ternative-black)](https://github.com/michelangeloromerochisco/ternative)
