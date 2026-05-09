# TurboQuant Gemma 4 MTP

This repository is a best-effort merge of three experimental llama.cpp forks:

- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) for the newer TurboQuant baseline.
- [test1111111111111112/llama-cpp-turboquant-gemma4](https://github.com/test1111111111111112/llama-cpp-turboquant-gemma4) for the extended Gemma 4 CUDA kernel work and inference-speed customizations.
- [AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) for Gemma 4 MTP assistant support, including `--mtp-head`, `--spec-type mtp`, and server integration.

The goal of this fork is to keep the fresh TurboQuant implementation, preserve the UserTest/test-fork fast kernels that improve Gemma inference, and add Atomic's MTP speculative decoding path in one working tree. This is an integration fork, not an upstream llama.cpp PR.

## Recommended Gemma 4 MTP server mode

This fork is intended to run Gemma 4 with reasoning/thinking disabled. Use `--reasoning off --reasoning-format none --reasoning-budget 0` together with Flash Attention, GPU KV offload, and a TurboQuant V cache. The MTP assistant is loaded into the target model with `--mtp-head`:

```sh
TARGET_GGUF=/path/to/gemma-4-target.gguf
ASSISTANT_GGUF=/path/to/gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf
CHAT_TEMPLATE=/path/to/chat_template.jinja

CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  --model "$TARGET_GGUF" \
  --mtp-head "$ASSISTANT_GGUF" \
  --spec-type mtp \
  --draft-block-size 3 --draft-max 8 --draft-min 0 \
  --n-gpu-layers 99 --gpu-layers-draft 99 \
  --cache-type-k q8_0 --cache-type-v turbo4 \
  --cache-type-k-draft q8_0 --cache-type-v-draft turbo4 \
  --flash-attn on --kv-offload --fit-target 4096 \
  --batch-size 512 --ubatch-size 128 \
  --ctx-size 400000 --parallel 2 --timeout 600 \
  --reasoning off --reasoning-format none --reasoning-budget 0 \
  --jinja --chat-template-file "$CHAT_TEMPLATE" \
  --slots --slot-save-path /tmp/llamacpp-slot-cache/ \
  --sleep-idle-seconds -1 --no-warmup --no-webui --alias default \
  --host 127.0.0.1 --port 8080
```

The assistant model used for validation was [`AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF`](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF), file `gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf`. The tested KV cache mode was `K=q8_0, V=turbo4` for both the target and MTP assistant path.

## Benchmark notes: target-only vs MTP

The following numbers are a single-run, best-effort comparison on an RTX 3090-class CUDA setup. They are not an official llama.cpp benchmark suite, but they are useful for validating this fork's current Gemma 4 behavior.

Common settings:

- Target model: Gemma 4 26B A4B IT GGUF, `Q4_K_M`.
- KV cache: target `K=q8_0, V=turbo4`; MTP draft path also `K=q8_0, V=turbo4`.
- Server mode: `--flash-attn on`, `--ctx-size 8192`, `--batch-size 512`, `--ubatch-size 128`, `--parallel 1`.
- Reasoning disabled: `--reasoning off --reasoning-format none --reasoning-budget 0`.
- The matrix uses `ignore_eos=true` so that each cell measures the requested decode length rather than the model stopping early.
- Prompt sizes below are the actual server-reported prompt tokens: approximately 248, 1000, and 4055 tokens.

### Summary

| Mode | Assistant | Avg decode tok/s | Min decode tok/s | Max decode tok/s | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Target-only | none | 113.50 | 106.94 | 117.04 | Fastest and most stable in this setup. |
| MTP | `Q4_K_M` assistant | 95.30 | 79.61 | 102.54 | High acceptance on long generations, but still slower than target-only. |
| MTP | `F16` assistant | 61.13 | 46.72 | 67.78 | Slowest; not recommended for this target/cache setup. |

### Memory

| Mode | Assistant | RSS after load | RSS after matrix | VRAM after load | VRAM after matrix |
| --- | --- | ---: | ---: | ---: | ---: |
| Target-only | none | 1.02 GiB | 1.56 GiB | 16590 MiB | 16624 MiB |
| MTP | `Q4_K_M` assistant | 1.30 GiB | 2.37 GiB | 16674 MiB | 16708 MiB |
| MTP | `F16` assistant | 1.89 GiB | 2.96 GiB | 16868 MiB | 16904 MiB |

### Target-only baseline

| Prompt tokens | Generated tokens | Prompt eval | Prompt eval tok/s | Decode time | Decode tok/s | Wall time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 248 | 256 | 307.82 ms | 805.68 | 2187.29 ms | 117.04 | 2503 ms |
| 248 | 1024 | 206.31 ms | 1202.10 | 8782.32 ms | 116.60 | 8990 ms |
| 248 | 4096 | 208.36 ms | 1190.26 | 35861.30 ms | 114.22 | 36071 ms |
| 1000 | 256 | 1005.24 ms | 994.79 | 2238.49 ms | 114.36 | 3247 ms |
| 1000 | 1024 | 1007.16 ms | 992.90 | 8975.85 ms | 114.08 | 9986 ms |
| 1000 | 4096 | 1006.25 ms | 993.79 | 36179.69 ms | 113.21 | 37189 ms |
| 4055 | 256 | 5683.67 ms | 713.45 | 2271.50 ms | 112.70 | 7963 ms |
| 4055 | 1024 | 5690.32 ms | 712.61 | 9117.41 ms | 112.31 | 14816 ms |
| 4055 | 4096 | 5691.17 ms | 712.51 | 38301.20 ms | 106.94 | 44000 ms |

### MTP with `Q4_K_M` assistant

| Prompt tokens | Generated tokens | Prompt eval | Decode time | Decode tok/s | Draft accepted / generated | Acceptance | Wall time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 248 | 256 | 519.67 ms | 3215.71 ms | 79.61 | 124 / 261 | 47.51% | 3739 ms |
| 248 | 1024 | 251.72 ms | 10378.03 ms | 98.67 | 611 / 823 | 74.24% | 10638 ms |
| 248 | 4096 | 252.08 ms | 41622.08 ms | 98.41 | 2598 / 2991 | 86.86% | 41888 ms |
| 1000 | 256 | 1473.59 ms | 2776.99 ms | 92.19 | 149 / 212 | 70.28% | 4254 ms |
| 1000 | 1024 | 1180.31 ms | 10659.17 ms | 96.07 | 650 / 744 | 87.37% | 11851 ms |
| 1000 | 4096 | 1177.64 ms | 39945.82 ms | 102.54 | 2694 / 2801 | 96.18% | 41127 ms |
| 4055 | 256 | 6382.69 ms | 2578.77 ms | 99.27 | 164 / 182 | 90.11% | 8978 ms |
| 4055 | 1024 | 6385.99 ms | 10603.25 ms | 96.57 | 667 / 712 | 93.68% | 17005 ms |
| 4055 | 4096 | 6389.74 ms | 43412.40 ms | 94.35 | 2700 / 2788 | 96.84% | 49821 ms |

### MTP with `F16` assistant

| Prompt tokens | Generated tokens | Prompt eval | Decode time | Decode tok/s | Draft accepted / generated | Acceptance | Wall time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 248 | 256 | 479.77 ms | 5479.34 ms | 46.72 | 128 / 254 | 50.39% | 5961 ms |
| 248 | 1024 | 253.15 ms | 17610.00 ms | 58.15 | 616 / 813 | 75.77% | 17897 ms |
| 248 | 4096 | 251.14 ms | 66280.84 ms | 61.80 | 2605 / 2977 | 87.50% | 66561 ms |
| 1000 | 256 | 1473.54 ms | 4354.38 ms | 58.79 | 151 / 208 | 72.60% | 5831 ms |
| 1000 | 1024 | 1180.88 ms | 15736.84 ms | 65.07 | 651 / 742 | 87.74% | 16945 ms |
| 1000 | 4096 | 1180.39 ms | 60434.57 ms | 67.78 | 2698 / 2794 | 96.56% | 61618 ms |
| 4055 | 256 | 6439.72 ms | 4002.20 ms | 63.96 | 164 / 182 | 90.11% | 10470 ms |
| 4055 | 1024 | 6446.33 ms | 15758.19 ms | 64.98 | 668 / 710 | 94.08% | 22233 ms |
| 4055 | 4096 | 6448.77 ms | 65141.17 ms | 62.88 | 2703 / 2782 | 97.16% | 71617 ms |

### Interpretation

- With the current Gemma 4 target, CUDA build, and TurboQuant KV cache, MTP is not a throughput win yet. The target-only path averages about 113.5 tok/s across the matrix.
- The `Q4_K_M` assistant reaches high acceptance on long generations, but the extra draft cost still leaves it about 16% slower on average than target-only.
- The `F16` assistant is much heavier and averages about 46% slower than target-only. Earlier non-forced Russian generation checks also showed unstable text degeneration with this assistant, so it is not recommended for this setup.
- Prompt evaluation slows down as the input approaches 4k tokens, but decode speed is mostly stable in target-only mode until the full 4k prompt + 4k output case.

## Upstream documentation

This README intentionally documents only this integration fork. The original README content is not duplicated here. For general llama.cpp usage, build instructions, API details, and upstream project documentation, see the source repositories listed above and the original [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) repository.
