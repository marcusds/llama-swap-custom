# llama-bench baseline — b10199 (SYCL, Arc B50)

Image: `ghcr.io/marcusds/llama-swap-sycl-2026:latest`
llama.cpp commit: `876a4321163249c43ca4e986818fab5ab081f282` (b10199)
Date: 2026-08-01
Command: `llama-bench -m <model> -ngl 99 -fa 1 -p 512 -n 128`

| model                          | size     | params | test  | t/s          |
| ------------------------------ | -------- | ------ | ----- | ------------- |
| qwen3 8B Q4_K - Medium          | 4.68 GiB | 8.19 B | pp512 | 871.58 ± 6.15 |
| qwen3 8B Q4_K - Medium          | 4.68 GiB | 8.19 B | tg128 | 33.17 ± 0.09  |
| qwen35 9B Q4_K - Medium         | 5.70 GiB | 9.20 B | pp512 | 878.29 ± 2.40 |
| qwen35 9B Q4_K - Medium         | 5.70 GiB | 9.20 B | tg128 | 27.36 ± 0.08  |

Re-run the identical command against the same models once b10208 lands to compare.

## b10217 comparison (2026-08-01)

llama.cpp commit: `ddd4ec1428a6201e18975ea52b07c71e0f9aef26` (b10217 — one release past the b10208/b10216 SYCL perf work mentioned; b10216 was superseded before we could pin it separately)

| model                   | test  | b10199 t/s    | b10217 t/s    | delta  |
| ----------------------- | ----- | ------------- | ------------- | ------ |
| qwen3 8B Q4_K_M          | pp512 | 871.58 ± 6.15 | 882.06 ± 6.37 | +1.2%  |
| qwen3 8B Q4_K_M          | tg128 | 33.17 ± 0.09  | 33.13 ± 0.04  | ~0%    |
| qwen3.5 9B Q4_K_XL       | pp512 | 878.29 ± 2.40 | 911.07 ± 3.44 | +3.7%  |
| qwen3.5 9B Q4_K_XL       | tg128 | 27.36 ± 0.08  | 27.54 ± 0.43  | +0.7%  |

Modest prompt-processing gains (~1-4%), generation throughput essentially flat on these
two dense Q4 models on the Arc B50. If the release's headline SYCL improvement targets a
specific kernel path (MoE, a particular quant type, flash-attn variant, etc.) not exercised
by these two models, it may not show up here — worth re-testing against whatever model/config
the release notes call out specifically.
