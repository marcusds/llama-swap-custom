# llama-swap SYCL — benchmarking handoff

Everything here runs **on home-server**. Scripts live in `~/llama-swap-bench/`.
Hardware: Intel Arc Pro B50 (Battlemage, BMG-G21), 16304 MiB VRAM, 70 W cap.

---

## 1. Current deployment

| | |
|---|---|
| image | `ghcr.io/marcusds/llama-swap-sycl-bmg:latest` |
| what it is | oneAPI 2026.1.2, AOT-compiled for `bmg_g21`, oneDNN restored, NEO 26.27 / IGC 2.38.2 |
| compose | `/home/marcus/containers/llama.cpp/compose.yaml` |
| model config | `/home/marcus/containers/llama.cpp/config/config.yaml` |
| rollback (image) | `compose.yaml.bak-20260801-043653` → `-2026:latest` |
| rollback (config) | `config/config.yaml.bak-20260801-045730` |

7 models on an f16 KV cache carry `env: ["GGML_SYCL_ENABLE_MKL_FA=0"]`.
11 models on a q8_0 KV cache deliberately do **not** — see §3.

---

## 2. Running the benchmarks

Every script stops the prod llama-swap container for the duration and restarts
it via an EXIT trap. Expect an LLM outage for the length of the run.

```bash
cd ~/llama-swap-bench

./bench.sh <image> <label>      # startup + pp512/pp4096/tg128 + q8 prefill
                                # writes results/<label>.txt
./threeway.sh                   # MKL vs oneDNN vs TILE, per KV type   (~5 min)
./kvsweep2.sh                   # full KV x MKL matrix, 5 models       (~35 min)
./repeat.sh                     # same config x5, sanity-check noise   (~4 min)
./vram.sh                       # real VRAM per model, f16 vs q8_0     (~15 min)
```

### Read the contention column

`kvsweep2.sh` prints an `FFMPEG` column (count of ffmpeg processes holding the
B50) next to every measurement. **Baseline is 2** — frigate decoding. Any row
above that is contaminated; rerun it rather than averaging it in.

This matters. An early sweep produced numbers up to **78% below** a repeat of
the identical config, almost certainly a jellyfin transcode overlapping the run.
`repeat.sh` on a quiet box gives 0.4% spread, so large gaps are contention, not
hardware noise.

### Watch where the GPU is going

```bash
sudo xpu-smi dump --device 0 --metrics u,p,c --interval 2 --number 10
```

Columns to watch: `utilization.compute` (llama.cpp), `utilization.media`
(frigate/jellyfin), `power.draw` vs `power.limit`, `clocks.throttle.reason`.

Typical during a benchmark:

```
utilization.compute  99.99%
utilization.media    57-67%      <- frigate QSV decode
power.draw           69.9-70.2 W  vs limit 70 W
clocks.graphics      2250-2483 MHz  vs max 2600
clocks.throttle      PowerBurst
```

Video decode does **not** contend for the XMX/Xe cores — it uses the separate
media engine. It steals the **power budget**: the card is pinned at 70 W, so the
graphics clock is held ~13% below max. All numbers below are therefore ~13%
under what this GPU does with frigate paused. Rankings are unaffected.

---

## 3. Flash-attention kernel selection — the important finding

`ggml/src/ggml-sycl/fattn.cpp` dispatches by priority: **MKL (300) → oneDNN
(150) → TILE/VEC**. MKL is tested first, so whenever it is eligible it preempts
oneDNN, even where oneDNN is faster.

Measured, medgemma-4b, pp8192 (t/s):

| KV | MKL | oneDNN | TILE (no XMX) |
|---|---|---|---|
| f16 | 1722 | **2237** | 1326 |
| q8_0 | **1669** | 1309 | 1309 |

- oneDNN rejects any non-F16 K/V outright (`fattn-onednn.cpp:38`), so on q8_0 it
  falls through to TILE — note `1309.45` vs `1309.14`, the same kernel.
- On f16 both are eligible, and oneDNN beats MKL by 30% on this GPU.

### Rule

> **MKL on, unless the KV cache is F16 *and* the build has oneDNN.**

Turn MKL **on** for: quantized KV (q8_0/q4_0/q5_0/K-quants), BF16/F32 KV, builds
without oneDNN, or F16 where oneDNN is ineligible (no mask, non-F16 mask,
`mask->ne[2]/ne[3] != 1`).

Turn MKL **off** only for: F16 KV on a build that has oneDNN.

Irrelevant either way: `n_kv < 1024` or `Q->ne[1] < 32` — MKL never engages, so
**all token generation** is unaffected. This is a prompt-processing knob only.

Upstream-worthy: MKL preempting oneDNN on F16 costs 30% here. One env var
reproduces it.

### Update 2026-08-05 — the gap is gone

Upstream PR [#25852](https://github.com/ggml-org/llama.cpp/pull/25852) (merged
2026-08-04, `sycl: parallelize the non-contiguous concat kernel`, claimed
+9.4% pp2048) landed in the daily build alongside further oneAPI/AOT/driver
bumps (oneAPI 2026.1.2, NEO 26.27.39122.11, IGC 2.38.2 — see the
`llama-swap-sycl` repo's Dockerfile args on any build after 2026-08-04). The
two changes shipped in the same image, so the following can't be attributed to
#25852 alone — but the net effect on this GPU is that **the MKL/oneDNN
priority bug from §3 no longer matters in practice.**

Re-ran `kvsweep2.sh` in full (5 models × f16/q8_0 × MKL on/off × pp2048/pp8192,
20260805, image `llama-swap-sycl-bmg:latest` @ llama.cpp `6ea215d` / b10276,
0/0 or baseline-2/2 ffmpeg contention on every row — clean run). All 32
MKL-on-vs-off deltas were **within ±1.5%**, matching `repeat.sh`'s ~0.4% noise
floor:

| model | KV | pp | MKL=1 | MKL=0 | delta |
|---|---|---|---|---|---|
| medgemma-4b | f16 | pp8192 | 2249.98 | 2244.51 | +0.2% |
| medgemma-4b | q8_0 | pp8192 | 2259.84 | 2259.20 | +0.03% |
| gemma-4-e4b | f16 | pp8192 | 1593.02 | 1586.83 | +0.4% |
| gemma-4-e2b-obl | f16 | pp8192 | 2711.22 | 2749.21 | -1.4% |
| orpheus-3b | f16 | pp8192 | 2017.02 | 2014.56 | +0.1% |

(full 32-row output: `~/llama-swap-bench/kvsweep3-20260805.log`)

For reference, the same medgemma-4b/pp8192/f16/MKL=1 cell was **1722** on
2026-08-01 and is **2249.98** now — a +31% jump with MKL forced *on*, on top of
oneDNN barely moving (2237 → 2246). Whatever combination of #25852 and the
driver bump did this, it raised MKL to meet oneDNN rather than the reverse.

**The §3 rule is now moot on any build after ~2026-08-04.** The 7 models'
`GGML_SYCL_ENABLE_MKL_FA=0` overrides are harmless (MKL is no longer worse) but
also no longer necessary — see §8. Re-verify the gap is still closed before
trusting this on a different GPU or a future upstream bump; this was measured
on the Arc Pro B50 only, and the doc's own §3 caveat about hardware-specific
kernel ranking still applies.

---

## 4. Why performance changed

| date | event | effect |
|---|---|---|
| 2026-04-26 | oneAPI 2026.0.0 drops oneDNN from `deep-learning-essentials` | builds silently lose oneDNN; `find_package(DNNL)` only warns |
| 2026-07-15 | oneDNN XMX flash attention lands (#25222) | **no benefit** — no oneDNN in the image |
| 2026-07-31 | oneMKL XMX flash attention lands (#25025) | **benefit** — oneMKL was always linked (`find_package(MKL REQUIRED)`) |
| 2026-08-01 | this work: oneDNN restored + `MKL_FA=0` on f16 models | oneDNN path unlocked |
| 2026-08-04 | SYCL concat kernel parallelized (#25852) + oneAPI 2026.1.2/NEO 26.27/IGC 2.38.2 bump | MKL/oneDNN gap closes — see §3 update |

medgemma-4b pp8192 f16: TILE 1326 → MKL 1722 (+30%) → oneDNN 2237 (+69%).

The old `-2026` image had MKL but not oneDNN — verify any image with:

```bash
docker run --rm --entrypoint bash <image> -c 'ldd /app/libggml-sycl.so | grep -iE "mkl|dnnl"'
```

---

## 5. VRAM — f16 vs q8_0 (measured via xpu-smi, model-only, idle baseline removed)

| model | ctx | f16 MiB | q8_0 MiB | saved |
|---|---|---|---|---|
| gemma-4-e4b | 131072 | 5565 | 4565 | 1000 (18%) |
| gemma-4-e4b-obl | 131072 | 2883 | 2507 | 376 (13%) |
| medgemma-4b | 131072 | 6392 | 4950 | 1442 (23%) |
| medgemma-1.5-4b | 131072 | 6389 | 4947 | 1442 (23%) |
| orpheus-tts | 16384 | 4296 | 3431 | 865 (20%) |

Card is 16304 MiB; the largest f16 model peaks at 6967 MiB total (43%). **No
memory pressure**, so q8_0 buys headroom you do not need while costing 13-25%
prefill (gemma ~14%, medgemma ~24%, orpheus ~55%). Recommendation: stay on f16.

Revisit if you ever run two models concurrently or push a much larger model.

---

## 6. Build-time guardrails (in `Dockerfile`)

The build now **fails loudly** instead of degrading silently:

- oneDNN missing → error (this is what caught the 2026.0.0 regression)
- Level Zero loader missing → error
- `GGML_SYCL_DEVICE_ARCH` set but AOT not enabled → error
- AOT requested but `ocloc` absent → installs it, then errors if still missing

oneAPI 2026.x needs two things 2025.3.3 bundled: `intel-oneapi-dnnl-devel`
(build) + `intel-oneapi-runtime-dnnl` + ldconfig entry (runtime), and
`intel-ocloc` for AOT. Both are conditional on the base image lacking them, so
they are no-ops on 2025.3.3.

---

## 7. Known traps

- **`pkill -f <pattern>` over SSH kills your own session** if the pattern appears
  in the command line. Same for `pgrep` — it self-matches and reports phantom
  "still running" processes. Use `pgrep -af X | grep -v pgrep`.
- **`SYCL_CACHE_PERSISTENT=1` segfaults llama-server on Battlemage** during model
  load. Verified on oneAPI 2025.3.3 and 2026.0.0, NEO 25.40 and 26.27, any
  `SYCL_CACHE_DIR`. Do not re-add it. AOT is the JIT fix instead.
- **Comparing f16-KV against q8_0-KV does not test MKL** — both use it. Toggle
  `GGML_SYCL_ENABLE_MKL_FA` to A/B the kernel.
- `docs/backend/SYCL.md` is stale on MKL FA: it claims quantized KV and
  `batch >= 1024` are required. The merged code gates only on GQA ratio, head
  dim, and `n_kv >= 1024`.
- llama-server does not print KV cache sizes at default verbosity; use `vram.sh`
  (xpu-smi) rather than parsing logs.

---

## 8. Open items

- **`GGML_SYCL_ENABLE_MKL_FA=0` on the 7 f16 models is now dead weight** (§3
  update, 2026-08-05) — the priority bug it worked around no longer measures
  as a regression on this build. Safe to remove for config simplicity, not
  urgent since it's harmless. Re-benchmark before removing if a lot of time
  has passed, in case a future upstream change reopens the gap.
- q8_0 rollout: **not applied**, and recommended against per §5. If you want it
  anyway, add `--cache-type-k q8_0 --cache-type-v q8_0` to those 7 models **and
  remove their `GGML_SYCL_ENABLE_MKL_FA=0`** — quantized KV needs MKL on.
- `orpheus-tts` got `MKL_FA=0` by pattern, not measurement. It is short-context,
  so `n_kv` rarely reaches the 1024 gate; near-inert either way.
- `llama-swap-sycl-2026` and `llama-swap-sycl-bmg` now share a base and differ
  only by AOT. Dropping `-2026` would cut ~25 min per nightly.
- Numbers are Battlemage-only. Which of two *eligible* kernels is faster is a
  hardware property; the eligibility gates are not.
