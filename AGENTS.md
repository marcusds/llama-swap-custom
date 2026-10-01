# Agent notes

## Keep the toolchain current

Whenever you touch this repo, check the pinned stack against what is published
and bump anything stale. Upstream llama.cpp lags; this fleet does not.

| Component | Pinned in | Check |
|---|---|---|
| oneAPI | `Dockerfile` `ONEAPI_VERSION`, `build.yml` fleet matrix + `oneapi_version` defaults | Docker Hub `intel/deep-learning-essentials` tags (`YYYY.M.P-devel-ubuntu24.04`) |
| CUDA | `Dockerfile.cuda` `CUDA_VERSION` (3 places), `build.yml` fleet matrix | Docker Hub `nvidia/cuda` tags (`X.Y.Z-devel-ubuntu24.04`, amd64 + arm64) |
| NEO / IGC / gmmlib | `Dockerfile` (`COMPUTE_RUNTIME_VERSION`, `IGC_VERSION*`, `IGDGMM_VERSION`), `build.yml` non-legacy `igc`/`neo` outputs | `intel/compute-runtime` releases; take the IGC + gmmlib **that release's notes pair it with**, never a newer IGC on its own |
| Level Zero | `Dockerfile` `LEVEL_ZERO_VERSION` | `oneapi-src/level-zero` releases (`libze1`/`libze-dev` `+u24.04` debs) |
| Go | `golang:X-bookworm` in both Dockerfiles | Docker Hub `golang` |
| Node | `setup_XX.x` in both Dockerfiles | current Node LTS |
| llama-swap (local default) | `ARG LS_VER` in both Dockerfiles | `mostlygeek/llama-swap` latest release (CI resolves this itself) |
| GitHub Actions | `uses:` in `.github/workflows/*.yml` | each action's latest major; read its breaking-change notes |

Rules:

- `llama-swap-sycl-multigpu` stays on oneAPI 2025.3.3 + NEO 25.40 / IGC 2.20.5
  on purpose (multi-GPU fallback). Do not bump it.
- Stay on Ubuntu 24.04 until Level Zero publishes `u26.04` debs or upstream
  llama.cpp's `.devops/` images move.
- After a CUDA bump, check the bundled CCCL
  (`/usr/local/cuda/targets/*/include/cccl/cuda/std/__cccl/version.h`). CUB
  DeviceTopK needs CCCL >= 3.4.3; if the CTK bundles less, add
  `-DGGML_CUDA_CCCL_VERSION=v3.4.3` to the cmake call.
- Also scan upstream llama.cpp for build changes since the last bump:
  `git log -- CMakeLists.txt ggml/CMakeLists.txt ggml/src/ggml-{sycl,cuda,cpu}/CMakeLists.txt .devops/ docs/build.md docs/backend/SYCL.md`.
- Dated benchmark records in `docs/SYCL-BENCHMARKING.md` and
  `bench-baseline-*.md` describe what was measured; never rewrite their
  versions during a bump.
- A push to `main` that touches a Dockerfile or `build.yml` rebuilds the whole
  fleet and cancels any in-progress build of the same image
  (`cancel-in-progress`). Check `gh run list` before pushing.
