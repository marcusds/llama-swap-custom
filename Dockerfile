# llama-swap + llama.cpp (Intel SYCL) built from source.
#
# This mirrors llama.cpp's official .devops/intel.Dockerfile "server" target
# (build from source with the oneAPI icx/icpx compilers, GGML_SYCL=ON), then
# layers the llama-swap proxy on top so the final image is a drop-in
# replacement for ghcr.io/mostlygeek/llama-swap:intel that *we* control.
#
# Build args of interest:
#   LLAMACPP_REF   git ref to build (tag like b9495, a commit sha, or a branch)
#   LS_REPO        llama-swap git repo to build from
#   LS_REF         llama-swap git ref to build (branch, tag, or sha)
#   GGML_SYCL_F16  ON/OFF  -- enable half-precision SYCL kernels
#   ONEAPI_VERSION intel/deep-learning-essentials base image tag
#   GGML_SYCL_DEVICE_ARCH
#                  empty (default) -> portable JIT build, runs on any Intel GPU.
#                  Set to an ocloc device name (bmg_g21, dg2-g10, lnl_m, ...) to
#                  compile ahead-of-time via spir64_gen. Removes SPIR-V JIT at
#                  every llama-server start -- which llama-swap pays on *every*
#                  model swap -- at the cost of an image that only runs on that
#                  GPU architecture.
#   COMPUTE_RUNTIME_VERSION / IGC_VERSION / IGDGMM_VERSION
#                  Intel GPU userspace stack. Defaults to the current NEO 26.18 /
#                  IGC 2.34.4 set. MULTI-GPU hosts hit a known bug there
#                  (ggml-org/llama.cpp#21747, intel/compute-runtime#921) and
#                  should override back to the 25.40 set -- see below.

ARG ONEAPI_VERSION=2025.3.3-0-devel-ubuntu24.04
ARG LS_REPO=https://github.com/marcusds/llama-swap.git
ARG LS_REF=memory-budget

# ─────────────────────────────────────────────────────────────────────────
# Build stage: build llama-swap from source (our fork/branch, not a release)
# ─────────────────────────────────────────────────────────────────────────
FROM golang:1.26-bookworm AS ls-build

ARG LS_REPO
ARG LS_REF
ARG TARGETARCH=amd64

RUN apt-get update && \
    apt-get install -y git curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git init -q && \
    git remote add origin "${LS_REPO}" && \
    git fetch --depth=1 origin "${LS_REF}" && \
    git checkout -q FETCH_HEAD && \
    git rev-parse HEAD > /src/LLAMASWAP_GIT_SHA

RUN cd ui-svelte && npm ci && npm run build

RUN GOOS=linux GOARCH=${TARGETARCH} go build \
        -tags embed_ui \
        -ldflags="-X main.commit=$(git rev-parse --short HEAD) -X main.version=custom_$(git rev-parse --short HEAD) -X main.date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        -o /src/llama-swap .

# ─────────────────────────────────────────────────────────────────────────
# Build stage: compile llama.cpp for SYCL from source
# ─────────────────────────────────────────────────────────────────────────
FROM intel/deep-learning-essentials:${ONEAPI_VERSION} AS build

ARG LLAMACPP_REF=master
ARG GGML_SYCL_F16=ON
ARG GGML_SYCL_DEVICE_ARCH=""
ARG LEVEL_ZERO_VERSION=1.28.2
ARG LEVEL_ZERO_UBUNTU_VERSION=u24.04

# bash (not dash) so `set -o pipefail` works: the cmake output below is piped
# through tee, and a config failure must not be swallowed by the pipe.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y git libssl-dev wget ca-certificates && \
    cd /tmp && \
    wget -q "https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/level-zero_${LEVEL_ZERO_VERSION}%2B${LEVEL_ZERO_UBUNTU_VERSION}_amd64.deb" -O level-zero.deb && \
    wget -q "https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/level-zero-devel_${LEVEL_ZERO_VERSION}%2B${LEVEL_ZERO_UBUNTU_VERSION}_amd64.deb" -O level-zero-devel.deb && \
    apt-get -o Dpkg::Options::="--force-overwrite" install -y ./level-zero.deb ./level-zero-devel.deb && \
    rm -f /tmp/level-zero.deb /tmp/level-zero-devel.deb

# Intel dropped oneDNN from the deep-learning-essentials image in 2026.0.0
# (2025.3.3 still bundles it). find_package(DNNL) then fails, ggml-sycl quietly
# loses the fused SDPA path, and nothing errors -- measured on an Arc Pro B50,
# an image without oneDNN did pp512 at 897 t/s vs 944 t/s with it. Reinstall
# from the oneAPI apt repo, which is already configured in the base image.
RUN if [ -z "$(find /opt/intel -name dnnl-config.cmake -print -quit)" ]; then \
        echo "oneDNN absent from base image -- installing intel-oneapi-dnnl-devel" && \
        apt-get update && \
        apt-get install -y intel-oneapi-dnnl-devel && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        echo "oneDNN bundled with base image"; \
    fi

# AOT (-fsycl-targets=spir64_gen) shells out to ocloc, Intel's offline GPU
# compiler. The 2025.3.x oneAPI images bundle it at /usr/bin/ocloc; 2026.x does
# not, and icpx then fails the link of libggml-sycl.so with a bare
# "gen compiler command failed" after only a -Waot-tool-not-found warning.
# Install the same compute-runtime the final image uses, so the ISA we generate
# matches the driver that will run it. JIT builds do not need ocloc.
ARG IGC_VERSION=v2.38.2
ARG IGC_VERSION_FULL=2_2.38.2+22051
ARG COMPUTE_RUNTIME_VERSION=26.27.39122.11
ARG COMPUTE_RUNTIME_VERSION_FULL=26.27.39122.11-0
ARG IGDGMM_VERSION=22.10.0
RUN if [ -n "${GGML_SYCL_DEVICE_ARCH}" ] && ! command -v ocloc >/dev/null; then \
        echo "AOT requested but ocloc missing -- installing compute-runtime ${COMPUTE_RUNTIME_VERSION}" && \
        mkdir -p /tmp/ocloc && cd /tmp/ocloc && \
        wget -q "https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_VERSION}/intel-igc-core-${IGC_VERSION_FULL}_amd64.deb" && \
        wget -q "https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_VERSION}/intel-igc-opencl-${IGC_VERSION_FULL}_amd64.deb" && \
        wget -q "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb" && \
        wget -q "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_${IGDGMM_VERSION}_amd64.deb" && \
        dpkg --install *.deb && \
        cd / && rm -rf /tmp/ocloc; \
    fi && \
    if [ -n "${GGML_SYCL_DEVICE_ARCH}" ] && ! command -v ocloc >/dev/null; then \
        echo "ERROR: AOT requested but ocloc is still unavailable"; exit 1; \
    fi

WORKDIR /app

# Shallow-clone exactly the requested ref (tag / branch / sha).
RUN git init -q && \
    git remote add origin https://github.com/ggml-org/llama.cpp.git && \
    git fetch --depth=1 origin "${LLAMACPP_REF}" && \
    git checkout -q FETCH_HEAD && \
    git rev-parse HEAD > /app/LLAMACPP_GIT_SHA

# Same flags as upstream's server image: dynamic backends (BACKEND_DL) so the
# SYCL + all CPU variants are loadable .so files alongside the binary.
# -DGGML_SYCL_DNN=ON is the cmake default, but a failed find_package(DNNL) only
# *warns* and builds on without oneDNN -- silently dropping the fused SDPA /
# flash-attention path. Same for AOT. Assert both landed instead of shipping a
# quietly slower image.
RUN set -e; \
    OPT=(); \
    if [ "${GGML_SYCL_F16}" = "ON" ]; then \
        echo "GGML_SYCL_F16 is set"; OPT+=(-DGGML_SYCL_F16=ON); \
    fi; \
    if [ -n "${GGML_SYCL_DEVICE_ARCH}" ]; then \
        echo "AOT build targeting ${GGML_SYCL_DEVICE_ARCH}"; \
        OPT+=("-DGGML_SYCL_DEVICE_ARCH=${GGML_SYCL_DEVICE_ARCH}"); \
    fi; \
    DNNL_CFG="$(find /opt/intel -name dnnl-config.cmake -print -quit)"; \
    if [ -n "$DNNL_CFG" ]; then \
        OPT+=("-DDNNL_DIR=$(dirname "$DNNL_CFG")"); \
    fi; \
    cmake -B build \
        -DGGML_NATIVE=OFF \
        -DGGML_SYCL=ON \
        -DCMAKE_C_COMPILER=icx \
        -DCMAKE_CXX_COMPILER=icpx \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_SYCL_DNN=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        "${OPT[@]}" 2>&1 | tee /tmp/cmake-config.log; \
    if ! grep -q "Found oneDNN" /tmp/cmake-config.log; then \
        echo "ERROR: oneDNN not found -- the fused SDPA path would be disabled"; exit 1; \
    fi; \
    if ! grep -q "Level Zero loader found" /tmp/cmake-config.log; then \
        echo "ERROR: Level Zero loader not found -- direct L0 allocation disabled"; exit 1; \
    fi; \
    if [ -n "${GGML_SYCL_DEVICE_ARCH}" ] && ! grep -q "AOT via spir64_gen" /tmp/cmake-config.log; then \
        echo "ERROR: GGML_SYCL_DEVICE_ARCH set but AOT was not enabled"; exit 1; \
    fi; \
    cmake --build build --config Release -j"$(nproc)"

# Collect the binary + every shared lib it dlopens into one dir.
# llama-bench ships too: without it there is no way to benchmark the exact
# build that is running in the image.
RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \; && \
    mkdir -p /app/full && \
    cp build/bin/llama-server /app/full/ && \
    cp build/bin/llama-bench /app/full/

# ─────────────────────────────────────────────────────────────────────────
# Runtime base: oneAPI + Intel GPU compute runtime (Level Zero / OpenCL / IGC)
# ─────────────────────────────────────────────────────────────────────────
FROM intel/deep-learning-essentials:${ONEAPI_VERSION} AS base

# NEO 26.27 + its matching IGC 2.38.2 / gmmlib 22.10.0 (pairing per the NEO
# 26.27 release notes). Newer than upstream's 26.18 default, and deliberately
# so: the multi-GPU Level Zero context regression (intel/compute-runtime#921,
# ggml-org/llama.cpp#21747) was only merged 2026-06-04, *after* 26.18 was cut
# on 2026-05-12. 26.27 (2026-07-21) is the first stack here that should carry
# the fix -- though Intel publishes no per-release issue mapping, so this is
# inferred from dates, not confirmed by a maintainer statement.
#
# If a multi-GPU host still misbehaves, fall back by overriding all five:
#   IGC_VERSION=v2.20.5 IGC_VERSION_FULL=2_2.20.5+19972
#   COMPUTE_RUNTIME_VERSION=25.40.35563.10
#   COMPUTE_RUNTIME_VERSION_FULL=25.40.35563.10-0 IGDGMM_VERSION=22.8.2
ARG IGC_VERSION=v2.38.2
ARG IGC_VERSION_FULL=2_2.38.2+22051
ARG COMPUTE_RUNTIME_VERSION=26.27.39122.11
ARG COMPUTE_RUNTIME_VERSION_FULL=26.27.39122.11-0
ARG IGDGMM_VERSION=22.10.0

RUN mkdir /tmp/neo && cd /tmp/neo && \
    wget -q https://github.com/intel/intel-graphics-compiler/releases/download/$IGC_VERSION/intel-igc-core-${IGC_VERSION_FULL}_amd64.deb && \
    wget -q https://github.com/intel/intel-graphics-compiler/releases/download/$IGC_VERSION/intel-igc-opencl-${IGC_VERSION_FULL}_amd64.deb && \
    wget -q https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/intel-ocloc_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb && \
    wget -q https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb && \
    wget -q https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/libigdgmm12_${IGDGMM_VERSION}_amd64.deb && \
    wget -q https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb && \
    dpkg --install *.deb && \
    rm -rf /tmp/neo

RUN apt-get update && \
    apt-get install -y libgomp1 curl && \
    apt autoremove -y && apt clean -y && \
    rm -rf /tmp/* /var/tmp/* && \
    find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete && \
    find /var/cache -type f -delete

# Runtime half of the oneDNN restoration above. libggml-sycl.so has no $ORIGIN
# in its RUNPATH and resolves libdnnl via the loader path, but on 2026.x the
# oneDNN lib dir is not on LD_LIBRARY_PATH even once installed -- so register it
# with ldconfig. Asserted, because a missing libdnnl at runtime would only show
# up as a backend that fails to load.
RUN if [ -z "$(find /opt/intel -name 'libdnnl.so*' -print -quit)" ]; then \
        apt-get update && \
        apt-get install -y intel-oneapi-runtime-dnnl && \
        apt clean -y && rm -rf /var/lib/apt/lists/*; \
    fi && \
    dirname "$(find /opt/intel -name 'libdnnl.so*' -print -quit)" > /etc/ld.so.conf.d/onednn.conf && \
    ldconfig && \
    if ! ldconfig -p | grep -q libdnnl; then \
        echo "ERROR: libdnnl is not resolvable at runtime"; exit 1; \
    fi

# ── Runtime tuning defaults, applied to every final image ────────────────
#
# ZES_ENABLE_SYSMAN  needed for sycl::aspect::ext_intel_free_memory, i.e. real
#               free-VRAM reporting -- required for --split-mode layer.
# UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS  allows single allocations >4 GiB.
#               AOT builds drop the -ze-intel-greater-than-4GB-buffer-required
#               link flag, so this is mandatory there and harmless otherwise.
#
# Deliberately NOT set:
#   SYCL_CACHE_PERSISTENT=1  Would persist JIT'd kernels across the
#     llama-server restart that llama-swap performs on every model swap --
#     but it SIGSEGVs during model load on Arc Pro B50 (Battlemage). Verified
#     crashing on both oneAPI 2025.3.3 and 2026.0.0, on both the NEO 25.40 and
#     26.27 GPU stacks, and independent of SYCL_CACHE_DIR / SYCL_CACHE_MAX_SIZE.
#     Use an AOT image (GGML_SYCL_DEVICE_ARCH) to remove JIT instead.
#   GGML_SYCL_ENABLE_GRAPH=1  upstream: still in development, no perf win yet.
#   SYCL_PROGRAM_COMPILE_OPTIONS=-cl-fp32-correctly-rounded-divide-sqrt
#     trades speed for precision.
ENV ZES_ENABLE_SYSMAN=1 \
    UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1

# ─────────────────────────────────────────────────────────────────────────
# Final image (upstream): llama-server + libs + the mostlygeek/llama-swap
# release binary
# ─────────────────────────────────────────────────────────────────────────
FROM base AS server

ARG TARGETARCH=amd64
ARG LS_VER=222
ARG LLAMACPP_REF=master
ARG BUILD_DATE=N/A
ARG GGML_SYCL_DEVICE_ARCH=""

LABEL org.opencontainers.image.title="llama-swap-sycl" \
      org.opencontainers.image.description="llama-swap + llama.cpp (Intel SYCL) built from source" \
      org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \
      io.llamacpp.ref="${LLAMACPP_REF}" \
      io.sycl.device_arch="${GGML_SYCL_DEVICE_ARCH}" \
      io.llamaswap.version="${LS_VER}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV LLAMA_ARG_HOST=0.0.0.0
ENV PATH="/app:${PATH}"

# llama-server + its dlopen'd backends share /app; the binary's $ORIGIN rpath
# finds them, matching the upstream server image layout.
COPY --from=build /app/lib/ /app
COPY --from=build /app/full/llama-server /app
COPY --from=build /app/full/llama-bench /app
COPY --from=build /app/LLAMACPP_GIT_SHA /app/LLAMACPP_GIT_SHA

# Drop in the llama-swap release binary.
RUN curl -fsSL -o /tmp/ls.tar.gz \
        "https://github.com/mostlygeek/llama-swap/releases/download/v${LS_VER}/llama-swap_${LS_VER}_linux_${TARGETARCH}.tar.gz" && \
    tar -zxf /tmp/ls.tar.gz -C /app llama-swap && \
    rm /tmp/ls.tar.gz

COPY config.example.yaml /app/config.yaml

WORKDIR /app

HEALTHCHECK CMD curl -f http://localhost:8080/ || exit 1
ENTRYPOINT [ "/app/llama-swap", "-config", "/app/config.yaml", "-listen", "0.0.0.0:8080" ]

# ─────────────────────────────────────────────────────────────────────────
# Final image (fork): llama-server + libs + llama-swap built from our
# fork/branch instead of an upstream release
# ─────────────────────────────────────────────────────────────────────────
FROM base AS server-fork

ARG TARGETARCH=amd64
ARG LS_REPO=https://github.com/marcusds/llama-swap.git
ARG LS_REF=memory-budget
ARG LLAMACPP_REF=master
ARG BUILD_DATE=N/A
ARG GGML_SYCL_DEVICE_ARCH=""

LABEL org.opencontainers.image.title="llama-swap-sycl" \
      org.opencontainers.image.description="llama-swap + llama.cpp (Intel SYCL) built from source" \
      org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \
      io.llamacpp.ref="${LLAMACPP_REF}" \
      io.sycl.device_arch="${GGML_SYCL_DEVICE_ARCH}" \
      io.llamaswap.repo="${LS_REPO}" \
      io.llamaswap.ref="${LS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV LLAMA_ARG_HOST=0.0.0.0
ENV PATH="/app:${PATH}"

COPY --from=build /app/lib/ /app
COPY --from=build /app/full/llama-server /app
COPY --from=build /app/full/llama-bench /app
COPY --from=build /app/LLAMACPP_GIT_SHA /app/LLAMACPP_GIT_SHA

# Drop in the llama-swap binary built from our fork/branch.
COPY --from=ls-build /src/llama-swap /app/llama-swap
COPY --from=ls-build /src/LLAMASWAP_GIT_SHA /app/LLAMASWAP_GIT_SHA

COPY config.example.yaml /app/config.yaml

WORKDIR /app

HEALTHCHECK CMD curl -f http://localhost:8080/ || exit 1
ENTRYPOINT [ "/app/llama-swap", "-config", "/app/config.yaml", "-listen", "0.0.0.0:8080" ]
