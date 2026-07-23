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
        -ldflags="-X main.commit=$(git rev-parse --short HEAD) -X main.version=custom_$(git rev-parse --short HEAD) -X main.date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        -o /src/llama-swap .

# ─────────────────────────────────────────────────────────────────────────
# Build stage: compile llama.cpp for SYCL from source
# ─────────────────────────────────────────────────────────────────────────
FROM intel/deep-learning-essentials:${ONEAPI_VERSION} AS build

ARG LLAMACPP_REF=master
ARG GGML_SYCL_F16=ON
ARG LEVEL_ZERO_VERSION=1.28.2
ARG LEVEL_ZERO_UBUNTU_VERSION=u24.04

RUN apt-get update && \
    apt-get install -y git libssl-dev wget ca-certificates && \
    cd /tmp && \
    wget -q "https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/level-zero_${LEVEL_ZERO_VERSION}%2B${LEVEL_ZERO_UBUNTU_VERSION}_amd64.deb" -O level-zero.deb && \
    wget -q "https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/level-zero-devel_${LEVEL_ZERO_VERSION}%2B${LEVEL_ZERO_UBUNTU_VERSION}_amd64.deb" -O level-zero-devel.deb && \
    apt-get -o Dpkg::Options::="--force-overwrite" install -y ./level-zero.deb ./level-zero-devel.deb && \
    rm -f /tmp/level-zero.deb /tmp/level-zero-devel.deb

WORKDIR /app

# Shallow-clone exactly the requested ref (tag / branch / sha).
RUN git init -q && \
    git remote add origin https://github.com/ggml-org/llama.cpp.git && \
    git fetch --depth=1 origin "${LLAMACPP_REF}" && \
    git checkout -q FETCH_HEAD && \
    git rev-parse HEAD > /app/LLAMACPP_GIT_SHA

# Same flags as upstream's server image: dynamic backends (BACKEND_DL) so the
# SYCL + all CPU variants are loadable .so files alongside the binary.
RUN if [ "${GGML_SYCL_F16}" = "ON" ]; then \
        echo "GGML_SYCL_F16 is set" && export OPT_SYCL_F16="-DGGML_SYCL_F16=ON"; \
    fi && \
    cmake -B build \
        -DGGML_NATIVE=OFF \
        -DGGML_SYCL=ON \
        -DCMAKE_C_COMPILER=icx \
        -DCMAKE_CXX_COMPILER=icpx \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        ${OPT_SYCL_F16} && \
    cmake --build build --config Release -j"$(nproc)"

# Collect the binary + every shared lib it dlopens into one dir.
RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \; && \
    mkdir -p /app/full && \
    cp build/bin/llama-server /app/full/

# ─────────────────────────────────────────────────────────────────────────
# Runtime base: oneAPI + Intel GPU compute runtime (Level Zero / OpenCL / IGC)
# ─────────────────────────────────────────────────────────────────────────
FROM intel/deep-learning-essentials:${ONEAPI_VERSION} AS base

ARG IGC_VERSION=v2.20.5
ARG IGC_VERSION_FULL=2_2.20.5+19972
ARG COMPUTE_RUNTIME_VERSION=25.40.35563.10
ARG COMPUTE_RUNTIME_VERSION_FULL=25.40.35563.10-0
ARG IGDGMM_VERSION=22.8.2

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

# ─────────────────────────────────────────────────────────────────────────
# Final image (upstream): llama-server + libs + the mostlygeek/llama-swap
# release binary
# ─────────────────────────────────────────────────────────────────────────
FROM base AS server

ARG TARGETARCH=amd64
ARG LS_VER=222
ARG LLAMACPP_REF=master
ARG BUILD_DATE=N/A

LABEL org.opencontainers.image.title="llama-swap-sycl" \
      org.opencontainers.image.description="llama-swap + llama.cpp (Intel SYCL) built from source" \
      org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \
      io.llamacpp.ref="${LLAMACPP_REF}" \
      io.llamaswap.version="${LS_VER}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV LLAMA_ARG_HOST=0.0.0.0
ENV PATH="/app:${PATH}"

# llama-server + its dlopen'd backends share /app; the binary's $ORIGIN rpath
# finds them, matching the upstream server image layout.
COPY --from=build /app/lib/ /app
COPY --from=build /app/full/llama-server /app
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

LABEL org.opencontainers.image.title="llama-swap-sycl" \
      org.opencontainers.image.description="llama-swap + llama.cpp (Intel SYCL) built from source" \
      org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \
      io.llamacpp.ref="${LLAMACPP_REF}" \
      io.llamaswap.repo="${LS_REPO}" \
      io.llamaswap.ref="${LS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV LLAMA_ARG_HOST=0.0.0.0
ENV PATH="/app:${PATH}"

COPY --from=build /app/lib/ /app
COPY --from=build /app/full/llama-server /app
COPY --from=build /app/LLAMACPP_GIT_SHA /app/LLAMACPP_GIT_SHA

# Drop in the llama-swap binary built from our fork/branch.
COPY --from=ls-build /src/llama-swap /app/llama-swap
COPY --from=ls-build /src/LLAMASWAP_GIT_SHA /app/LLAMASWAP_GIT_SHA

COPY config.example.yaml /app/config.yaml

WORKDIR /app

HEALTHCHECK CMD curl -f http://localhost:8080/ || exit 1
ENTRYPOINT [ "/app/llama-swap", "-config", "/app/config.yaml", "-listen", "0.0.0.0:8080" ]
