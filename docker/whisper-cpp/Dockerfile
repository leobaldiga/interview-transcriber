# whisper.cpp server with CUDA, built from source.
#
# Targets storm1: RTX 3060 => compute capability 8.6. Override CUDA_ARCH for
# other cards (4090 = 89, A100 = 80).
#
# No model is downloaded here — mount a host models directory read-only and
# point WHISPER_MODEL at a file inside it. See docker/whisper-cpp/README.md.

ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

ARG WHISPER_CPP_REF=master
ARG CUDA_ARCH=86

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch ${WHISPER_CPP_REF} https://github.com/ggml-org/whisper.cpp.git .

# ggml-cuda calls CUDA Driver API symbols (cuMemCreate & friends), but the devel
# image ships no real libcuda.so.1 — the driver is injected at runtime by the
# nvidia container toolkit. Link against the stub, which is named libcuda.so and
# so does not satisfy the libcuda.so.1 SONAME without this symlink.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

# GGML_CUDA is the current flag; WHISPER_CUBLAS is a hard build error upstream.
# -rpath-link (not -rpath) keeps the stub path out of the produced binaries.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DWHISPER_BUILD_SERVER=ON \
        -DWHISPER_BUILD_TESTS=OFF \
        -DWHISPER_BUILD_EXAMPLES=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    && cmake --build build --config Release -j "$(nproc)" --target whisper-server \
    && mkdir -p /opt/whisper/bin /opt/whisper/lib \
    && cp build/bin/whisper-server /opt/whisper/bin/ \
    && find build -name '*.so*' ! -name 'libcuda.so*' -exec cp -P {} /opt/whisper/lib/ \;


FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# ffmpeg is required for whisper-server --convert (non-WAV uploads).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/whisper/bin/whisper-server /usr/local/bin/whisper-server
COPY --from=build /opt/whisper/lib/ /usr/local/lib/
RUN ldconfig

ENV WHISPER_MODEL=/models/ggml-large-v3-turbo.bin \
    WHISPER_PORT=8081 \
    WHISPER_THREADS=4 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${WHISPER_PORT}/health" || exit 1

# --convert lets the server accept m4a/mp3/webm by shelling out to ffmpeg.
CMD ["sh", "-c", "exec whisper-server \
     --model \"$WHISPER_MODEL\" \
     --host 0.0.0.0 \
     --port \"$WHISPER_PORT\" \
     --threads \"$WHISPER_THREADS\" \
     --convert"]
