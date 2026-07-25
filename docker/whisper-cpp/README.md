# whisper.cpp CUDA server

Builds `whisper-server` from source with `GGML_CUDA=ON` and serves it on
`127.0.0.1:8081`. No model is baked into the image.

## 1. Fetch a ggml model yourself

Models are not downloaded during the build. Pick a directory on the host and
grab a GGML/GGUF-format whisper model into it:

```bash
mkdir -p ~/models/whisper
cd ~/models/whisper

# Option A — the download helper from the whisper.cpp repo
curl -L -o download-ggml-model.sh \
  https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/models/download-ggml-model.sh
bash download-ggml-model.sh large-v3-turbo .

# Option B — straight from Hugging Face
curl -L -o ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

A quantised model (`ggml-large-v3-turbo-q5_0.bin`) fits an RTX 3060's 12 GB with
plenty of headroom for pyannote to run alongside it.

## 2. Build and run

```bash
export WHISPER_MODELS_DIR=~/models/whisper
export WHISPER_GGML_MODEL=ggml-large-v3-turbo.bin
export CUDA_ARCH=86            # RTX 3060

docker compose build whisper-cpp
docker compose up -d whisper-cpp
docker compose logs -f whisper-cpp
```

Without compose:

```bash
docker build --build-arg CUDA_ARCH=86 -t whisper-cpp:cuda docker/whisper-cpp
docker run -d --name whisper-cpp --gpus all \
  -p 127.0.0.1:8081:8081 \
  -v ~/models/whisper:/models:ro \
  -e WHISPER_MODEL=/models/ggml-large-v3-turbo.bin \
  whisper-cpp:cuda
```

## 3. Verify

```bash
curl -s http://127.0.0.1:8081/health          # {"status":"ok"}
nvidia-smi                                     # whisper-server should hold VRAM
```

`{"status":"loading model"}` with HTTP 503 means the model is still being read;
a connection refused means the container is not up.

## Pinning

`WHISPER_CPP_REF` defaults to `master`. Pin it to a release tag once a build is
known good, so a rebuild cannot silently change ASR behaviour:

```bash
WHISPER_CPP_REF=v1.7.4 docker compose build whisper-cpp
```

## Notes

- `--convert` is enabled so the server accepts m4a/mp3/webm by shelling out to
  ffmpeg. Without it, non-WAV uploads return HTTP 500.
- Upstream serialises every `/inference` call behind a global mutex — the server
  transcribes one file at a time. The app's job queue already enforces this.
