#!/usr/bin/env bash
# Interview Transcriber v5 — Setup Script
# Ubuntu 24.04 + NVIDIA GPU
set -euo pipefail

echo ""
echo "=========================================="
echo "  Interview Transcriber v5 — Setup"
echo "=========================================="
echo ""

# ── 1. Python version check ─────────────────────────────────────────
PYTHON=$(command -v python3 || true)
if [ -z "$PYTHON" ]; then
  echo "[ERROR] python3 not found. Install with: sudo apt install python3"
  exit 1
fi
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  echo "[ERROR] Python 3.10+ required (found $PY_VER)"
  exit 1
fi
echo "[OK] Python $PY_VER"

# ── 2. Check for pip ────────────────────────────────────────────────
if ! $PYTHON -m pip --version &>/dev/null; then
  echo "[ERROR] pip not found. Install with: sudo apt install python3-pip"
  exit 1
fi

# ── 3. Create .venv (FastAPI + faster-whisper, no PyTorch) ──────────
echo ""
echo "Creating .venv (FastAPI + faster-whisper)..."
$PYTHON -m venv .venv
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install \
  fastapi \
  "uvicorn[standard]" \
  python-multipart \
  faster-whisper \
  soundfile \
  ffmpeg-python \
  --quiet
echo "[OK] .venv created"

# ── 4. Create .venv-pyannote (PyTorch + pyannote) ───────────────────
echo ""
echo "Creating .venv-pyannote (PyTorch + pyannote.audio)..."
echo "  This downloads ~3 GB of PyTorch CUDA wheels — may take a few minutes."
$PYTHON -m venv .venv-pyannote
.venv-pyannote/bin/pip install --upgrade pip --quiet
.venv-pyannote/bin/pip install \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 \
  --quiet
.venv-pyannote/bin/pip install \
  "pyannote.audio==3.3.2" \
  "huggingface_hub==0.25.2" \
  --quiet
echo "[OK] .venv-pyannote created"

# ── 5. HuggingFace token ────────────────────────────────────────────
echo ""
echo "HuggingFace token (required for pyannote diarization pipeline)."
echo "Get yours at: https://huggingface.co/settings/tokens"
echo "You must also accept the model terms at:"
echo "  https://huggingface.co/pyannote/speaker-diarization-3.1"
echo ""
if [ -f ".hf_token" ] && [ -s ".hf_token" ]; then
  echo "[SKIP] .hf_token already exists — remove it to re-enter."
else
  read -rp "Paste your HuggingFace token (hf_...): " HF_TOKEN_INPUT
  if [ -n "$HF_TOKEN_INPUT" ]; then
    echo "$HF_TOKEN_INPUT" > .hf_token
    chmod 600 .hf_token
    echo "[OK] Token saved to .hf_token"
  else
    echo "[WARN] No token entered — diarization will fail at runtime."
    touch .hf_token
  fi
fi

# ── 6. Create outputs/ directory ────────────────────────────────────
mkdir -p outputs uploads
echo "[OK] outputs/ and uploads/ directories created"

# ── 7. Write .env ────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  cat > .env <<'EOF'
HF_TOKEN_FILE=.hf_token
HOST=0.0.0.0
PORT=8765
EOF
  echo "[OK] .env created"
else
  echo "[SKIP] .env already exists"
fi

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "Start the server:"
echo ""
echo "  source .venv/bin/activate"
echo "  export \$(grep -v '^#' .env | xargs)"
echo "  python app.py"
echo ""
echo "Then open http://localhost:8765 (or your Tailscale IP)."
echo ""
