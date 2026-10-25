#!/usr/bin/env bash
set -e

echo "=========================================="
echo "⚡ Initializing LTX-2.5 Production Boot..."
echo "=========================================="

# 1. System packages check (Agar already installed hain toh apt skip karega)
if ! command -v ffmpeg &> /dev/null || ! command -v tmux &> /dev/null; then
  echo "📦 Installing system packages (ffmpeg, tmux, aria2)..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq tmux aria2 ffmpeg &
  APT_PID=$!
else
  echo "✅ System packages already present."
  APT_PID=""
fi

# 2. Repo Clone Check
if [ ! -d "LTX-2" ]; then
  echo "📥 Cloning LTX-2 repo..."
  git clone --depth 1 https://github.com/Lightricks/LTX-2.git
else
  echo "✅ LTX-2 directory already exists."
fi

# 3. Model Files Verification (Volume Check)
MODEL_DIR="LTX-2/models/ltx-2.5"
TRANSFORMER_FILE="$MODEL_DIR/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
DOWNLOAD_PID=""

if [ -f "$TRANSFORMER_FILE" ]; then
  echo "⚡ Models already exist in Volume! Skipping Hugging Face download."
else
  echo "📥 Models missing! Starting background download..."
  mkdir -p "$MODEL_DIR"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  pip install -q -U hf_transfer huggingface_hub

  hf download Lightricks/LTX-2.5 \
    diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    vae/ltx-2.5-video-vae-bf16.safetensors \
    vae/ltx-2.5-audio-vae-bf16.safetensors \
    model_patches/ltx-2.5-duration-head-bf16.safetensors \
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --token "$HF_TOKEN" \
    --local-dir "$MODEL_DIR" &
  DOWNLOAD_PID=$!
fi

# 4. Wait for APT packages if running
if [ -n "$APT_PID" ]; then
  wait $APT_PID
fi

# 5. Move or Update server.py inside LTX-2
if [ -f "server.py" ]; then
  echo "📄 Syncing server.py into LTX-2..."
  cp -f server.py LTX-2/
fi

# 6. Python Environment Setup
cd LTX-2

# Virtualenv check (Agar already configured hai toh sync instant bypass hoga)
if [ ! -f ".venv/bin/uvicorn" ]; then
  echo "⚙️ Setting up virtual environment..."
  uv sync --compile-bytecode
  uv pip install fastapi uvicorn python-multipart
else
  echo "✅ Virtual environment already configured."
fi

# 7. Wait for models if downloading in background
if [ -n "$DOWNLOAD_PID" ]; then
  echo "⏳ Waiting for models download to finish..."
  wait $DOWNLOAD_PID
fi

echo "=========================================="
echo "🚀 Everything ready! Launching FastAPI..."
echo "=========================================="

# 8. Start FastAPI Server
exec .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
