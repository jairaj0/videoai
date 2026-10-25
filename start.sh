#!/usr/bin/env bash
set -e

# --- 0. Speed Test (Directly check Download & Upload bandwidth) ---
echo "=== Testing Internet Speed ==="
pip install -q speedtest-cli
speedtest-cli --simple
echo "=============================="

# 1. Background APT install (ffmpeg, tmux, aria2)
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq tmux aria2 ffmpeg &
APT_PID=$!

# 2. Git Clone (Shallow clone 1-2 sec leta hai, ise synchronously run karenge taaki directory conflict na ho)
if [ ! -d "LTX-2" ]; then
  echo "Cloning LTX-2 repo..."
  git clone --depth 1 https://github.com/Lightricks/LTX-2.git
fi

# 3. Enable HF Transfer (Rust-based multi-threaded max-bandwidth download engine)
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q -U hf_transfer huggingface_hub

# 4. Background Model Download (Parallel multi-connection saturating network)
mkdir -p LTX-2/models/ltx-2.5

hf download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  model_patches/ltx-2.5-duration-head-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --token "$HF_TOKEN" \
  --local-dir LTX-2/models/ltx-2.5 &
DOWNLOAD_PID=$!

# 5. Wait for apt packages
wait $APT_PID

# --- Move server.py into LTX-2 folder ---
if [ -f "server.py" ]; then
  echo "Moving server.py into LTX-2 directory..."
  mv server.py LTX-2/
fi

# 6. Set up Python environment & dependencies inside LTX-2
cd LTX-2
uv sync --compile-bytecode
.venv/bin/pip install fastapi uvicorn python-multipart

# 7. Wait for models download to complete
wait $DOWNLOAD_PID

echo "=========================================="
echo " Setup complete! Starting FastAPI Server..."
echo "=========================================="

# 8. Start FastAPI Server (Keep container running)
exec .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
