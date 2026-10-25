#!/usr/bin/env bash
set -e

# --- 0. Speed Test (Directly check Download & Upload bandwidth) ---
echo "=== Testing Internet Speed ==="
pip install -q speedtest-cli
speedtest-cli --simple
echo "=============================="

# 1. Faster APT mirror & non-interactive install (Added ffmpeg for audio/video processing)
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq tmux aria2 ffmpeg &
APT_PID=$!

# 2. Git Clone parallelly (shallow depth)
if [ ! -d "LTX-2" ]; then
  git clone --depth 1 https://github.com/Lightricks/LTX-2.git &
  GIT_PID=$!
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

# 5. Wait for apt and git clone to finish
wait $APT_PID
if [ -n "$GIT_PID" ]; then
  wait $GIT_PID
fi

# --- Move server.py into LTX-2 folder ---
if [ -f "server.py" ]; then
  echo "Moving server.py into LTX-2 directory..."
  mv server.py LTX-2/
fi

# 6. uv sync using maximum available CPU threads
cd LTX-2
uv sync --compile-bytecode

# Install server requirements inside virtual environment
uv pip install fastapi uvicorn python-multipart

# 7. Wait for models download to complete
wait $DOWNLOAD_PID

source .venv/bin/activate
echo "Setup finished at max speed!"

# Optional: Server start karne ke liye (uncomment kar sakte hain)
uvicorn server:app --host 0.0.0.0 --port 8000
