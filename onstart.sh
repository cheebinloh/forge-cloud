#!/bin/bash
# Unleashed Forge cloud video — Vast.ai onstart. Runs as root on every boot
# of the instance; everything lands in /workspace, which Vast keeps across
# stop/start, so only the first boot is slow (pip + ~47 GB of models).
#
# Env from the template:  TS_AUTHKEY (tailscale), TS_HOSTNAME (vast-video),
#                         VIDEO_PASSWORD (for the public HTTPS door), CLOUD_REPO
exec > >(tee -a /workspace/onstart.log) 2>&1
set -x
cd /workspace || exit 1
S=/workspace/status.txt
st() { echo "$1" >> "$S"; }
: > "$S"
st "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl ffmpeg libgl1 libglib2.0-0 > /dev/null

# --- tailscale: userspace networking, containers have no /dev/net/tun
if ! command -v tailscale > /dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi
if [ -n "$TS_AUTHKEY" ]; then
  st "Joining tailnet"
  (tailscaled --tun=userspace-networking --socks5-server=localhost:1055 \
     --state=/workspace/tailscale.state > /workspace/tailscaled.log 2>&1 &)
  sleep 4
  tailscale up --authkey="$TS_AUTHKEY" --hostname="${TS_HOSTNAME:-vast-video}" \
    --accept-dns=false --reset || st "tailscale up failed (see onstart.log)"
fi

# --- our code (public repo; keys never live in it)
st "Fetching server"
REPO="${CLOUD_REPO:-https://github.com/cheebinloh/forge-cloud}"
rm -rf /workspace/forge-cloud
git clone --depth 1 "$REPO" /workspace/forge-cloud || st "git clone failed"

# --- ComfyUI
st "Installing ComfyUI"
if [ ! -d /workspace/ComfyUI ]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI
fi
pip install -q -r /workspace/ComfyUI/requirements.txt \
  websocket-client fastapi uvicorn python-multipart pillow huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- models: 5B first so the box is usable early, 14B keeps downloading behind
st "Downloading models (5B, ~18 GB)"
python /workspace/forge-cloud/models.py 5b

st "Starting ComfyUI"
cd /workspace/ComfyUI
(python main.py --listen 127.0.0.1 --port 8188 --preview-method auto \
   > /workspace/comfy.log 2>&1 &)
cd /workspace/forge-cloud
(DATA_DIR=/workspace/data uvicorn server:app --host 0.0.0.0 --port 4890 \
   > /workspace/server.log 2>&1 &)
for i in $(seq 1 60); do
  curl -fs http://127.0.0.1:4890/api/health > /dev/null && break
  sleep 2
done
st "Ready (5B) — 14B still downloading"
python /workspace/forge-cloud/models.py 14b
st "Ready"
