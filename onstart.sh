#!/bin/bash
# Unleashed Forge cloud - Vast.ai onstart. Runs as root on every boot of the
# instance; everything lands in /workspace, which Vast keeps across
# stop/start, so only the first boot is slow (pip + model downloads).
#
# Env from the template:
#   SERVICES       comfy,forge (any subset)     TS_AUTHKEY / TS_HOSTNAME  tailscale
#   MANIFEST_B64   base64 json list of Civitai files for Forge (checkpoints, loras)
#   CIVITAI_TOKEN  Civitai API key for those downloads
#   CLOUD_TOKEN    per-instance secret the phone must send on writes
#   VIDEO_PASSWORD password for the public door; CLOUD_REPO the code to clone
exec > >(tee -a /workspace/onstart.log) 2>&1
set -x
cd /workspace || exit 1
S=/workspace/status.txt
st() { echo "$1" >> "$S"; }
: > "$S"
SERVICES="${SERVICES:-comfy}"
WAN_MODELS="${WAN_MODELS:-5b,14b}"
has() { case ",$SERVICES," in *",$1,"*) return 0;; *) return 1;; esac; }
wan() { case ",$WAN_MODELS," in *",$1,"*) return 0;; *) return 1;; esac; }

st "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl ffmpeg libgl1 libglib2.0-0 gcc g++ aria2 > /dev/null
# comfy-kitchen compiles a triton helper at import and links -lcuda; the
# runtime image ships only libcuda.so.1, so give it the dev-style name too
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so

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
pip install -q websocket-client fastapi uvicorn python-multipart pillow huggingface_hub hf_transfer requests
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /workspace/data /workspace/models/checkpoints /workspace/models/loras \
         /workspace/models/vae /workspace/models/text_encoder

# the phone-facing server comes up first: it reports the boot stages below
cd /workspace/forge-cloud
(DATA_DIR=/workspace/data uvicorn server:app --host 0.0.0.0 --port 4890 \
   > /workspace/server.log 2>&1 &)

# --- Forge (vForge): the user's Civitai picks + flux text encoders
if has forge; then
  st "Installing Forge"
  if [ ! -d /workspace/forge ]; then
    git clone --depth 1 https://github.com/lllyasviel/stable-diffusion-webui-forge /workspace/forge
  fi
  st "Downloading Forge models"
  python /workspace/forge-cloud/models.py forge     # manifest + flux TEs/VAE, writes models/status.json
  st "Starting Forge"
  cd /workspace/forge
  # first pass installs Forge's pins and exits; scikit-image then needs a wheel
  # that matches the numpy it ends up with, or processing.py dies on import
  python launch.py --skip-torch-cuda-test --skip-version-check --exit > /workspace/forge-install.log 2>&1 || true
  pip install -q --force-reinstall --no-deps scikit-image==0.21.0
  # the image's torch stays (launch.py only installs torch when it is missing)
  (python launch.py --api --listen --port 1888 --skip-torch-cuda-test --skip-version-check \
     --ckpt-dir /workspace/models/checkpoints --lora-dir /workspace/models/loras \
     --vae-dir /workspace/models/vae --text-encoder-dir /workspace/models/text_encoder \
     --no-download-sd-model --api-log --cuda-malloc \
     > /workspace/forge.log 2>&1 &)
  cd /workspace
fi

# --- ComfyUI (vComfy): Wan 2.2 video
if has comfy; then
  st "Installing ComfyUI"
  if [ ! -d /workspace/ComfyUI ]; then
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI
  fi
  pip install -q -r /workspace/ComfyUI/requirements.txt
  if wan 5b; then
    st "Downloading models (5B, ~18 GB)"
    python /workspace/forge-cloud/models.py 5b
  elif wan 14b; then
    st "Downloading models (14B, ~29 GB)"
    python /workspace/forge-cloud/models.py 14b
  fi
  st "Starting ComfyUI"
  cd /workspace/ComfyUI
  (python main.py --listen 127.0.0.1 --port 8188 --preview-method auto \
     > /workspace/comfy.log 2>&1 &)
  cd /workspace
fi

for i in $(seq 1 60); do
  curl -fs http://127.0.0.1:4890/api/health > /dev/null && break
  sleep 2
done
if has comfy && wan 5b && wan 14b; then
  st "Ready (5B) - 14B still downloading"
  python /workspace/forge-cloud/models.py 14b
fi
st "Ready"
