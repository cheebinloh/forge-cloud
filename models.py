"""Fetch model files onto the box.

    python models.py 5b       # Wan 2.2 TI2V-5B fp16 + VAE + umt5 (~18 GB)   -> ComfyUI/models
    python models.py 14b      # Wan 2.2 I2V-A14B high/low fp8 + Wan 2.1 VAE   -> ComfyUI/models
    python models.py forge    # the Civitai picks in $MANIFEST_B64 + flux TEs  -> /workspace/models

Idempotent: files already present are skipped. Progress for the phone goes to
/workspace/models/status.json.
"""
import base64
import json
import os
import shutil
import sys
import time

from huggingface_hub import hf_hub_download

ROOT = os.environ.get("COMFY_MODELS", "/workspace/ComfyUI/models")
FORGE_ROOT = os.environ.get("FORGE_MODELS", "/workspace/models")
STATUS = os.path.join(FORGE_ROOT, "status.json")
REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
SETS = {
    "5b": [
        (REPO, "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", "diffusion_models"),
        (REPO, "split_files/vae/wan2.2_vae.safetensors", "vae"),
        (REPO, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    ],
    "14b": [
        (REPO, "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "diffusion_models"),
        (REPO, "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "diffusion_models"),
        (REPO, "split_files/vae/wan_2.1_vae.safetensors", "vae"),
        (REPO, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    ],
}
# what a flux / krea checkpoint needs beside itself in Forge
FLUX_EXTRAS = [
    ("comfyanonymous/flux_text_encoders", "clip_l.safetensors", "text_encoder"),
    ("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn.safetensors", "text_encoder"),
    ("black-forest-labs/FLUX.1-schnell", "ae.safetensors", "vae"),
]


def _status(**kw):
    try:
        cur = json.load(open(STATUS)) if os.path.exists(STATUS) else {}
    except Exception:
        cur = {}
    cur.update(kw, at=time.time())
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    json.dump(cur, open(STATUS, "w"))


def hf(repo, fn, sub, root=ROOT):
    dst = os.path.join(root, sub, os.path.basename(fn))
    if os.path.exists(dst):
        print("have", dst, flush=True)
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print("get", repo, fn, flush=True)
    p = hf_hub_download(repo, fn, local_dir=os.path.join(root, "_dl"))
    shutil.move(p, dst)
    print("done", dst, round(os.path.getsize(dst) / 1e9, 2), "GB", flush=True)
    return dst


def fetch_set(which):
    for repo, fn, sub in SETS[which]:
        hf(repo, fn, sub)
    shutil.rmtree(os.path.join(ROOT, "_dl"), ignore_errors=True)


def civitai(url, dst, token):
    """Download one Civitai file with aria2 (parallel connections, resumable);
    falls back to curl. Returns True on success."""
    import subprocess
    sep = "&" if "?" in url else "?"
    full = f"{url}{sep}token={token}" if token else url
    d, n = os.path.split(dst)
    os.makedirs(d, exist_ok=True)
    cmd = ["aria2c", "-x", "8", "-s", "8", "-k", "8M", "--file-allocation=none",
           "--continue=true", "--auto-file-renaming=false", "--allow-overwrite=true",
           "--summary-interval=0", "-d", d, "-o", n, full]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst):
        print("aria2 failed, trying curl:", r.stderr[-300:], flush=True)
        r = subprocess.run(["curl", "-L", "-sS", "--fail", "-o", dst, full],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("curl failed:", r.stderr[-300:], flush=True)
            return False
    # a tiny file is an error page (bad token / gated model), not a model
    return os.path.getsize(dst) > 1_000_000


def forge():
    """The manifest the phone picked at boot + what flux checkpoints need."""
    raw = os.environ.get("MANIFEST_B64", "")
    token = os.environ.get("CIVITAI_TOKEN", "")
    try:
        items = json.loads(base64.b64decode(raw).decode()) if raw else []
    except Exception:
        items = []
    need_flux = any("flux" in (it.get("base") or "").lower() or "krea" in (it.get("base") or "").lower()
                    for it in items if it.get("type") == "checkpoint")
    total = len(items) + (len(FLUX_EXTRAS) if need_flux else 0)
    done, failed = 0, []
    _status(total=total, done=0, current="", failed=[])
    for it in items:
        sub = "checkpoints" if it.get("type") == "checkpoint" else "loras"
        dst = os.path.join(FORGE_ROOT, sub, it["file_name"])
        _status(done=done, current=it.get("name") or it["file_name"])
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            print("have", dst, flush=True)
        elif not civitai(it["url"], dst, token):
            failed.append(it.get("name") or it["file_name"])
            if os.path.exists(dst):
                os.remove(dst)
        done += 1
        _status(done=done, failed=failed)
    if need_flux:
        for repo, fn, sub in FLUX_EXTRAS:
            _status(done=done, current=os.path.basename(fn))
            try:
                hf(repo, fn, sub, root=FORGE_ROOT)
            except Exception as e:
                failed.append(os.path.basename(fn) + ": " + str(e)[:80])
            done += 1
            _status(done=done, failed=failed)
        shutil.rmtree(os.path.join(FORGE_ROOT, "_dl"), ignore_errors=True)
    # what Forge should treat as flux, for the server's checkpoint switch
    json.dump({it["file_name"]: (it.get("base") or "") for it in items
               if it.get("type") == "checkpoint"},
              open(os.path.join(FORGE_ROOT, "bases.json"), "w"))
    _status(done=done, current="", failed=failed, finished=True)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["5b"]:
        if w == "forge":
            forge()
        else:
            fetch_set(w)
