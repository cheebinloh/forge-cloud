"""Fetch the Wan 2.2 files ComfyUI needs into its models tree.

    python models.py 5b      # TI2V-5B fp16 + its VAE + umt5 (~18 GB)
    python models.py 14b     # I2V-A14B high/low fp8 + Wan 2.1 VAE (~29 GB)

Idempotent: files already present are skipped, so a stopped-and-restarted
box only re-checks.
"""
import os
import shutil
import sys

from huggingface_hub import hf_hub_download

ROOT = os.environ.get("COMFY_MODELS", "/workspace/ComfyUI/models")
REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
SETS = {
    "5b": [
        ("split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", "diffusion_models"),
        ("split_files/vae/wan2.2_vae.safetensors", "vae"),
        ("split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    ],
    "14b": [
        ("split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "diffusion_models"),
        ("split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "diffusion_models"),
        ("split_files/vae/wan_2.1_vae.safetensors", "vae"),
        ("split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    ],
}


def fetch(which):
    for fn, sub in SETS[which]:
        dst = os.path.join(ROOT, sub, os.path.basename(fn))
        if os.path.exists(dst):
            print("have", dst, flush=True)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print("get", fn, flush=True)
        p = hf_hub_download(REPO, fn, local_dir=os.path.join(ROOT, "_dl"))
        shutil.move(p, dst)
        print("done", dst, round(os.path.getsize(dst) / 1e9, 2), "GB", flush=True)
    shutil.rmtree(os.path.join(ROOT, "_dl"), ignore_errors=True)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["5b"]:
        fetch(w)
