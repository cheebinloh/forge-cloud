"""ComfyUI client for Wan 2.2 image-to-video — the cloud copy.

Standalone on purpose (no imports from the rest of the suite) so the
template only needs this folder. Talks to a local ComfyUI over HTTP, mirrors
step progress from its websocket into PROG, and builds the two graphs:
5B (TI2V, 24 fps) and 14B (I2V high/low noise experts, 16 fps), plus the
"continue from the last frame" variant of either.
"""
import json
import os
import threading
import time
import urllib.parse
import urllib.request
import uuid

BASE = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")

# model files, overridable so a local GGUF test can share the code
M = {
    "loader": os.environ.get("WAN_LOADER", "unet"),          # unet | gguf
    "5b": os.environ.get("WAN5B_UNET", "wan2.2_ti2v_5B_fp16.safetensors"),
    "5b_vae": os.environ.get("WAN5B_VAE", "wan2.2_vae.safetensors"),
    "14b_high": os.environ.get("WAN14B_HIGH", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"),
    "14b_low": os.environ.get("WAN14B_LOW", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"),
    "14b_vae": os.environ.get("WAN14B_VAE", "wan_2.1_vae.safetensors"),
    "clip": os.environ.get("WAN_CLIP", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
}
FPS = {"5b": 24, "14b": 16}
# Wan wants 4k+1 frames
FRAMES = {"5b": {2: 49, 3: 81, 5: 121, 8: 193},
          "14b": {2: 33, 3: 49, 5: 81, 8: 129}}
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
       "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
       "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
       "杂乱的背景，三条腿，背景人很多，倒着走")

PROG = {"active": False, "value": 0, "max": 1, "at": 0.0, "node": None,
        "preview": None}


def _get(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def alive(timeout=3):
    try:
        _get("/system_stats", timeout=timeout)
        return True
    except Exception:
        return False


def interrupt():
    try:
        urllib.request.urlopen(urllib.request.Request(
            BASE + "/interrupt", method="POST"), timeout=10)
    except Exception:
        pass


def free():
    try:
        body = json.dumps({"unload_models": True, "free_memory": True}).encode()
        urllib.request.urlopen(urllib.request.Request(
            BASE + "/free", data=body, headers={"Content-Type": "application/json"}),
            timeout=60)
    except Exception:
        pass


def upload(path, content_type="image/png"):
    """Put a file into ComfyUI's input folder; returns the name it got."""
    import mimetypes
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    with open(path, "rb") as f:
        data = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
            f"filename=\"{name}\"\r\nContent-Type: {content_type}\r\n\r\n").encode() \
        + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + "/upload/image", data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["name"]


def _watch_ws(client_id, done_evt):
    try:
        import websocket
    except ImportError:
        return
    try:
        host = BASE.split("://", 1)[1]
        ws = websocket.create_connection(f"ws://{host}/ws?clientId={client_id}", timeout=8)
        ws.settimeout(5)
        while not done_evt.is_set():
            try:
                msg = ws.recv()
            except Exception:
                continue
            if not isinstance(msg, str):
                if len(msg) > 8 and int.from_bytes(msg[:4], "big") == 1:
                    import base64
                    PROG.update(preview=base64.b64encode(msg[8:]).decode(), at=time.time())
                continue
            m = json.loads(msg)
            if m.get("type") == "executing":
                PROG.update(node=(m.get("data") or {}).get("node"), at=time.time())
            if m.get("type") == "progress":
                d = m["data"]
                PROG.update(value=d.get("value", 0), max=max(1, d.get("max", 1)),
                            at=time.time())
        ws.close()
    except Exception:
        pass


def run_graph(graph, steps=1, timeout=3600):
    """Submit, mirror progress, return the bytes of the first image/video output."""
    client_id = uuid.uuid4().hex
    PROG.update(active=True, value=0, max=steps, at=time.time(), node=None, preview=None)
    done = threading.Event()
    threading.Thread(target=_watch_ws, args=(client_id, done), daemon=True).start()
    try:
        body = json.dumps({"prompt": graph, "client_id": client_id}).encode()
        req = urllib.request.Request(BASE + "/prompt", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                pid = json.loads(r.read())["prompt_id"]
        except urllib.error.HTTPError as e:
            raise RuntimeError("ComfyUI rejected the graph: " + e.read().decode()[:300])
        t0 = time.time()
        while time.time() - t0 < timeout:
            entry = _get(f"/history/{pid}", timeout=10).get(pid)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    if any(m[0] == "execution_interrupted" for m in msgs):
                        raise RuntimeError("interrupted")
                    err = [m for m in msgs if m[0] == "execution_error"]
                    raise RuntimeError((err[0][1].get("exception_message", "")
                                        if err else "ComfyUI execution error")[:300])
                outs = []
                for out in entry.get("outputs", {}).values():
                    outs += out.get("images", []) + out.get("video", [])
                if outs:
                    q = urllib.parse.urlencode({
                        "filename": outs[0]["filename"],
                        "subfolder": outs[0].get("subfolder", ""),
                        "type": outs[0].get("type", "output")})
                    with urllib.request.urlopen(f"{BASE}/view?{q}", timeout=300) as r:
                        return r.read()
            time.sleep(1.0)
        raise TimeoutError("ComfyUI did not finish in time")
    finally:
        done.set()
        PROG.update(active=False, value=0, at=time.time(), preview=None)


def _unet(name):
    if M["loader"] == "gguf":
        return {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": name}}
    return {"class_type": "UNETLoader",
            "inputs": {"unet_name": name, "weight_dtype": "default"}}


def i2v_graph(model, image_name, prompt, negative, width, height, length, steps, seed):
    """model: '5b' (TI2V, one sampler, uni_pc, 24 fps) or '14b' (two experts:
    high noise for the first half of the steps, low noise for the rest, 16 fps)."""
    g = {
        "c": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": M["clip"], "type": "wan", "device": "default"}},
        "p": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["c", 0], "text": prompt}},
        "n": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["c", 0], "text": negative or NEG}},
        "i": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "cv": {"class_type": "CreateVideo", "inputs": {"images": ["d", 0], "fps": float(FPS[model])}},
        "s": {"class_type": "SaveVideo",
              "inputs": {"video": ["cv", 0], "filename_prefix": "uf_video",
                         "format": "mp4", "codec": "h264"}},
    }
    if model == "14b":
        half = max(1, steps // 2)
        g.update({
            "uh": _unet(M["14b_high"]), "ul": _unet(M["14b_low"]),
            "v": {"class_type": "VAELoader", "inputs": {"vae_name": M["14b_vae"]}},
            "mh": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["uh", 0], "shift": 8.0}},
            "ml": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["ul", 0], "shift": 8.0}},
            "l": {"class_type": "WanImageToVideo",
                  "inputs": {"positive": ["p", 0], "negative": ["n", 0], "vae": ["v", 0],
                             "width": width, "height": height, "length": length,
                             "batch_size": 1, "start_image": ["i", 0]}},
            "k1": {"class_type": "KSamplerAdvanced",
                   "inputs": {"model": ["mh", 0], "add_noise": "enable", "noise_seed": seed,
                              "steps": steps, "cfg": 3.5, "sampler_name": "euler",
                              "scheduler": "simple", "positive": ["l", 0], "negative": ["l", 1],
                              "latent_image": ["l", 2], "start_at_step": 0,
                              "end_at_step": half, "return_with_leftover_noise": "enable"}},
            "k2": {"class_type": "KSamplerAdvanced",
                   "inputs": {"model": ["ml", 0], "add_noise": "disable", "noise_seed": 0,
                              "steps": steps, "cfg": 3.5, "sampler_name": "euler",
                              "scheduler": "simple", "positive": ["l", 0], "negative": ["l", 1],
                              "latent_image": ["k1", 0], "start_at_step": half,
                              "end_at_step": 10000, "return_with_leftover_noise": "disable"}},
            "d": {"class_type": "VAEDecode", "inputs": {"samples": ["k2", 0], "vae": ["v", 0]}},
        })
    else:
        g.update({
            "u": _unet(M["5b"]),
            "v": {"class_type": "VAELoader", "inputs": {"vae_name": M["5b_vae"]}},
            "m": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["u", 0], "shift": 8.0}},
            "l": {"class_type": "Wan22ImageToVideoLatent",
                  "inputs": {"vae": ["v", 0], "width": width, "height": height,
                             "length": length, "batch_size": 1, "start_image": ["i", 0]}},
            "k": {"class_type": "KSampler",
                  "inputs": {"model": ["m", 0], "positive": ["p", 0], "negative": ["n", 0],
                             "latent_image": ["l", 0], "seed": seed, "steps": steps,
                             "cfg": 5.0, "sampler_name": "uni_pc", "scheduler": "simple",
                             "denoise": 1.0}},
            "d": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["v", 0]}},
        })
    return g


def continue_graph(model, video_name, n_old, prompt, negative, width, height,
                   length, steps, seed):
    """The clip's last frame seeds a new segment; old + new (minus the seam
    duplicate) come out as one video."""
    g = i2v_graph(model, "", prompt, negative, width, height, length, steps, seed)
    del g["i"]
    g["lv"] = {"class_type": "LoadVideo", "inputs": {"file": video_name}}
    g["gc"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["lv", 0]}}
    g["lf"] = {"class_type": "ImageFromBatch",
               "inputs": {"image": ["gc", 0], "batch_index": max(0, n_old - 1), "length": 1}}
    g["l"]["inputs"]["start_image"] = ["lf", 0]
    g["nb"] = {"class_type": "ImageFromBatch",
               "inputs": {"image": ["d", 0], "batch_index": 1, "length": max(1, length - 1)}}
    g["cat"] = {"class_type": "ImageBatch", "inputs": {"image1": ["gc", 0], "image2": ["nb", 0]}}
    g["cv"]["inputs"]["images"] = ["cat", 0]
    return g
