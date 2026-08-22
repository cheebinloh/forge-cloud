"""Cloud video server — the small phone-facing side of the Vast.ai template.

One job at a time: upload a frame (or pick a finished clip to continue),
describe the motion, run Wan 2.2 5B or 14B in the local ComfyUI, keep the
mp4 + cover + info under DATA_DIR/videos. Auth: requests arriving over
Tailscale (100.64/10) are trusted; anything else needs the VIDEO_PASSWORD
cookie. Run:  uvicorn server:app --host 0.0.0.0 --port 4890
"""
import hashlib
import hmac
import json
import os
import random
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel

import wan

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_DIR", "/workspace/data"))
UPLOADS, VIDEOS = DATA / "uploads", DATA / "videos"
for d in (UPLOADS, VIDEOS):
    d.mkdir(parents=True, exist_ok=True)
PASSWORD = os.environ.get("VIDEO_PASSWORD", "")
# per-instance secret the phone page learns same-origin from its own server;
# a foreign site in the phone's browser can post here (the IP is trusted) but
# cannot know this, so every mutating call must carry it
TOKEN = os.environ.get("CLOUD_TOKEN", "")
COOKIE = "ufv"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
STATUS_FILE = Path(os.environ.get("STATUS_FILE", "/workspace/status.txt"))
FORGE_URL = os.environ.get("FORGE_URL", "http://127.0.0.1:1888")
FORGE_MODELS = Path(os.environ.get("FORGE_MODELS", "/workspace/models"))
SERVICES = [x for x in os.environ.get("SERVICES", "comfy").split(",") if x]

app = FastAPI(title="Unleashed Forge cloud video")
# the phone page lives on the home PC and calls here straight over Tailscale
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


# ------------------------------------------------------------------ auth
def _token():
    return hmac.new(b"uf-cloud", PASSWORD.encode(), hashlib.sha256).hexdigest()


def _trusted(request):
    ip = (request.client.host if request.client else "") or ""
    if ip.startswith("127.") or ip == "::1":
        return True
    if ip.startswith("100.") and 64 <= int(ip.split(".")[1]) <= 127:   # tailscale CGNAT
        return True
    return bool(PASSWORD) and request.cookies.get(COOKIE) == _token()


def _has_token(request):
    return bool(TOKEN) and hmac.compare_digest(request.headers.get("x-token", ""), TOKEN)


@app.middleware("http")
async def _gate(request, call_next):
    p = request.url.path
    if request.method == "OPTIONS" or p in ("/login", "/api/health"):
        return await call_next(request)
    if not _trusted(request):
        if p == "/" or p.endswith(".html"):
            return RedirectResponse("/login")
        return JSONResponse({"error": "login required"}, status_code=401)
    if request.method in ("POST", "PUT", "DELETE") and TOKEN and not _has_token(request) \
            and not (PASSWORD and request.cookies.get(COOKIE) == _token()):
        return JSONResponse({"error": "missing token"}, status_code=403)
    return await call_next(request)



LOGIN = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Cloud video</title><body style="font-family:-apple-system,system-ui;background:#000;color:#fff;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
<form method=post style="width:280px"><h3 style="font-weight:600">Cloud video 云端视频</h3>
<input name=pw type=password placeholder="Password 密码" autofocus style="width:100%;padding:14px;
border-radius:12px;border:0;background:#1c1c1e;color:#fff;font-size:17px;box-sizing:border-box">
<button style="width:100%;margin-top:12px;padding:14px;border:0;border-radius:12px;background:#0a84ff;
color:#fff;font-size:17px;font-weight:600">Enter 进入</button>%s</form>"""


@app.get("/login")
def login_page():
    if not PASSWORD:
        return HTMLResponse(LOGIN % "<p style='color:#ff453a'>No VIDEO_PASSWORD set on the server — "
                            "use the Tailscale address instead.</p>")
    return HTMLResponse(LOGIN % "")


@app.post("/login")
def login(pw: str = Form("")):
    if not PASSWORD or not hmac.compare_digest(pw, PASSWORD):
        return HTMLResponse(LOGIN % "<p style='color:#ff453a'>Wrong password</p>", status_code=401)
    r = RedirectResponse("/", status_code=303)
    r.set_cookie(COOKIE, _token(), max_age=30 * 86400, httponly=True, samesite="lax")
    return r


# ------------------------------------------------------------------ page
@app.get("/")
def index():
    # the page only renders for trusted clients, so it may carry the token
    html = (HERE / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", TOKEN)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


def _gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return ""


def _forge_alive():
    try:
        import urllib.request
        urllib.request.urlopen(FORGE_URL + "/sdapi/v1/progress?skip_current_image=true",
                               timeout=3).read()
        return True
    except Exception:
        return False


def _dl_status():
    try:
        return json.loads((FORGE_MODELS / "status.json").read_text())
    except Exception:
        return None


@app.get("/api/health")
def health():
    """What the phone's Cloud page polls while the box is coming up."""
    boot = ""
    try:
        boot = STATUS_FILE.read_text(encoding="utf-8").strip().splitlines()[-1]
    except Exception:
        pass
    return {"ok": True, "services": SERVICES,
            "comfy": wan.alive() if "comfy" in SERVICES else None,
            "forge": _forge_alive() if "forge" in SERVICES else None,
            "gpu": _gpu(), "boot": boot,
            "models": {k: _model_present(k) for k in ("5b", "14b")},
            "downloads": _dl_status(), "busy": VIDEO["on"]}


@app.get("/api/manifest")
def manifest():
    """The Civitai picks this box booted with, with where they landed."""
    import base64
    try:
        items = json.loads(base64.b64decode(os.environ.get("MANIFEST_B64", "")).decode())
    except Exception:
        items = []
    for it in items:
        sub = "checkpoints" if it.get("type") == "checkpoint" else "loras"
        it["present"] = (FORGE_MODELS / sub / it.get("file_name", "")).exists()
    return items


class ForgeSelect(BaseModel):
    checkpoint: str


@app.post("/api/forge/select")
def forge_select(req: ForgeSelect):
    """Switch Forge's checkpoint. Flux / Krea checkpoints need the separate
    text encoders + VAE as 'additional modules'; SDXL must not have them."""
    import urllib.request
    try:
        bases = json.loads((FORGE_MODELS / "bases.json").read_text())
    except Exception:
        bases = {}
    stem = req.checkpoint.split(" [")[0]
    base = ""
    for fn, b in bases.items():
        if fn == stem or Path(fn).stem == Path(stem).stem:
            base = (b or "").lower()
    flux = "flux" in base or "krea" in base
    mods = []
    if flux:
        te = FORGE_MODELS / "text_encoder"
        mods = [str(te / "clip_l.safetensors"), str(te / "t5xxl_fp8_e4m3fn.safetensors"),
                str(FORGE_MODELS / "vae" / "ae.safetensors")]
    body = json.dumps({"sd_model_checkpoint": req.checkpoint,
                       "forge_additional_modules": mods,
                       "forge_preset": "flux" if flux else "xl"}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            FORGE_URL + "/sdapi/v1/options", data=body,
            headers={"Content-Type": "application/json"}, method="POST"), timeout=900).read()
    except Exception as e:
        return JSONResponse({"error": f"Forge: {str(e)[:200]}"}, status_code=502)
    return {"ok": True, "flux": flux, "modules": mods}


def _model_present(model):
    root = Path(os.environ.get("COMFY_MODELS", "/workspace/ComfyUI/models"))
    names = [wan.M["5b"]] if model == "5b" else [wan.M["14b_high"], wan.M["14b_low"]]
    return all((root / "diffusion_models" / n).exists() or (root / "unet" / n).exists()
               for n in names)


# ------------------------------------------------------------------ media
def _safe(rel):
    try:
        p = (DATA / rel).resolve()
    except OSError:
        return None
    return p if DATA.resolve() in p.parents and p.is_file() else None


def _ranged(p, request, media_type):
    size = p.stat().st_size
    rng = request.headers.get("range", "")
    start, end = 0, size - 1
    if rng.startswith("bytes="):
        a, _, b = rng[6:].partition("-")
        try:
            if a:
                start, end = int(a), (int(b) if b else size - 1)
            elif b:
                start = max(0, size - int(b))
        except ValueError:
            pass
        end = min(end, size - 1)
        if start > end:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    with open(p, "rb") as f:
        f.seek(start)
        data = f.read(end - start + 1)
    return Response(data, status_code=206 if rng else 200, media_type=media_type,
                    headers={"Accept-Ranges": "bytes",
                             "Content-Range": f"bytes {start}-{end}/{size}",
                             "Cache-Control": "max-age=600"})


@app.get("/media/{rel:path}")
def media(rel: str, request: Request):
    from fastapi.responses import FileResponse
    p = _safe(rel)
    if p is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    if p.suffix.lower() == ".mp4":
        return _ranged(p, request, "video/mp4")
    return FileResponse(p, headers={"Cache-Control": "max-age=600"})


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """A frame from the phone's camera roll -> uploads/<id>.png (RGB, EXIF
    orientation applied, so portrait shots stay portrait)."""
    from PIL import Image, ImageOps
    import io
    raw = await file.read()
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception:
        return JSONResponse({"error": "not an image"}, status_code=400)
    im.thumbnail((2048, 2048))
    name = f"u_{uuid.uuid4().hex[:10]}.png"
    im.save(UPLOADS / name)
    return {"name": name, "url": f"/media/uploads/{name}", "width": im.width, "height": im.height}


def _video_info(p):
    try:
        return json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    except Exception:
        return {}


@app.get("/api/videos")
def videos():
    out = []
    for p in sorted(VIDEOS.glob("*.mp4"), key=lambda x: -x.stat().st_mtime):
        info = _video_info(p)
        out.append({"name": p.name, "url": f"/media/videos/{p.name}",
                    "poster": f"/media/videos/{p.stem}.jpg" if p.with_suffix(".jpg").exists() else None,
                    "mtime": int(p.stat().st_mtime), "size": p.stat().st_size, **info})
    return out


@app.delete("/api/videos/{name}")
def delete_video(name: str):
    p = _safe(f"videos/{name}")
    if p is None or p.suffix.lower() != ".mp4":
        return JSONResponse({"error": "not found"}, status_code=404)
    for q in (p, p.with_suffix(".jpg"), p.with_suffix(".json")):
        if q.exists():
            q.unlink()
    return {"deleted": name}


# ------------------------------------------------------------------ the job
VIDEO = {"on": False, "phase": "", "url": None, "poster": None, "error": None,
         "started": 0.0, "done": 0.0, "model": ""}


class VideoReq(BaseModel):
    image: str = ""          # uploads/<name>.png  (from /api/upload)
    video: str = ""          # or videos/<name>.mp4 to continue
    prompt: str = ""
    negative: str = ""
    seconds: int = 5
    size: int = 704          # short side: 480 or 704 (720p)
    model: str = "5b"        # 5b | 14b
    steps: int = 20
    seed: int = -1


def _probe(p):
    """Frame count / size of a clip without a sidecar (PyAV ships with ComfyUI)."""
    try:
        import av
        c = av.open(str(p))
        s = c.streams.video[0]
        n = sum(1 for _ in c.decode(s))
        info = {"frames": n, "width": s.width, "height": s.height,
                "fps": float(s.average_rate or 24)}
        p.with_suffix(".json").write_text(json.dumps(info), encoding="utf-8")
        return info
    except Exception:
        return None


def _job(req, src):
    model = "14b" if req.model == "14b" else "5b"
    VIDEO["model"] = model
    try:
        if not wan.alive():
            raise RuntimeError("ComfyUI is not running")
        cont = src.suffix.lower() == ".mp4"
        VIDEO["phase"] = "Uploading " + ("clip" if cont else "frame")
        name = wan.upload(src, "video/mp4" if cont else "image/png")

        fps = wan.FPS[model]
        length = wan.FRAMES[model].get(int(req.seconds), wan.FRAMES[model][5])
        steps = max(4, min(40, int(req.steps)))
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
        from PIL import Image
        if cont:
            info = _video_info(src) or _probe(src)
            if not info:
                raise RuntimeError("cannot read this clip's frame count")
            n_old, w, h = int(info["frames"]), int(info["width"]), int(info["height"])
            # a clip continues at its own model's frame rate
            model = info.get("model_key", model)
            fps = wan.FPS[model]
            length = wan.FRAMES[model].get(int(req.seconds), wan.FRAMES[model][5])
            graph = wan.continue_graph(model, name, n_old, req.prompt, req.negative,
                                       w, h, length, steps, seed)
            frames = n_old + length - 1
        else:
            im = Image.open(src)
            iw, ih = im.size
            short = max(256, min(720, int(req.size)))
            if iw >= ih:
                h, w = short, round(iw / ih * short / 32) * 32
            else:
                w, h = short, round(ih / iw * short / 32) * 32
            w, h = min(w, 1280), min(h, 1280)
            graph = wan.i2v_graph(model, name, req.prompt, req.negative,
                                  w, h, length, steps, seed)
            frames = length

        VIDEO["phase"] = "Generating"
        data = wan.run_graph(graph, steps=steps, timeout=5400)

        VIDEO["phase"] = "Saving"
        stem = f"v_{datetime.now():%Y%m%d_%H%M%S}_{seed}"
        (VIDEOS / f"{stem}.mp4").write_bytes(data)
        if cont:
            if src.with_suffix(".jpg").exists():
                shutil.copyfile(src.with_suffix(".jpg"), VIDEOS / f"{stem}.jpg")
        else:
            im.convert("RGB").resize((w, h)).save(VIDEOS / f"{stem}.jpg", quality=88)
        (VIDEOS / f"{stem}.json").write_text(json.dumps({
            "frames": frames, "fps": fps, "width": w, "height": h,
            "seconds": round(frames / fps, 2), "prompt": req.prompt, "seed": seed,
            "model": "Wan2.2 " + model.upper(), "model_key": model,
            "source": req.video or req.image}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        VIDEO.update(url=f"/media/videos/{stem}.mp4", poster=f"/media/videos/{stem}.jpg")
    except Exception as e:
        VIDEO["error"] = str(e)[:300] or type(e).__name__
    finally:
        VIDEO.update(on=False, phase="", done=time.time())


@app.post("/api/video")
def video_start(req: VideoReq):
    if VIDEO["on"]:
        return JSONResponse({"error": "Already generating"}, status_code=409)
    src = _safe(req.video or req.image)
    if src is None or src.suffix.lower() not in IMG_EXT | {".mp4"}:
        return JSONResponse({"error": "pick an image first"}, status_code=404)
    if not _model_present("14b" if req.model == "14b" else "5b"):
        return JSONResponse({"error": f"the {req.model} model is not downloaded yet"},
                            status_code=503)
    VIDEO.update(on=True, phase="Starting", url=None, poster=None, error=None,
                 started=time.time(), done=0.0)
    threading.Thread(target=_job, args=(req, src), daemon=True).start()
    return {"started": True}


@app.get("/api/progress")
def progress():
    cp = wan.PROG
    sampling = VIDEO["phase"] == "Generating" and cp["max"]
    return {"active": VIDEO["on"], "phase": VIDEO["phase"], "model": VIDEO["model"],
            "progress": (cp["value"] / cp["max"]) if (VIDEO["on"] and sampling) else 0,
            "step": cp["value"] if sampling else 0, "steps": cp["max"] if sampling else 0,
            "preview": cp.get("preview") if VIDEO["on"] else None,
            "url": VIDEO["url"], "poster": VIDEO["poster"], "error": VIDEO["error"],
            "elapsed": int(time.time() - VIDEO["started"]) if VIDEO["started"] else 0,
            "done": VIDEO["done"]}


@app.post("/api/stop")
def stop():
    wan.interrupt()
    return {"stopped": True}
