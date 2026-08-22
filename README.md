# Unleashed Forge — cloud video (Vast.ai template)

Image → video with Wan 2.2 (5B fp16 and 14B fp8) on a rented GPU, driven from
the Unleashed Forge phone page over Tailscale.

- `onstart.sh` — the Vast.ai onstart script: installs ComfyUI, joins the
  tailnet as `$TS_HOSTNAME`, downloads the models, starts ComfyUI and the server.
- `server.py` + `wan.py` — FastAPI server on :4890 (upload frame, generate,
  continue, list/play/delete clips). Tailscale clients are trusted; others need
  `VIDEO_PASSWORD`.
- `index.html` — a minimal phone page served by the box itself.
- `models.py` — fetches the model files from HuggingFace (idempotent).

Secrets come in as environment variables on the instance; nothing here holds a key.
