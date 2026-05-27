from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os

from detector import detect_image
from video_detector import detect_video
from news_detector import detect_news
from audio_detector import detect_audio

app = FastAPI(
    title="TruthLens API",
    description="AI authenticity detection for images, video, audio, and news.",
    version="1.2.0",
)

# Allow all origins for development. In production, restrict to your frontend domain.
# Example: allow_origins=["https://your-frontend.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska", "video/webm"}
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/ogg", "audio/flac", "audio/x-m4a"}
MAX_IMAGE_SIZE = 20 * 1024 * 1024   # 20 MB
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200 MB
MAX_AUDIO_SIZE = 50 * 1024 * 1024   # 50 MB


@app.get("/")
def root():
    return {"status": "ok", "service": "TruthLens API", "version": "1.2.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ── IMAGE ──────────────────────────────────────────────────────────────────────

@app.post("/detect/image")
async def detect_img(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB limit.")
    content_type = file.content_type or ""
    if content_type and content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported image type: {content_type}")
    try:
        result = detect_image(contents)
        return {"success": True, "filename": file.filename, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── VIDEO ──────────────────────────────────────────────────────────────────────

@app.post("/detect/video")
async def detect_vid(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=413, detail="Video exceeds 200 MB limit.")
    try:
        result = detect_video(contents)
        return {"success": True, "filename": file.filename, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── AUDIO ──────────────────────────────────────────────────────────────────────

@app.post("/detect/audio")
async def detect_aud(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=413, detail="Audio exceeds 50 MB limit.")
    try:
        result = detect_audio(contents, file.filename or "audio.wav")
        return {"success": True, "filename": file.filename, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── NEWS ───────────────────────────────────────────────────────────────────────

@app.post("/detect/news")
async def detect_news_ep(text: str = Form(...)):
    if not text or len(text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Text must be at least 20 characters.")
    try:
        result = detect_news(text)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── VERSIONED ALIASES (for Flutter / future API clients) ──────────────────────
# These mirror the routes above under /v1/ so the Flutter app and web
# can share the same base URL and just switch prefix if needed.

@app.post("/v1/detect/image")
async def v1_detect_img(file: UploadFile = File(...)):
    return await detect_img(file)

@app.post("/v1/detect/video")
async def v1_detect_vid(file: UploadFile = File(...)):
    return await detect_vid(file)

@app.post("/v1/detect/audio")
async def v1_detect_aud(file: UploadFile = File(...)):
    return await detect_aud(file)

@app.post("/v1/detect/news")
async def v1_detect_news(text: str = Form(...)):
    return await detect_news_ep(text)