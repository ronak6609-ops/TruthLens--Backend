from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os

# ✅ NO model imports at top level — lazy loaded inside each route

app = FastAPI(
    title="TruthLens API",
    description="AI authenticity detection for images, video, audio, and news.",
    version="1.2.0",
)

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
MAX_IMAGE_SIZE = 20 * 1024 * 1024
MAX_VIDEO_SIZE = 200 * 1024 * 1024
MAX_AUDIO_SIZE = 50 * 1024 * 1024


@app.get("/")
def root():
    return {"status": "ok", "service": "TruthLens API", "version": "1.2.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ── IMAGE ──────────────────────────────────────────────────────────────────────

@app.post("/detect/image")
async def detect_img(file: UploadFile = File(...)):
    from detector import detect_image  # lazy import
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
    from video_detector import detect_video  # lazy import
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
    from audio_detector import detect_audio  # lazy import
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
    from news_detector import detect_news  # lazy import
    if not text or len(text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Text must be at least 20 characters.")
    try:
        result = detect_news(text)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── VERSIONED ALIASES ─────────────────────────────────────────────────────────

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