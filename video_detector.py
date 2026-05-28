import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os

# No model loaded at startup — uses detector.py lazy model
from detector import detect_image

def extract_frames(video_path: str, max_frames: int = 10):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    dur   = total / fps if fps > 0 else 0
    indices = [int(i * total / max_frames) for i in range(max_frames)] if total > max_frames else list(range(total))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, total, fps, dur

def detect_video(video_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        frames, total_frames, fps, duration = extract_frames(tmp_path)
        if not frames:
            return {"success": False, "error": "Could not extract frames"}
        frame_results = []
        ai_scores = []
        for i, (orig_idx, frame_arr) in enumerate(frames):
            pil_img = Image.fromarray(frame_arr)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            result = detect_image(buf.getvalue())
            ts = orig_idx / fps if fps > 0 else 0
            frame_results.append({
                "frame": i + 1,
                "timestamp": round(ts, 2),
                "is_ai": result["is_ai_generated"],
                "ai_score": result["fake_score"],
                "real_score": result["real_score"],
            })
            ai_scores.append(result["fake_score"])
            print(f"Frame {i+1}: AI={result['fake_score']:.1f}%")
        avg_ai  = np.mean(ai_scores)
        max_ai  = np.max(ai_scores)
        ai_cnt  = sum(1 for r in frame_results if r["is_ai"])
        ai_pct  = ai_cnt / len(frame_results) * 100
        is_ai   = ai_pct > 50
        return {
            "success": True,
            "is_ai_generated": is_ai,
            "confidence": round(avg_ai, 2),
            "label": "AI Generated" if is_ai else "Real Video",
            "summary": {
                "total_frames_analyzed": len(frame_results),
                "ai_frames": ai_cnt,
                "real_frames": len(frame_results) - ai_cnt,
                "ai_frame_percentage": round(ai_pct, 1),
                "avg_ai_score": round(avg_ai, 2),
                "max_ai_score": round(max_ai, 2),
                "video_duration": round(duration, 1),
                "fps": round(fps, 1),
            },
            "frame_results": frame_results,
        }
    finally:
        try: os.unlink(tmp_path)
        except: pass