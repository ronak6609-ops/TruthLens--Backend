import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os

# No model loaded at startup — uses detector.py lazy model
from detector import detect_image

# Target frame size before inference — reduces tensor overhead significantly
INFERENCE_RESIZE = (224, 224)
# Number of frames to sample (beginning, 1/3, 2/3, end)
MAX_FRAMES = 4


def extract_frames(video_path: str, max_frames: int = MAX_FRAMES):
    """
    Extract strategically sampled frames: beginning, 1/3, 2/3, end.
    Skips redundant middle frames to minimize inference time.
    Frames are resized early to reduce memory usage.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    dur   = total / fps if fps > 0 else 0

    if total <= 0:
        cap.release()
        return [], total, fps, dur

    # Strategic sampling: spread across the video
    if total <= max_frames:
        indices = list(range(total))
    else:
        # beginning / ~1/3 / ~2/3 / end — avoids redundant adjacent frames
        indices = [
            0,
            int(total * 0.33),
            int(total * 0.66),
            total - 1,
        ]
        # Deduplicate while preserving order
        seen = set()
        indices = [x for x in indices if not (x in seen or seen.add(x))]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR→RGB and resize immediately to save memory
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, INFERENCE_RESIZE, interpolation=cv2.INTER_LINEAR)
            frames.append((idx, resized))

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
            # Frame already resized; convert to PIL and encode as JPEG for detect_image
            pil_img = Image.fromarray(frame_arr)
            buf = io.BytesIO()
            # quality=80 balances speed vs accuracy; frame already at 224x224
            pil_img.save(buf, format="JPEG", quality=80)
            img_bytes = buf.getvalue()
            buf.close()

            result = detect_image(img_bytes)
            ts = orig_idx / fps if fps > 0 else 0

            frame_results.append({
                "frame": i + 1,
                "timestamp": round(ts, 2),
                "is_ai": result["is_ai_generated"],
                "ai_score": result["fake_score"],
                "real_score": result["real_score"],
            })
            ai_scores.append(result["fake_score"])
            print(f"Frame {i + 1} (t={ts:.1f}s): AI={result['fake_score']:.1f}%")

        avg_ai = float(np.mean(ai_scores))
        max_ai = float(np.max(ai_scores))
        ai_cnt = sum(1 for r in frame_results if r["is_ai"])
        ai_pct = ai_cnt / len(frame_results) * 100
        is_ai  = ai_pct > 50

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
        try:
            os.unlink(tmp_path)
        except Exception:
            pass