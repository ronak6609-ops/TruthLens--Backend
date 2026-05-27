import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
from detector import detect_image

def extract_frames(video_path: str, max_frames: int = 10) -> list:
    """Extract evenly spaced frames from video"""
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
    
    # Pick evenly spaced frame indices
    if total_frames <= max_frames:
        indices = list(range(total_frames))
    else:
        indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx, frame_rgb))
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames, total_frames, fps, duration


def detect_video(video_bytes: bytes) -> dict:
    """Analyze video for AI generation / deepfake"""
    
    # Save video to temp file (OpenCV needs file path)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    
    try:
        frames, total_frames, fps, duration = extract_frames(tmp_path, max_frames=10)
        
        if not frames:
            return {
                "success": False,
                "error": "Could not extract frames from video"
            }
        
        # Analyze each frame
        frame_results = []
        ai_scores = []
        
        for frame_idx, (orig_idx, frame_array) in enumerate(frames):
            # Convert numpy array to bytes for detector
            pil_image = Image.fromarray(frame_array)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            frame_bytes = buffer.getvalue()
            
            # Run image detector on this frame
            result = detect_image(frame_bytes)
            
            timestamp = orig_idx / fps if fps > 0 else 0
            
            frame_results.append({
                "frame": frame_idx + 1,
                "timestamp": round(timestamp, 2),
                "is_ai": result["is_ai_generated"],
                "ai_score": result["fake_score"],
                "real_score": result["real_score"],
            })
            
            ai_scores.append(result["fake_score"])
            print(f"Frame {frame_idx+1}: AI={result['fake_score']:.1f}%")
        
        # Calculate overall verdict
        avg_ai_score = np.mean(ai_scores)
        max_ai_score = np.max(ai_scores)
        ai_frame_count = sum(1 for r in frame_results if r["is_ai"])
        ai_frame_percentage = (ai_frame_count / len(frame_results)) * 100
        
        # Final verdict: AI if majority of frames are AI
        is_ai = ai_frame_percentage > 50
        
        print(f"Video result: avg={avg_ai_score:.1f}% max={max_ai_score:.1f}% ai_frames={ai_frame_percentage:.0f}%")
        
        return {
            "success": True,
            "is_ai_generated": is_ai,
            "confidence": round(avg_ai_score, 2),
            "label": "AI Generated" if is_ai else "Real Video",
            "summary": {
                "total_frames_analyzed": len(frame_results),
                "ai_frames": ai_frame_count,
                "real_frames": len(frame_results) - ai_frame_count,
                "ai_frame_percentage": round(ai_frame_percentage, 1),
                "avg_ai_score": round(avg_ai_score, 2),
                "max_ai_score": round(max_ai_score, 2),
                "video_duration": round(duration, 1),
                "fps": round(fps, 1),
            },
            "frame_results": frame_results,
        }
    
    finally:
        os.unlink(tmp_path)