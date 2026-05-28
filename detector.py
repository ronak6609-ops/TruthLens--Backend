from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io

# ── Lazy loading ─────────────────────────────────────────────────
_processor = None
_model = None
MODEL_ID = "haywoodsloan/ai-image-detector-deploy"

def _load():
    global _processor, _model
    if _processor is None or _model is None:
        print("Loading image detection model... (first request)")
        _processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        _model.eval()
        print("✅ Image model loaded!")

def detect_image(image_bytes: bytes) -> dict:
    _load()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = _processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    id2label = _model.config.id2label
    scores = {id2label[i]: probs[i].item() for i in range(len(probs))}
    print("Image model scores:", scores)
    fake_score = 0.0
    real_score = 0.0
    for label, score in scores.items():
        l = label.lower()
        if any(x in l for x in ['fake', 'artificial', 'ai', 'generated']):
            fake_score = max(fake_score, score)
        elif any(x in l for x in ['real', 'human', 'natural']):
            real_score = max(real_score, score)
    if fake_score == 0.0 and real_score == 0.0:
        vals = list(scores.values())
        fake_score = vals[-1]
        real_score = vals[0]
    is_ai = fake_score > real_score
    confidence = max(fake_score, real_score)
    print(f"Image result: fake={fake_score:.3f} real={real_score:.3f} → {'AI' if is_ai else 'REAL'}")
    return {
        "is_ai_generated": is_ai,
        "confidence": round(confidence * 100, 2),
        "label": "AI Generated" if is_ai else "Real Image",
        "fake_score": round(fake_score * 100, 2),
        "real_score": round(real_score * 100, 2),
    }