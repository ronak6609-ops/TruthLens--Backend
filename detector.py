from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io

print("Loading model... please wait")

MODEL_ID = "haywoodsloan/ai-image-detector-deploy"

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()

print("id2label:", model.config.id2label)
print("✅ Model loaded!")


def detect_image(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    id2label = model.config.id2label

    scores = {id2label[i]: probs[i].item() for i in range(len(probs))}
    print("Scores:", scores)

    ai_score = 0.0
    real_score = 0.0

    for label, score in scores.items():
        l = label.lower()
        if any(x in l for x in ['ai', 'fake', 'artificial', 'generated']):
            ai_score = max(ai_score, score)
        elif any(x in l for x in ['real', 'human', 'natural']):
            real_score = max(real_score, score)

    if ai_score == 0.0 and real_score == 0.0:
        vals = list(scores.values())
        ai_score = vals[-1]
        real_score = vals[0]

    is_ai = ai_score > real_score
    confidence = max(ai_score, real_score)

    print(f"ai:{ai_score:.3f} real:{real_score:.3f} → {'AI' if is_ai else 'REAL'}")

    return {
        "is_ai_generated": is_ai,
        "confidence": round(confidence * 100, 2),
        "label": "AI Generated" if is_ai else "Real Image",
        "fake_score": round(ai_score * 100, 2),
        "real_score": round(real_score * 100, 2),
    }