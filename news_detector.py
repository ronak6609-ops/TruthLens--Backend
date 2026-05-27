from transformers import pipeline
import re

print("Loading fake news detection model...")

# Best fake news model - trained on LIAR dataset
# Labels: REAL / FAKE
news_model = pipeline(
    "text-classification",
    model="hamzab/roberta-fake-news-classification"
)

print("✅ Fake news model loaded!")


def clean_text(text: str) -> str:
    text = re.sub(r'http\S+', '', text)       # remove URLs
    text = re.sub(r'\s+', ' ', text)           # clean whitespace
    return text.strip()


def detect_news(text: str) -> dict:
    if len(text.strip()) < 20:
        return {
            "success": False,
            "error": "Text too short. Please provide at least 20 characters."
        }

    cleaned = clean_text(text)

    # Truncate to 512 tokens max (model limit)
    words = cleaned.split()
    if len(words) > 400:
        cleaned = ' '.join(words[:400])

    result = news_model(cleaned)[0]
    print("News model output:", result)

    label = result['label'].upper()
    score = result['score']

    is_fake = label in ['FAKE', 'LABEL_0', '0']

    # Some models flip label meaning — check confidence
    confidence = round(score * 100, 2)

    return {
        "success": True,
        "is_fake": is_fake,
        "confidence": confidence,
        "label": "Fake News" if is_fake else "Likely Real",
        "raw_label": label,
        "raw_score": score,
        "text_length": len(text),
        "word_count": len(words),
    }