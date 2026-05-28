from transformers import pipeline
import re

# ── Lazy loading ─────────────────────────────────────────────────
_news_model = None

def _load():
    global _news_model
    if _news_model is None:
        print("Loading fake news model... (first request)")
        _news_model = pipeline(
            "text-classification",
            model="hamzab/roberta-fake-news-classification"
        )
        print("✅ News model loaded!")


def clean_text(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def detect_news(text: str) -> dict:
    if len(text.strip()) < 20:
        return {"success": False, "error": "Text too short."}

    _load()

    cleaned = clean_text(text)
    words = cleaned.split()
    if len(words) > 400:
        cleaned = ' '.join(words[:400])

    result = _news_model(cleaned)[0]
    print("News model output:", result)

    label   = result['label'].upper()
    score   = result['score']
    is_fake = label in ['FAKE', 'LABEL_0', '0']

    return {
        "success": True,
        "is_fake": is_fake,
        "confidence": round(score * 100, 2),
        "label": "Fake News" if is_fake else "Likely Real",
        "raw_label": label,
        "word_count": len(words),
    }