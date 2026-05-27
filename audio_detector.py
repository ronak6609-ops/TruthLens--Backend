import numpy as np
import tempfile
import os

print("Loading audio detection model... please wait")

try:
    import librosa
    LIBROSA_AVAILABLE = True
    print("✅ librosa available")
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa not installed — acoustic analysis disabled")

try:
    from transformers import pipeline
    audio_model = pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection-V2")
    MODEL_AVAILABLE = True
    print("✅ Audio model loaded!")
except Exception as e:
    MODEL_AVAILABLE = False
    print(f"⚠️ Audio model failed: {e}")


def extract_acoustic_features(audio_path: str) -> dict:
    if not LIBROSA_AVAILABLE:
        return {"acoustic_ai_score": 0.0, "duration": 0.0, "pitch_variance": 0.0, "ai_hints": 0}
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=30)
        mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_std  = float(np.std(mfccs))
        zcr_std   = float(np.std(librosa.feature.zero_crossing_rate(y)))
        rms_std   = float(np.std(librosa.feature.rms(y=y)))
        duration  = float(librosa.get_duration(y=y, sr=sr))
        try:
            f0, vf, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0v = f0[vf] if vf is not None else np.array([])
            pitch_var = float(np.std(f0v)) if len(f0v) > 0 else 0.0
        except Exception:
            pitch_var = 0.0
        hints = sum([mfcc_std < 20, zcr_std < 0.02, rms_std < 0.02, 0 < pitch_var < 15])
        print(f"Acoustic: mfcc={mfcc_std:.2f} zcr={zcr_std:.4f} rms={rms_std:.4f} pitch={pitch_var:.2f} hints={hints}")
        return {"acoustic_ai_score": hints / 4.0, "duration": round(duration, 2),
                "pitch_variance": round(pitch_var, 2), "ai_hints": hints}
    except Exception as e:
        print(f"Acoustic analysis error: {e}")
        return {"acoustic_ai_score": 0.0, "duration": 0.0, "pitch_variance": 0.0, "ai_hints": 0}


def detect_audio(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    ext = os.path.splitext(filename)[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        fake_score = 0.0
        real_score = 0.0

        if MODEL_AVAILABLE:
            try:
                results = audio_model(tmp_path)
                print("Audio model output:", results)
                for r in results:
                    label = r['label'].lower()
                    if any(x in label for x in ['fake','spoof','synthetic','ai','deepfake']):
                        fake_score = max(fake_score, r['score'])
                    elif any(x in label for x in ['real','genuine','human','bonafide']):
                        real_score = max(real_score, r['score'])
                if fake_score == 0.0 and real_score == 0.0:
                    vals = sorted(results, key=lambda x: x['score'], reverse=True)
                    fake_score = vals[0]['score']
                    real_score = 1.0 - fake_score
            except Exception as e:
                print(f"Model inference error: {e}")
                fake_score = 0.5
                real_score = 0.5

        features = extract_acoustic_features(tmp_path)
        acoustic  = features["acoustic_ai_score"]
        final     = (fake_score * 0.70) + (acoustic * 0.30) if MODEL_AVAILABLE else acoustic
        is_fake   = final > 0.50

        print(f"Audio result: model={fake_score:.2f} acoustic={acoustic:.2f} final={final:.2f} → {'FAKE' if is_fake else 'REAL'}")

        return {
            "success": True,
            "is_fake": is_fake,
            "is_ai_generated": is_fake,
            "confidence": round(final * 100, 2),
            "label": "AI / Deepfake Audio" if is_fake else "Real Human Voice",
            "fake_score": round(final * 100, 2),
            "real_score": round((1 - final) * 100, 2),
            "details": {
                "model_fake_score": round(fake_score * 100, 2),
                "acoustic_ai_score": round(acoustic * 100, 2),
                "duration": features["duration"],
                "pitch_variance": features["pitch_variance"],
                "ai_hints": features["ai_hints"],
            },
        }
    except Exception as e:
        print(f"Audio detection error: {e}")
        return {"success": False, "error": str(e), "is_fake": False,
                "confidence": 0, "label": "Error", "fake_score": 0, "real_score": 0}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass