"""
app.py  —  Image Authenticity Analyzer
Run:  python app.py
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ExifTags
import imagehash

from modules.feature_extractor import extract_features_with_steps

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load trained model ─────────────────────────────────────────
MODEL_PATH = "model/model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "model/model.pkl not found. Run:  python train_model.py  first."
    )
with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)
print("✅ Model loaded.")


# ── Per-metric suspicion scores (0-100) for UI bars ────────────
def _sus_dct(v):
    if v < 5:   return 90
    if v < 10:  return 65
    if v < 25:  return 15
    if v < 45:  return 10
    if v < 65:  return 35
    return 60

def _sus_fft(v):
    if v < 80:  return 85
    if v < 150: return 55
    if v < 280: return 15
    if v < 380: return 40
    return 75

def _sus_prnu(v):          # v is already ×100
    if v < 10:  return 90
    if v < 30:  return 65
    if v < 60:  return 35
    if v < 90:  return 15
    return 5


# ── Plain-English explanations ──────────────────────────────────
def _explain_dct(v, s):
    if s >= 70:
        return (f"Score {v:.1f} — blocks look unnaturally smooth or "
                "over-sharpened. Real photos have varied texture; this one doesn't.")
    if s >= 35:
        return (f"Score {v:.1f} — slight texture irregularities. "
                "Could be heavy compression or minor editing.")
    return (f"Score {v:.1f} — texture looks natural and varied, "
            "consistent with a real photograph.")

def _explain_fft(v, s):
    if s >= 70:
        return (f"Score {v:.1f} — abnormal frequency spectrum. "
                "Often caused by GAN upsampling grids or heavy compositing.")
    if s >= 35:
        return (f"Score {v:.1f} — some frequency irregularities. "
                "Possibly from resizing or social-media re-compression.")
    return (f"Score {v:.1f} — frequency distribution looks normal, "
            "matching what a real camera produces.")

def _explain_prnu(v, s):
    if s >= 70:
        return (f"Score {v:.1f} — little or no sensor fingerprint detected. "
                "AI images have no physical sensor so this is absent.")
    if s >= 35:
        return (f"Score {v:.1f} — partial sensor fingerprint. "
                "Some regions inconsistent — may indicate splicing.")
    return (f"Score {v:.1f} — strong, consistent sensor noise found. "
            "Reliable sign the photo came from a real camera.")


# ── Helpers ─────────────────────────────────────────────────────
def _get_metadata(path):
    try:
        exif = Image.open(path)._getexif()
        if not exif:
            return {}
        return {ExifTags.TAGS.get(k, k): str(v)
                for k, v in exif.items()
                if k in ExifTags.TAGS and len(str(v)) < 120}
    except Exception:
        return {}

def _check_duplicate(path):
    try:
        h         = str(imagehash.phash(Image.open(path)))
        seen_file = "static/seen_hashes.txt"
        seen      = set(open(seen_file).read().splitlines()) if os.path.exists(seen_file) else set()
        is_dup    = h in seen
        if not is_dup:
            open(seen_file, "a").write(h + "\n")
        return is_dup
    except Exception:
        return False


# ── Routes ───────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files or request.files["image"].filename == "":
        return "No file selected", 400

    file = request.files["image"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    # Feature extraction
    try:
        data = extract_features_with_steps(path)
    except Exception as e:
        return f"Feature extraction failed: {e}", 500

    features      = data["features"]
    dct_blocks    = data["dct_blocks"]
    fft_peaks     = data["fft_peaks"]
    prnu_progress = data["prnu_progress"]
    dct_mean      = data["dct_mean"]
    fft_mean      = data["fft_mean"]
    prnu_mean     = data["prnu_mean"]

    # Model prediction
    feat_arr   = np.array(features).reshape(1, -1)
    pred_label = int(pipeline.predict(feat_arr)[0])
    pred_proba = pipeline.predict_proba(feat_arr)[0]   # [p_real, p_ai]

    confidence = min(int(round(max(pred_proba) * 100)), 98)

    if max(pred_proba) < 0.60:
        result = "Uncertain"
    elif pred_label == 0:
        result = "Likely Real"
    else:
        result = "Likely AI Generated"

    # UI suspicion bars
    dct_suspicion  = _sus_dct(dct_mean)
    fft_suspicion  = _sus_fft(fft_mean)
    prnu_suspicion = _sus_prnu(prnu_mean)

    explanation = (
        f"Model confidence: {confidence}%. "
        f"P(real) = {pred_proba[0]:.2f}, P(AI) = {pred_proba[1]:.2f}. "
        f"Analyzed 22 forensic features: DCT texture, FFT spectrum, "
        f"PRNU sensor fingerprint, LBP micro-texture, edge coherence, "
        f"color statistics, and JPEG blocking artifacts."
    )

    return render_template(
        "index.html",
        image          = file.filename,
        dct_blocks     = dct_blocks,
        fft_peaks      = fft_peaks,
        prnu_progress  = prnu_progress,
        dct_mean       = dct_mean,
        fft_mean       = fft_mean,
        prnu_mean      = prnu_mean,
        dct_suspicion  = dct_suspicion,
        fft_suspicion  = fft_suspicion,
        prnu_suspicion = prnu_suspicion,
        dct_explain    = _explain_dct(dct_mean,   dct_suspicion),
        fft_explain    = _explain_fft(fft_mean,   fft_suspicion),
        prnu_explain   = _explain_prnu(prnu_mean, prnu_suspicion),
        result         = result,
        confidence     = confidence,
        explanation    = explanation,
        metadata       = _get_metadata(path),
        duplicate      = _check_duplicate(path),
        p_real         = round(float(pred_proba[0]) * 100, 1),
        p_ai           = round(float(pred_proba[1]) * 100, 1),
    )


if __name__ == "__main__":
    app.run(debug=True)