# app.py
import os
import joblib
import traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import nltk

# Ensure preprocess module is importable (needed for unpickling tokenizer if used)
import preprocess  # noqa: F401

# Download required NLTK resources at runtime if missing
NLTK_DATA_PATH = "/root/nltk_data"
nltk.data.path.append(NLTK_DATA_PATH)
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_PATH)

# Environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "models/pipeline.pkl")
PORT = int(os.environ.get("PORT", 5000))

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Label mapping
LABEL_MAP = {0: "negative", 1: "positive"}

# Load pipeline
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run `python train.py` to create it.")
pipeline = joblib.load(MODEL_PATH)

# Extract label classes
try:
    classes = pipeline.named_steps['clf'].classes_
except Exception:
    classes = getattr(pipeline, "label_classes_", None)
    if classes is None:
        raise RuntimeError("Unable to read label classes from pipeline.")

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.json or {}
    text = body.get("text") or body.get("review") or ""
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Predict probabilities
        probs = pipeline.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        raw_label = classes[idx]
        label = LABEL_MAP.get(raw_label, str(raw_label))
        score = float(probs[idx] * 100.0)
        scores = {LABEL_MAP.get(c, str(c)): float(probs[i] * 100.0) for i, c in enumerate(classes)}
        return jsonify({
            "sentiment": label,
            "confidence": round(score, 2),
            "all_scores": {k: round(v, 2) for k, v in scores.items()}
        })
    except Exception as e:
        print("Predict proba failed:", e)
        traceback.print_exc()
        try:
            pred = pipeline.predict([text])[0]
            label = LABEL_MAP.get(pred, str(pred))
            return jsonify({
                "sentiment": label,
                "confidence": None,
                "all_scores": {},
                "error": str(e)
            })
        except Exception as e2:
            print("Fallback predict failed:", e2)
            traceback.print_exc()
            return jsonify({"error": "Prediction failed", "details": str(e2)}), 500

# Main
if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=PORT, debug=debug)
