import os
os.environ["NUMBA_DISABLE_JIT"] = "1"


from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, session, redirect, url_for, send_file
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename
from torchvision import models
from datetime import datetime
from io import BytesIO

# ================= APP =================
app = Flask(__name__)
app.secret_key = "instrunet-temp-history"

UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"wav", "mp3", "flac"}
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # reduced size
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= DEVICE =================
device = torch.device("cpu")  # FORCE CPU for free plan

# ================= LOAD MODEL (LOAD ONCE) =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_resnet18_irmas.pth")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 11)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ================= LABELS =================
INSTRUMENTS = [
    "Cello","Clarinet","Flute","Acoustic Guitar","Electric Guitar",
    "Organ","Piano","Saxophone","Trumpet","Violin","Voice"
]

# ================= UTILS =================
def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ================= PAGES =================
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/analysis")
def analysis_page():
    return render_template("analysis.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/history")
def history_page():
    history = session.get("history", [])
    return render_template("history.html", history=history[:5])

# ================= UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("audio")

    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    original_name = secure_filename(file.filename)
    filename = f"{uuid.uuid4()}_{original_name}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    session["last_original_name"] = original_name

    return jsonify({"filename": filename})

@app.route("/uploads/<filename>")
def serve_audio(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= ANALYZE (OPTIMIZED) =================
@app.route("/analyze", methods=["POST"])
def analyze():
    filename = request.json.get("filename")
    if not filename:
        return jsonify({"error": "Missing filename"}), 400

    path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        # 🔥 LIMIT TO 10 SECONDS
        y, sr = librosa.load(path, sr=22050, mono=True, duration=10)
    except Exception:
        return jsonify({"error": "Failed to decode audio"}), 400

    if len(y) < sr:
        return jsonify({"error": "Audio too short"}), 400

    window_sec, hop_sec = 2.0, 1.0
    seg_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)

    segments = []

    for start in range(0, len(y) - seg_len + 1, hop_len):
        seg = y[start:start + seg_len]

        rms = np.mean(librosa.feature.rms(y=seg))
        if rms < 1e-4:
            continue

        mel = librosa.feature.melspectrogram(
            y=seg,
            sr=sr,
            n_mels=64,
            n_fft=1024,
            hop_length=256,
            fmax=8000
        )

        mel = librosa.power_to_db(mel, ref=np.max)

        if mel.shape[1] >= 64:
            mel = mel[:, :64]
        else:
            mel = np.pad(mel, ((0,0),(0,64-mel.shape[1])))

        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        segments.append(mel)

        if len(segments) >= 25:
            break

    if not segments:
        return jsonify({"error": "No meaningful audio detected"}), 400

    x = torch.from_numpy(np.stack(segments)).unsqueeze(1).float()
    x = F.interpolate(x, size=(128,128), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).numpy()

    avg = np.mean(probs, axis=0)
    top_idx = int(np.argmax(avg))

    probs_dict = {
        INSTRUMENTS[i]: float(avg[i])
        for i in range(len(INSTRUMENTS))
    }

    sorted_probs = sorted(
        probs_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    if "history" not in session:
        session["history"] = []

    original_name = session.get("last_original_name", filename)

    session["history"].insert(0, {
        "id": str(uuid.uuid4()),
        "filename": filename,
        "original_name": original_name,
        "prediction": INSTRUMENTS[top_idx],
        "confidence": float(avg[top_idx]),
        "file_path": f"uploads/{filename}",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "detected": sorted_probs[:5]
    })

    session["history"] = session["history"][:10]
    session.modified = True

    return jsonify({
        "filename": filename,
        "average": probs_dict
    })

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
