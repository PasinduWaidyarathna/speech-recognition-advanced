from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from mangum import Mangum
import tensorflow as tf
import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
import tempfile
import os
import io

app = FastAPI()
handler = Mangum(app)

# Global model variable
model = None
MODEL_PATH = "models/lung_sound_classification_model_1.keras"

# Load model
def load_audio_model(path: str):
    try:
        m = tf.keras.models.load_model(path)
        print("Model loaded successfully")
        return m
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize model at cold start
if os.path.exists(MODEL_PATH):
    model = load_audio_model(MODEL_PATH)
else:
    print(f"Model file not found at {MODEL_PATH}")

# Prediction logic
def predict_health(audio_data: np.ndarray, sr: int, model) -> tuple:
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=400)
    mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)
    preds = model.predict(mfccs_processed)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))
    status = "Healthy" if class_id == 1 else "Unhealthy"
    return status, confidence

@app.get("/")
def health_check():
    return {"status": "Healthy"}

@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not audio_file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
        tmp.write(await audio_file.read())

    try:
        audio, sr = librosa.load(tmp_path, sr=None)
        status, confidence = predict_health(audio, sr, model)
        return JSONResponse({"status": status, "confidence": confidence})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/noise-reduction")
async def noise_reduction(
    noise_only: UploadFile = File(...),
    heart_noisy: UploadFile = File(...)
):
    if not noise_only.filename.lower().endswith(".wav") or \
       not heart_noisy.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files supported")

    paths = {}
    for key, file in [("noise", noise_only), ("heart", heart_noisy)]:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        paths[key] = tmp.name
        tmp.write(await file.read())
        tmp.close()

    try:
        noisy, sr = librosa.load(paths['heart'], sr=None)
        noise, _ = librosa.load(paths['noise'], sr=sr)
        clean = nr.reduce_noise(y=noisy, y_noise=noise, sr=sr)

        buffer = io.BytesIO()
        sf.write(buffer, clean, sr, format='WAV')
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type='audio/wav',
            headers={
                'Content-Disposition': 'attachment; filename="cleaned.wav"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        for p in paths.values():
            if os.path.exists(p):
                os.remove(p)
