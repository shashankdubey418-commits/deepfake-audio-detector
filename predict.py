from pathlib import Path
import sys

import numpy as np
import torch
import torchaudio
import joblib

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model
)


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    PROJECT_DIR
    / "deepfake_audio_classifier.joblib"
)

MODEL_NAME = "facebook/wav2vec2-base"

TARGET_SR = 16000

MAX_AUDIO_SECONDS = 5


# ============================================================
# CHECK INPUT
# ============================================================

if len(sys.argv) != 2:

    print(
        "\nUsage:"
        "\npython3 predict.py <audio_file.wav>"
    )

    print(
        "\nExample:"
        "\npython3 predict.py my_audio.wav"
    )

    sys.exit(1)


audio_file = Path(sys.argv[1])


if not audio_file.exists():

    print(
        f"\nERROR: File not found:"
        f"\n{audio_file}"
    )

    sys.exit(1)


if not MODEL_FILE.exists():

    print(
        "\nERROR: Trained model not found:"
        f"\n{MODEL_FILE}"
    )

    print(
        "\nTrain the model first using:"
        "\npython3 trainbaseline.py"
    )

    sys.exit(1)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


print("=" * 60)
print("AUDIO DEEPFAKE DETECTOR")
print("=" * 60)

print(
    f"\nAudio file:"
    f"\n{audio_file}"
)

print(
    f"\nDevice: {DEVICE}"
)


# ============================================================
# LOAD CLASSIFIER
# ============================================================

print("\nLoading trained classifier...")

classifier = joblib.load(
    MODEL_FILE
)

print("Classifier loaded.")


# ============================================================
# LOAD WAV2VEC2
# ============================================================

print("\nLoading Wav2Vec2...")

feature_extractor = (
    Wav2Vec2FeatureExtractor
    .from_pretrained(MODEL_NAME)
)

wav2vec_model = (
    Wav2Vec2Model
    .from_pretrained(MODEL_NAME)
)

wav2vec_model.to(DEVICE)
wav2vec_model.eval()

print("Wav2Vec2 loaded.")


# ============================================================
# LOAD AUDIO
# ============================================================

print("\nReading audio...")

waveform, sr = torchaudio.load(
    str(audio_file)
)


# ============================================================
# STEREO -> MONO
# ============================================================

if waveform.shape[0] > 1:

    waveform = waveform.mean(
        dim=0,
        keepdim=True
    )


# ============================================================
# RESAMPLE
# ============================================================

if sr != TARGET_SR:

    print(
        f"Resampling {sr} Hz → {TARGET_SR} Hz..."
    )

    waveform = (
        torchaudio.functional
        .resample(
            waveform,
            sr,
            TARGET_SR
        )
    )


waveform = waveform.squeeze(0)


# ============================================================
# LIMIT AUDIO LENGTH
# ============================================================

max_samples = (
    TARGET_SR * MAX_AUDIO_SECONDS
)

if waveform.shape[0] > max_samples:

    print(
        f"\nAudio is longer than "
        f"{MAX_AUDIO_SECONDS} seconds."
    )

    print(
        f"Using the first "
        f"{MAX_AUDIO_SECONDS} seconds."
    )

    waveform = waveform[:max_samples]


# ============================================================
# WAV2VEC2 FEATURE EXTRACTION
# ============================================================

print("\nExtracting audio features...")

inputs = feature_extractor(
    waveform.numpy(),
    sampling_rate=TARGET_SR,
    return_tensors="pt"
)

input_values = (
    inputs.input_values
    .to(DEVICE)
)


with torch.inference_mode():

    outputs = wav2vec_model(
        input_values
    )

    embedding = (
        outputs.last_hidden_state
        .mean(dim=1)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning deepfake detector...")

prediction = classifier.predict(
    embedding.reshape(1, -1)
)[0]


probabilities = (
    classifier
    .predict_proba(
        embedding.reshape(1, -1)
    )[0]
)


real_probability = probabilities[0]
fake_probability = probabilities[1]


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)


if prediction == 0:

    label = "REAL"
    confidence = real_probability

else:

    label = "FAKE / CLONED"
    confidence = fake_probability


print(
    f"\nPrediction: {label}"
)

print(
    f"Confidence: {confidence * 100:.2f}%"
)

print(
    f"\nReal probability:"
    f" {real_probability * 100:.2f}%"
)

print(
    f"Fake probability:"
    f" {fake_probability * 100:.2f}%"
)

print(
    "\n" + "=" * 60
)
