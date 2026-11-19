from pathlib import Path
from io import BytesIO
from math import gcd

import soundfile as sf
from scipy.signal import resample_poly

import joblib
import numpy as np
import streamlit as st
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


# ------------------------------------------------------------
# SETTINGS — kept consistent with predict.py
# ------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "deepfake_audio_classifier.joblib"
MODEL_NAME = "facebook/wav2vec2-base"
TARGET_SR = 16000
MAX_AUDIO_SECONDS = 5


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Loading your trained classifier...")
def load_classifier():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}. "
            "Keep deepfake_audio_classifier.joblib beside app.py."
        )
    return joblib.load(MODEL_FILE)


@st.cache_resource(show_spinner="Loading Wav2Vec2 (first run may take a while)...")
def load_wav2vec():
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    model.eval()
    return extractor, model, device


# ------------------------------------------------------------
# PREDICTION — same preprocessing and embedding flow as predict.py
# ------------------------------------------------------------
def analyze_audio(audio_bytes):
    classifier = load_classifier()
    extractor, wav2vec_model, device = load_wav2vec()

    # Read WAV directly with SoundFile. This avoids torchaudio's
    # TorchCodec backend, which can fail on some macOS/PyTorch setups.
    audio_array, sample_rate = sf.read(
        BytesIO(audio_bytes),
        dtype="float32",
        always_2d=True,
    )

    if audio_array.size == 0 or audio_array.shape[0] == 0:
        raise ValueError("The uploaded audio file is empty.")

    original_channels = audio_array.shape[1]
    original_duration = audio_array.shape[0] / sample_rate

    # Convert to mono (frames x channels -> frames)
    if original_channels > 1:
        audio_array = audio_array.mean(axis=1)

    # Resample to 16 kHz using SciPy, without TorchCodec.
    if sample_rate != TARGET_SR:
        common = gcd(int(sample_rate), TARGET_SR)
        up = TARGET_SR // common
        down = int(sample_rate) // common
        audio_array = resample_poly(audio_array, up, down).astype(np.float32)

    # Force a clean one-dimensional float32 waveform.
    audio_array = np.asarray(audio_array, dtype=np.float32).reshape(-1)

    if not np.isfinite(audio_array).all():
        raise ValueError("The audio contains invalid numeric values.")

    if sample_rate <= 0:
        raise ValueError(f"Invalid audio sample rate: {sample_rate} Hz.")

    waveform = torch.from_numpy(audio_array.copy())

    # Wav2Vec2's convolutional feature extractor needs a minimum
    # number of samples. At 16 kHz, 400 samples is about 25 ms.
    if waveform.numel() < 400:
        duration_ms = waveform.numel() / TARGET_SR * 1000
        raise ValueError(
            f"The decoded audio is too short for Wav2Vec2 "
            f"({waveform.numel()} samples, about {duration_ms:.1f} ms "
            f"after resampling). Please upload a valid WAV recording "
            f"that is at least 1 second long."
        )

    # Use first 5 seconds, matching predict.py
    max_samples = TARGET_SR * MAX_AUDIO_SECONDS
    was_trimmed = waveform.shape[0] > max_samples
    if was_trimmed:
        waveform = waveform[:max_samples]

    inputs = extractor(
        waveform.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    )
    input_values = inputs.input_values.to(device)

    with torch.inference_mode():
        outputs = wav2vec_model(input_values)
        embedding = (
            outputs.last_hidden_state
            .mean(dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    sample = embedding.reshape(1, -1)
    prediction = classifier.predict(sample)[0]
    probabilities = classifier.predict_proba(sample)[0]

    # Use classifier.classes_ to associate probabilities with class labels.
    classes = list(classifier.classes_)
    probability_by_class = {
        class_label: float(probabilities[index])
        for index, class_label in enumerate(classes)
    }

    # This follows the label convention in the supplied predict.py:
    # class 0 = REAL; any other class = FAKE / CLONED.
    if prediction == 0:
        label = "REAL"
    else:
        label = "FAKE / CLONED"

    confidence = probability_by_class[prediction]
    real_probability = probability_by_class.get(0, 0.0)
    fake_probability = probability_by_class.get(1, 0.0)

    return {
        "label": label,
        "confidence": confidence,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "duration": original_duration,
        "sample_rate": sample_rate,
        "channels": original_channels,
        "trimmed": was_trimmed,
        "device": str(device),
    }


# ------------------------------------------------------------
# PAGE DESIGN
# ------------------------------------------------------------
st.set_page_config(
    page_title="Audio Deepfake Detector",
    page_icon="🎙️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: #0b1220; color: #edf2f7; }
      [data-testid="stHeader"] { background: rgba(0,0,0,0); }
      [data-testid="stSidebar"] { background: #111c2e; }
      .hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid #26364e;
        border-radius: 18px;
        background: linear-gradient(120deg, #14243b, #101a2b);
        margin-bottom: 1.2rem;
      }
      .hero h1 { margin: 0; color: #f7fafc; font-size: 2.15rem; }
      .hero p { color: #a9b8cc; margin: .55rem 0 0 0; }
      .panel {
        background: #111c2e;
        border: 1px solid #26364e;
        border-radius: 16px;
        padding: 1.2rem;
      }
      .muted { color: #a9b8cc; }
      .result-real { color: #4ade80; font-size: 2rem; font-weight: 800; }
      .result-fake { color: #fb7185; font-size: 2rem; font-weight: 800; }
      div[data-testid="stFileUploader"] {
        background: #111c2e;
        border: 1px dashed #49617f;
        border-radius: 14px;
        padding: 1rem;
      }
      .stButton button {
        width: 100%; border-radius: 10px; min-height: 2.8rem;
        background: #2563eb; color: white; border: 0; font-weight: 700;
      }
      .stButton button:hover { background: #1d4ed8; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🎙️ Audio Deepfake Detector</h1>
      <p>Analyze an audio recording for signs of AI-generated or cloned speech.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## About the model")
    st.write("**Feature extractor:** Wav2Vec2")
    st.write("**Classifier:** Logistic Regression")
    st.write("**Input sample rate:** 16 kHz")
    st.write("**Analyzed audio:** First 5 seconds")
    st.divider()
    st.caption("This interface runs your existing trained model locally.")
    st.caption(
        "A model prediction is an estimate, not proof that a recording "
        "is authentic or manipulated."
    )

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown("### Upload audio")
    uploaded_file = st.file_uploader(
        "Choose a WAV audio file",
        type=["wav"],
        help="Upload a WAV recording to analyze it.",
    )

    if uploaded_file is not None:
        st.audio(uploaded_file.getvalue(), format="audio/wav")
        st.caption(f"Selected file: **{uploaded_file.name}**")

    analyze_clicked = st.button(
        "Analyze audio",
        type="primary",
        disabled=uploaded_file is None,
    )

with right:
    st.markdown("### How analysis works")
    st.markdown(
        """
        <div class="panel">
          <p>1. Read the uploaded WAV file</p>
          <p>2. Convert stereo audio to mono</p>
          <p>3. Resample audio to 16 kHz</p>
          <p>4. Use up to the first 5 seconds</p>
          <p>5. Extract Wav2Vec2 embeddings</p>
          <p>6. Classify with Logistic Regression</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if analyze_clicked and uploaded_file is not None:
    st.divider()
    st.markdown("### Analysis result")

    try:
        with st.spinner("Analyzing audio. The first run may take longer while Wav2Vec2 loads..."):
            result = analyze_audio(uploaded_file.getvalue())

        result_col, detail_col = st.columns([1, 1], gap="large")

        with result_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**PREDICTION**")
            result_class = (
                "result-real" if result["label"] == "REAL" else "result-fake"
            )
            st.markdown(
                f'<div class="{result_class}">{result["label"]}</div>',
                unsafe_allow_html=True,
            )
            st.metric("Prediction confidence", f'{result["confidence"] * 100:.2f}%')
            st.markdown("</div>", unsafe_allow_html=True)

        with detail_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**CLASS PROBABILITIES**")
            st.write(f'Real: {result["real_probability"] * 100:.2f}%')
            st.progress(result["real_probability"])
            st.write(f'Fake: {result["fake_probability"] * 100:.2f}%')
            st.progress(result["fake_probability"])
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Audio details")
        info1, info2, info3, info4 = st.columns(4)
        info1.metric("Duration", f'{result["duration"]:.2f} sec')
        info2.metric("Sample rate", f'{result["sample_rate"]:,} Hz')
        info3.metric("Channels", str(result["channels"]))
        info4.metric("Device", result["device"].upper())

        if result["trimmed"]:
            st.info(
                "This recording is longer than 5 seconds. "
                "As in your command-line predictor, only its first 5 seconds were analyzed."
            )

        st.warning(
            "Interpret the confidence as the classifier's estimated class probability, "
            "not a guarantee of authenticity. Performance depends on the training data "
            "and may vary for unfamiliar audio."
        )

    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Could not analyze this audio: {error}")
        st.exception(error)

st.divider()
st.caption("Audio Deepfake Detection • Wav2Vec2 + Logistic Regression")
