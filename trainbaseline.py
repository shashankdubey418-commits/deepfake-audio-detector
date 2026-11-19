from pathlib import Path
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    PROJECT_DIR
    / "archive"
    / "for-norm"
    / "for-norm"
)

TRAIN_DIR = DATASET_DIR / "training"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "testing"

MODEL_NAME = "facebook/wav2vec2-base"

TARGET_SR = 16000

# Small first run
TRAIN_PER_CLASS = 10000
VAL_PER_CLASS = 250
TEST_PER_CLASS = 250

# Process one audio file at a time
MAX_AUDIO_SECONDS = 5

# Save progress every 25 files
CHECKPOINT_EVERY = 25

CACHE_DIR = PROJECT_DIR / "embedding_cache"
CHECKPOINT_DIR = CACHE_DIR / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("=" * 65)
print("AUDIO DEEPFAKE DETECTION PROJECT")
print("=" * 65)

print(f"\nDataset:")
print(DATASET_DIR)

print(f"\nDevice: {DEVICE}")

print(
    f"\nUsing:"
    f"\n  Training: {TRAIN_PER_CLASS} real + {TRAIN_PER_CLASS} fake"
    f"\n  Validation: {VAL_PER_CLASS} real + {VAL_PER_CLASS} fake"
    f"\n  Testing: {TEST_PER_CLASS} real + {TEST_PER_CLASS} fake"
)


# ============================================================
# CHECK DATASET
# ============================================================

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

    if not folder.exists():

        raise RuntimeError(
            f"Dataset folder not found:\n{folder}"
        )


# ============================================================
# FIND FILES
# ============================================================

def find_files(split_dir, limit):

    real_dir = split_dir / "real"
    fake_dir = split_dir / "fake"

    real_files = sorted(
        real_dir.rglob("*.wav")
    )

    fake_files = sorted(
        fake_dir.rglob("*.wav")
    )

    real_files = real_files[:limit]
    fake_files = fake_files[:limit]

    return real_files, fake_files


train_real, train_fake = find_files(
    TRAIN_DIR,
    TRAIN_PER_CLASS
)

val_real, val_fake = find_files(
    VAL_DIR,
    VAL_PER_CLASS
)

test_real, test_fake = find_files(
    TEST_DIR,
    TEST_PER_CLASS
)


print("\n" + "=" * 65)
print("DATASET SUMMARY")
print("=" * 65)

print(
    f"\nTraining:"
    f"\n  Real: {len(train_real)}"
    f"\n  Fake: {len(train_fake)}"
)

print(
    f"\nValidation:"
    f"\n  Real: {len(val_real)}"
    f"\n  Fake: {len(val_fake)}"
)

print(
    f"\nTesting:"
    f"\n  Real: {len(test_real)}"
    f"\n  Fake: {len(test_fake)}"
)


# ============================================================
# LOAD WAV2VEC2
# ============================================================

print("\n" + "=" * 65)
print("LOADING WAV2VEC2")
print("=" * 65)

print(
    f"\nLoading {MODEL_NAME}..."
)

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

print(
    "\nWav2Vec2 loaded successfully."
)


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(filepath):

    waveform, sr = torchaudio.load(
        str(filepath)
    )

    # Stereo -> mono
    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    # Resample
    if sr != TARGET_SR:

        waveform = (
            torchaudio.functional
            .resample(
                waveform,
                sr,
                TARGET_SR
            )
        )

    waveform = waveform.squeeze(0)

    # Limit audio length
    max_samples = (
        TARGET_SR * MAX_AUDIO_SECONDS
    )

    if waveform.shape[0] > max_samples:

        waveform = waveform[:max_samples]

    return waveform


# ============================================================
# EXTRACT ONE EMBEDDING
# ============================================================

def extract_embedding(filepath):

    waveform = load_audio(filepath)

    if waveform.numel() == 0:

        raise RuntimeError(
            "Empty audio file"
        )

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

    del input_values
    del outputs

    if DEVICE.type == "mps":

        torch.mps.empty_cache()

    return embedding


# ============================================================
# CHECKPOINT
# ============================================================

def get_checkpoint(
    split_name,
    class_name
):

    return (
        CHECKPOINT_DIR
        / f"{split_name}_{class_name}.npz"
    )


# ============================================================
# PROCESS CLASS
# ============================================================

def process_class(
    files,
    label,
    split_name,
    class_name
):

    checkpoint = get_checkpoint(
        split_name,
        class_name
    )

    embeddings = []
    processed = []

    # --------------------------------------------------------
    # LOAD EXISTING CHECKPOINT
    # --------------------------------------------------------

    if checkpoint.exists():

        print(
            f"\nFound checkpoint:"
            f"\n{checkpoint}"
        )

        data = np.load(
            checkpoint,
            allow_pickle=True
        )

        embeddings = list(
            data["embeddings"]
        )

        processed = list(
            data["files"]
        )

        processed = [
            str(x)
            for x in processed
        ]

        print(
            f"Already processed:"
            f" {len(processed)}"
        )

    processed_set = set(processed)

    remaining = [
        f
        for f in files
        if str(f) not in processed_set
    ]

    print(
        f"\nProcessing:"
        f" {split_name} / {class_name}"
    )

    print(
        f"Remaining:"
        f" {len(remaining)}"
    )

    # --------------------------------------------------------
    # PROCESS FILES
    # --------------------------------------------------------

    for filepath in tqdm(
        remaining,
        desc=f"{split_name} {class_name}"
    ):

        try:

            embedding = extract_embedding(
                filepath
            )

            embeddings.append(
                embedding
            )

            processed.append(
                str(filepath)
            )

        except Exception as e:

            print(
                f"\nSkipped:"
                f"\n{filepath}"
                f"\nReason: {e}"
            )

        # ----------------------------------------------------
        # SAVE CHECKPOINT
        # ----------------------------------------------------

        if (
            len(processed)
            % CHECKPOINT_EVERY
            == 0
        ):

            np.savez_compressed(
                checkpoint,
                embeddings=np.array(
                    embeddings,
                    dtype=np.float32
                ),
                files=np.array(
                    processed,
                    dtype=str
                )
            )

    # Final save
    np.savez_compressed(
        checkpoint,
        embeddings=np.array(
            embeddings,
            dtype=np.float32
        ),
        files=np.array(
            processed,
            dtype=str
        )
    )

    X = np.array(
        embeddings,
        dtype=np.float32
    )

    y = np.full(
        len(X),
        label,
        dtype=np.int64
    )

    print(
        f"\nCompleted:"
        f" {len(X)} samples"
    )

    return X, y


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    real_files,
    fake_files,
    split_name
):

    print("\n" + "=" * 65)
    print(
        f"PROCESSING {split_name.upper()}"
    )
    print("=" * 65)

    X_real, y_real = process_class(
        real_files,
        0,
        split_name,
        "real"
    )

    X_fake, y_fake = process_class(
        fake_files,
        1,
        split_name,
        "fake"
    )

    if len(X_real) == 0 or len(X_fake) == 0:

        raise RuntimeError(
            f"No usable samples in {split_name}."
        )

    X = np.vstack(
        [X_real, X_fake]
    )

    y = np.concatenate(
        [y_real, y_fake]
    )

    return X, y


# ============================================================
# EXTRACT FEATURES
# ============================================================

X_train, y_train = process_split(
    train_real,
    train_fake,
    "training"
)

X_val, y_val = process_split(
    val_real,
    val_fake,
    "validation"
)

X_test, y_test = process_split(
    test_real,
    test_fake,
    "testing"
)


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

print("\n" + "=" * 65)
print("TRAINING CLASSIFIER")
print("=" * 65)

classifier = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)

classifier.fit(
    X_train,
    y_train
)

print(
    "\nTraining completed."
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("VALIDATION RESULTS")
print("=" * 65)

val_predictions = classifier.predict(
    X_val
)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

print(
    f"\nValidation Accuracy:"
    f" {val_accuracy * 100:.2f}%"
)

print(
    classification_report(
        y_val,
        val_predictions,
        target_names=[
            "real",
            "fake"
        ]
    )
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)

test_predictions = classifier.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(
    f"\nFINAL TEST ACCURACY:"
    f" {test_accuracy * 100:.2f}%"
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "real",
            "fake"
        ]
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)

print(
    "\nConfusion Matrix:"
)

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    PROJECT_DIR
    / "deepfake_audio_classifier.joblib"
)

joblib.dump(
    classifier,
    model_path
)

print(
    f"\nModel saved:"
    f"\n{model_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 65)

print(
    f"\nTraining samples:"
    f" {len(y_train)}"
)

print(
    f"Validation samples:"
    f" {len(y_val)}"
)

print(
    f"Testing samples:"
    f" {len(y_test)}"
)

print(
    f"\nFinal test accuracy:"
    f" {test_accuracy * 100:.2f}%"
)

print(
    "\nYour first complete model is ready."
)
