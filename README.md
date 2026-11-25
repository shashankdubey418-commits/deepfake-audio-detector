# Audio Deepfake Detection System

An AI-powered system that detects whether an audio recording is **REAL or FAKE (AI-generated or cloned)** using deep learning-based audio feature extraction and machine learning classification.

## Overview

With the increasing use of Artificial Intelligence to generate and clone human voices, distinguishing genuine audio from artificially generated speech has become increasingly important.

This project aims to address that challenge by developing an Audio Deepfake Detection System using **Wav2Vec2 and Logistic Regression**.

The system extracts meaningful representations from audio recordings using a pretrained Wav2Vec2 model and uses a trained Logistic Regression classifier to predict whether the audio is real or fake.

The project also includes a simple web interface built with Streamlit, allowing users to upload audio files and view detection results.

## Features

- **Audio Classification:** Classifies audio recordings as REAL or FAKE.
- **Wav2Vec2 Feature Extraction:** Uses a pretrained transformer-based speech model to extract audio embeddings.
- **Machine Learning Classification:** Uses Logistic Regression for binary classification.
- **Audio Preprocessing:** Converts stereo audio to mono and resamples it to 16 kHz.
- **Web Interface:** Provides an interactive interface using Streamlit.
- **Prediction Probabilities:** Displays the model's estimated probabilities for both classes.
- **WAV File Support:** Accepts WAV audio recordings as input.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning model execution |
| Wav2Vec2 | Audio feature extraction |
| Hugging Face Transformers | Loading the pretrained Wav2Vec2 model |
| Scikit-learn | Logistic Regression classifier and feature scaling |
| NumPy | Numerical operations |
| SoundFile | Reading audio files |
| SciPy | Audio resampling |
| Joblib | Saving and loading the trained classifier |
| Streamlit | Web application interface |

## How It Works

The project follows a machine learning pipeline:

1. **Audio Input:** The user uploads a WAV audio recording.
2. **Preprocessing:** The audio is converted to mono and resampled to 16 kHz.
3. **Feature Extraction:** The pretrained `facebook/wav2vec2-base` model extracts audio representations.
4. **Embedding Generation:** The extracted representations are mean-pooled to generate a fixed-size feature vector.
5. **Classification:** The feature vector is passed to the trained Logistic Regression classifier.
6. **Prediction:** The system predicts whether the recording is REAL or FAKE and displays the corresponding prediction probabilities.

## Model Architecture

The project uses two main models:

### 1. Wav2Vec2

- Pretrained model: `facebook/wav2vec2-base`
- Used for extracting meaningful representations from raw audio.
- Converts audio signals into contextual feature representations.
- Mean pooling is applied to obtain a fixed-size audio embedding.

### 2. Logistic Regression

- Used as the final binary classification model.
- Takes the extracted Wav2Vec2 embeddings as input.
- Predicts one of two classes:
  - `0` – REAL
  - `1` – FAKE

A StandardScaler is used to standardize the extracted features before classification.

## Dataset

The project uses the **Fake-or-Real (FoR) Audio Dataset**, obtained from Kaggle.

Dataset: [The Fake-or-Real Dataset on Kaggle](https://www.kaggle.com/datasets)

The dataset contains genuine and fake audio recordings that can be used to train and evaluate audio deepfake detection models.

**Note:** Download the dataset separately and organize it according to the paths expected by your training scripts. The dataset itself is not included in this repository.

## Project Structure

```text
Audio-Deepfake-Detection/
│
├── app.py
│   └── Streamlit web application
│
├── predict.py
│   └── Predicts whether an audio file is real or fake
│
├── deepfake_audio_classifier.joblib
│   └── Trained machine learning classifier
│
├── requirements.txt
│   └── Project dependencies
│
└── README.md
    └── Project documentation
```

*Add your training scripts, dataset folders, and other files to this structure if they are part of your repository.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Audio-Deepfake-Detection.git
```

### 2. Navigate to the Project Folder

```bash
cd Audio-Deepfake-Detection
```

### 3. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate it:

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install torch transformers scikit-learn numpy joblib streamlit soundfile scipy
```

### 5. Run the Web Application

```bash
streamlit run app.py
```

The application will open in your browser. Upload a WAV audio file to get a prediction.

## Running Prediction Through Terminal

You can also use the command-line prediction script:

```bash
python predict.py "path/to/audio.wav"
```

Replace the path with the location of your audio file.

## Model Performance

During one of the reported evaluation runs, the model achieved:

| Metric | Result |
|---|---:|
| Training Samples | 9,998 |
| Validation Samples | 500 |
| Testing Samples | 500 |
| Test Accuracy | 74.80% |

*These results represent a particular experimental run and may vary depending on the dataset split, preprocessing, and training configuration.*

## Limitations

- The current implementation is designed for WAV audio input.
- The classifier's performance depends on the quality and diversity of the training dataset.
- Audio recordings from sources or generators not represented in the training data may be classified less accurately.
- Prediction probabilities are model estimates and should not be interpreted as guaranteed certainty.
- This system is an experimental deepfake detection project, not a forensic verification tool.

## Future Improvements

- Improve classification accuracy through better feature extraction and model tuning.
- Experiment with additional deep learning architectures.
- Add support for more audio formats.
- Evaluate performance on unseen voice-cloning and text-to-speech systems.
- Explore real-time audio deepfake detection.
- Improve the user interface and provide more detailed prediction reports.

## Author

**Shashank Dubey**

B.Tech  Computer Science and Engineering

## Disclaimer

This project is developed for educational and research purposes. Its predictions should not be treated as definitive proof that an audio recording is genuine or manipulated.
