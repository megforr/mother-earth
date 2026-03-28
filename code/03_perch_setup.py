"""
Biodiversity Project — Step 3: Perch 2.0 Bioacoustics Setup
============================================================
Sets up Google DeepMind's Perch model for identifying bird and wildlife
species from audio recordings.

This script:
1. Installs dependencies
2. Downloads the Perch model from Kaggle / HuggingFace
3. Runs inference on a sample audio file
4. Shows you how to batch-process recordings from the field

Dependencies (install first):
    pip install tensorflow tensorflow-hub kagglehub librosa soundfile numpy pandas

For AudioMoth recordings: https://www.openacousticdevices.info/audiomoth
For Xeno-Canto test audio: https://xeno-canto.org/

Model reference: https://github.com/google-research/perch
Kaggle model:    https://www.kaggle.com/models/google/bird-vocalization-classifier
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "audio"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Option A: Load Perch via TensorFlow Hub (simplest)
# ---------------------------------------------------------------------------

def load_perch_model_tfhub():
    """
    Load Perch using TensorFlow Hub.
    Requires: pip install tensorflow tensorflow-hub
    """
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError:
        print("Install TensorFlow: pip install tensorflow tensorflow-hub")
        return None

    print("Loading Perch model from TensorFlow Hub...")
    # Perch 2.0 model handle (bird vocalization classifier)
    MODEL_URL = "https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8"

    try:
        model = hub.load(MODEL_URL)
        print("  ✅ Perch model loaded successfully!")
        return model
    except Exception as e:
        print(f"  ❌ Could not load from TFHub: {e}")
        print("  Try downloading manually from Kaggle instead.")
        return None


# ---------------------------------------------------------------------------
# Option B: Load Perch via Kaggle Hub (recommended)
# ---------------------------------------------------------------------------

def load_perch_model_kaggle():
    """
    Load Perch using Kaggle Hub.
    Requires: pip install kagglehub tensorflow
    Requires Kaggle account and API key in ~/.kaggle/kaggle.json
    """
    try:
        import kagglehub
        import tensorflow as tf
    except ImportError:
        print("Install kagglehub: pip install kagglehub tensorflow")
        return None

    print("Downloading Perch from Kaggle...")
    try:
        path = kagglehub.model_download(
            "google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier"
        )
        print(f"  Model cached at: {path}")
        model = tf.saved_model.load(path)
        print("  ✅ Perch model loaded!")
        return model
    except Exception as e:
        print(f"  ❌ Kaggle error: {e}")
        print("  Make sure you have a Kaggle account and API key set up.")
        print("  Get your API key from: https://www.kaggle.com/settings/account")
        return None


# ---------------------------------------------------------------------------
# Audio Processing Utilities
# ---------------------------------------------------------------------------

def load_audio(audio_path, target_sr=32000):
    """
    Load an audio file and resample to 32kHz (Perch's expected sample rate).

    Args:
        audio_path: Path to .wav or .mp3 file
        target_sr:  Target sample rate (Perch uses 32000 Hz)

    Returns:
        numpy array of audio samples
    """
    try:
        import librosa
    except ImportError:
        print("Install librosa: pip install librosa")
        return None

    audio, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    print(f"  Loaded: {Path(audio_path).name} ({len(audio)/target_sr:.1f}s @ {sr}Hz)")
    return audio


def segment_audio(audio, sr=32000, window_sec=5.0, hop_sec=2.5):
    """
    Segment audio into overlapping windows for Perch inference.
    Perch processes 5-second windows.

    Args:
        audio:      1D numpy array of audio samples
        sr:         Sample rate
        window_sec: Window length in seconds
        hop_sec:    Step size between windows

    Returns:
        List of audio segments
    """
    window_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)
    segments = []
    start = 0
    while start + window_len <= len(audio):
        segments.append(audio[start:start + window_len])
        start += hop_len
    # Pad and add the last partial segment if it's > 1 second
    remaining = audio[start:]
    if len(remaining) > sr:
        padded = np.zeros(window_len)
        padded[:len(remaining)] = remaining
        segments.append(padded)
    print(f"  Segmented into {len(segments)} windows of {window_sec}s")
    return segments


def run_perch_inference(model, segments, sr=32000, top_k=5):
    """
    Run Perch inference on a list of audio segments.

    Args:
        model:    Loaded Perch TF model
        segments: List of numpy audio arrays (5s @ 32kHz each)
        sr:       Sample rate
        top_k:    Number of top species to return per segment

    Returns:
        pd.DataFrame with species predictions
    """
    import tensorflow as tf

    results = []
    for i, seg in enumerate(segments):
        # Perch expects shape [batch, samples]
        input_tensor = tf.constant(seg[np.newaxis, :], dtype=tf.float32)

        # Get logits (model output)
        output = model.signatures["serving_default"](input_features=input_tensor)

        # Logits → probabilities
        logits = output["output_0"].numpy()[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))  # softmax

        top_indices = np.argsort(probs)[::-1][:top_k]
        for rank, idx in enumerate(top_indices):
            results.append({
                "segment": i,
                "time_start_sec": i * 2.5,
                "rank": rank + 1,
                "class_index": idx,
                "probability": float(probs[idx]),
            })

    return pd.DataFrame(results)


def batch_process_recordings(model, audio_dir, output_csv):
    """
    Process all .wav files in a directory and save results.

    Args:
        model:      Loaded Perch model
        audio_dir:  Path to directory with .wav files
        output_csv: Path to output CSV
    """
    audio_dir = Path(audio_dir)
    all_results = []

    wav_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.WAV"))
    print(f"Found {len(wav_files)} audio files in {audio_dir}")

    for wav_file in wav_files:
        print(f"\nProcessing: {wav_file.name}")
        audio = load_audio(wav_file)
        if audio is None:
            continue
        segments = segment_audio(audio)
        results_df = run_perch_inference(model, segments)
        results_df["filename"] = wav_file.name
        all_results.append(results_df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(output_csv, index=False)
        print(f"\n✅ Saved inference results → {output_csv}")
        return combined
    else:
        print("No results to save.")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Perch 2.0 Bioacoustics Setup")
    print("=" * 60)

    # Try loading model (try Kaggle first, fall back to TFHub)
    model = load_perch_model_kaggle()
    if model is None:
        model = load_perch_model_tfhub()

    if model is None:
        print("\n⚠️  Could not load Perch automatically.")
        print("Manual setup instructions:")
        print("  1. Go to: https://www.kaggle.com/models/google/bird-vocalization-classifier")
        print("  2. Download the model locally")
        print("  3. Update the path in load_perch_model_kaggle() above")
    else:
        print("\n=== Batch Processing Audio Files ===")
        print(f"Place your AudioMoth .wav files in: {DATA_DIR}")
        print("Then re-run this script to process them.\n")

        # Process any .wav files already in the audio directory
        if any(DATA_DIR.glob("*.wav")):
            results = batch_process_recordings(
                model,
                DATA_DIR,
                OUTPUT_DIR / "perch_detections.csv"
            )
            print(f"\nTop detected species:")
            if not results.empty:
                print(results.groupby("class_index")["probability"].max()
                      .sort_values(ascending=False).head(20))
        else:
            print(f"No audio files found in {DATA_DIR}.")
            print("Add .wav files recorded with AudioMoth or any field recorder.")
            print("AudioMoth device: https://www.openacousticdevices.info/audiomoth (~$70)")

    print("\n✅ Perch setup complete!")
    print("Next: Run 01_fetch_species_data.py to get regional species occurrence data.")
