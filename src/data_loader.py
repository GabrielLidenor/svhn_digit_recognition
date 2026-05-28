import os
import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical
from huggingface_hub import hf_hub_download
import config

def load_and_preprocess_data():
    """Checks for local data. If missing, automatically downloads it from

    Hugging Face via API, then processes it for training.
    """

    # 1. Automatic API Fetching Layer
    if not os.path.exists(config.LOCAL_DATA_PATH):
        print(f"Dataset not found locally at {config.LOCAL_DATA_PATH}")
        print("📥 Fetching dataset directly from Hugging Face Hub via API...")

        os.makedirs(config.LOCAL_DATA_DIR, exist_ok=True)

        # This downloads the file from your HF dataset repo straight to your computer
        hf_hub_download(
            repo_id=config.HF_REPO_ID,
            filename=config.HF_FILENAME,
            repo_type="dataset",
            local_dir=config.LOCAL_DATA_DIR
        )
        print("✅ Download complete!")

    # 2. Standard Data Processing Layer
    print("Parsing dataset file...")
    with h5py.File(config.LOCAL_DATA_PATH, "r") as h5f:
        X_train = h5f['X_train'][:]
        y_train = h5f['y_train'][:]
        X_test = h5f['X_test'][:]
        y_test = h5f['y_test'][:]

    # Scale pixel values to [0, 1] range
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Reshape for CNN input channel requirements
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # One-hot encode target labels
    y_train = to_categorical(y_train, num_classes=config.NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=config.NUM_CLASSES)

    return X_train, y_train, X_test, y_test
