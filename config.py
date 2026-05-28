import os

# Paths
DATA_PATH = "gabriellidneor/svhn-digit-recognition"
HF_FILENAME = "SVHN_single_grey1.h5"

# Local Target Path (Where the API will download the file)
LOCAL_DATA_DIR = os.path.join("data", "raw")
LOCAL_DATA_PATH = os.path.join(LOCAL_DATA_DIR, HF_FILENAME)

# Training Parameters
BATCH_SIZE = 64
EPOCHS = 20
INPUT_SHAPE = (32, 32, 1)  # SVHN standard shape
NUM_CLASSES = 10
