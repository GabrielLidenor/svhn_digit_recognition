# 🔢 Street View House Numbers (SVHN) Digit Recognition Pipeline

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Data-Hugging%20Face-yellow)
![Tests](https://img.shields.io/badge/Tests-PyTest-green.svg)

A modular, production-grade computer vision pipeline engineered to classify cropped house numbers from the SVHN dataset. Built with clean software engineering principles, object-oriented model architectures, automated data ingestion, and rigorous unit testing.

---

## 🚀 Key Features

* **Modular OOP Architecture:** Clean inheritance hierarchy using `BaseSVHNModel` to decouple model definitions, hyperparameters, and training logic.
* **CLI-Driven Pipeline:** Easily toggle between architectures (`--model cnn|ann`) and run end-to-end training via parameterized execution.
* **Automated Data Layer:** Seamless integration with the Hugging Face Hub API—data streams automatically on demand without committing large datasets to Git.
* **4-Pillar ML Unit Testing:** Built-in `pytest` suite validating tensor shape geometry, probability contracts, single-sample gradient flow, and pipeline execution.
* **Artifact Management:** Automated folder creation and saving of compiled `.keras` models while maintaining a clean Git history.

---

## 📂 Project Structure

```text
svhn_digit_recognition/
├── artifacts/
│   └── models/               # Saved trained model binaries (.keras)
├── data/
│   └── raw/                  # Automatically populated via HF Hub API (.h5)
├── src/
│   ├── models/               # Decoupled model architectures
│   │   ├── __init__.py       # Exposes SVHNANN and SVHNCNN cleanly
│   │   ├── base.py           # Abstract Base Class (BaseSVHNModel)
│   │   ├── ann.py            # Deep Artificial Neural Network implementation
│   │   └── cnn.py            # Convolutional Neural Network implementation
│   ├── data_loader.py        # Data ingestion, normalization, and one-hot mapping
│   └── evaluate.py           # Evaluation tools (classification reports & matrices)
├── tests/                    # Fast offline test suite (uses dummy tensors)
│   ├── conftest.py           # Shared PyTest fixtures
│   ├── test_ann.py           # ANN shape, probability, & compilation tests
│   └── test_cnn.py           # CNN geometry & single-sample overfit tests
├── config.py                 # Centralized hyperparameters and path configs
├── train.py                  # CLI pipeline orchestrator
├── requirements-dev.txt      # Pinned development & testing dependencies
└── requirements.txt          # Production runtime dependencies
```
## ⚡ Setup & Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/GabrielLidenor/svhn_digit_recognition.git](https://github.com/GabrielLidenor/svhn_digit_recognition.git)
cd svhn_digit_recognition
```

## 2. Verify Python Version
Ensure you have Python 3.11 or 3.12 installed:

```bash
python3.12 --version
# Output should be Python 3.12.x (or 3.11.x)
```

## 3. Create & Activate Virtual Environment

```bash
# Create environment using Python 3.12
python3.12 -m venv .venv

# Activate on macOS/Linux:
source .venv/bin/activate

# Activate on Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

## 4. Install Dependencies

```bash
# Upgrade pip inside the virtual environment
python -m pip install --upgrade pip

# Install development & model dependencies
pip install -r requirements-dev.txt
```

## 🧪 Unit Testing Strategy

```bash
python -m pytest
```

Following modern ML engineering standards (Breck et al., Google Research, 2017), our test suite covers four core ML testing pillars:

* Shape Geometry: Validates tensor transformation across layers to match expected output dimensions (batch_size, num_classes).
* Probability Contracts: Asserts output tensors contain no NaN/Inf values and sum to $1.0$ across predictions.
* Gradient Flow: Executes single-sample overfitting routines to verify backpropagation and weight updates.
* Integration Steps: Performs trial train_on_batch steps to check compilation and loss/optimizer compatibility.

## 🏃 Running the Training Pipeline

```bash
# Train default Convolutional Neural Network (CNN)
python train.py --model cnn

# Train baseline Artificial Neural Network (ANN)
python train.py --model ann
```

Trained binaries will automatically be exported and saved inside artifacts/models/svhn_<model_type>.keras.

## 📊 Model Benchmarks

| Model Architecture | Input Shape | Params | Test Accuracy |
| :--- | :---: | :---: | :---: |
| **ANN (Baseline)** | `(32, 32, 3)` | ~800K | TBD |
| **CNN (Conv2D + BN)** | `(32, 32, 3)` | ~250K | TBD |

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
