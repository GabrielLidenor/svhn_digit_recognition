# 🔢 Street View House Numbers (SVHN) Digit Recognition Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Data-Hugging%20Face-yellow)

A modular, production-ready deep learning pipeline engineered to automatically classify cropped house numbers from street-view imagery using the SVHN dataset. This project transitions an experimental notebook workspace into an enterprise-grade, decoupled software engineering layout.

---

## 🚀 Key Features
* **Decoupled Architecture:** Clean separation of concerns between data loading, model definition, and performance metrics.
* **Automated Data Layer:** Seamless integration with the Hugging Face Hub API—data downloads automatically on execution without bloated git histories.
* **Dual-Architecture Benchmarking:** Out-of-the-box support for both Multi-Layer Artificial Neural Networks (ANN) and deep Convolutional Neural Networks (CNN).

---

## 📂 Project Structure

```text
svhn_digit_recognition/
│
├── data/
│   ├── raw/                    # Automatically populated via Hugging Face Hub API (.h5)
│   └── processed/              # Formatted numpy partitions prepared for model consumption
│
├── notebooks/
│   └── exploration.ipynb       # Contains EDA, hyperparameter testing, and visual scratchpads
│
├── src/                        # Source core modules
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion pipelines, normalization, and one-hot mapping
│   ├── models.py               # Optimized Deep ANN and Deep CNN graph definitions
│   └── evaluate.py             # Validation tooling (Classification Reports, Confusion Matrices)
│
├── config.py                   # Centralized runtime hyperparameters and API path anchors
├── train.py                    # Root orchestrator executing the full runtime cycle
├── .gitignore                  # Keeps heavy byte arrays and runtimes local
└── requirements.txt            # Strict dependency pinning
