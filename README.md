# CAVAD: Correlation-Aware Variational Autoencoder for Zero-Day Network Anomaly Detection

This repository contains the reference implementation of **CAVAD** (Correlation-Aware Variational Autoencoder with Cross-Attention) for **zero-day network anomaly detection** from raw packet bytes.

CAVAD is an unsupervised framework: it is trained only on benign traffic and detects attacks as **out-of-distribution (OOD)** samples in a learned latent space.

---

## Key ideas

- **Raw packet modeling**  
  Works directly on packet bytes instead of hand-crafted flow features, preserving fine-grained header and payload information.

- **Correlation-aware VAE**  
  Uses a **full-covariance Gaussian** posterior in the latent space to capture dependencies between latent dimensions, instead of assuming independence.

- **Header–payload cross-attention**  
  A **bidirectional cross-attention module** enforces semantic consistency between protocol headers and payloads, helping to detect masquerading traffic.

- **Mahalanobis-based anomaly score**  
  Combines reconstruction error with a **Mahalanobis distance** in the latent space to better separate benign and malicious traffic.

---

## Datasets

The experiments in the paper are based on three public intrusion-detection datasets:

- **CIC-IDS2017**
- **CSE-CIC-IDS2018**
- **TON\_IoT**

Each dataset is preprocessed from raw PCAPs into fixed-length packet sequences (sessions). See the paper for full preprocessing details.

---

## Project structure (typical)

A common layout for this project is:

- `preprocess/` – scripts to parse PCAPs, build sessions, pad/truncate packets, and export tensors.
- `models/` – CAVAD model components (packet encoder, session encoder, VAE, decoder).
- `configs/` – dataset/model/training configuration files (e.g., YAML/JSON).
- `train_cavad.py` – training entry point.
- `eval_cavad.py` – evaluation and anomaly scoring.
- `utils/` – helper functions (logging, metrics, dataset loaders, etc.).

> Adjust names and paths above to match your actual repository layout.

---

## Getting started

### 1. Install dependencies

Create a virtual environment (recommended) and install packages:

```bash
pip install -r requirements.txt
