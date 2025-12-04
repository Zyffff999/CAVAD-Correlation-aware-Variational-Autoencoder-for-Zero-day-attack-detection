# CAVAD: Correlation-Aware Variational Autoencoder for Zero-Day Network Anomaly Detection

This repository contains the implementation of **CAVAD** (Correlation-Aware Variational Autoencoder with Cross-Attention) for zero-day network anomaly detection from raw packet bytes.

CAVAD is an unsupervised framework trained only on benign traffic and detects attacks as out-of-distribution (OOD) samples in a learned latent space.

---

## Key Ideas

- **Raw packet modeling**: Works directly on packet bytes instead of hand-crafted flow features, preserving fine-grained header and payload information.

- **Correlation-aware VAE**: Uses a full-covariance Gaussian posterior in the latent space to capture dependencies between latent dimensions, instead of assuming independence.

- **Header-payload cross-attention**: A bidirectional cross-attention module enforces semantic consistency between protocol headers and payloads, helping to detect masquerading traffic.

- **Mahalanobis-based anomaly score**: Combines reconstruction error with a Mahalanobis distance in the latent space to better separate benign and malicious traffic.

---

## Project Architecture

```
CAVAD/
├── Data_Preprocess/           # Data preprocessing scripts
├── model/                     # CAVAD model components
├── training/                  # Training utilities
├── evaluation/                # Evaluation utilities
├── utils/                     # Helper utilities
├── main.py                    # Main training script
├── test.py                    # Model evaluation script
├── vis_latent.py             # Latent space visualization
├── data_preprocess.py        # Main data preprocessing script
└── config.py                 # Configuration for all datasets
```

### File Descriptions

**Core Scripts:**
- `main.py`: Main training script for CAVAD model
- `test.py`: Comprehensive evaluation script for trained models
- `config.py`: Centralized configuration file containing dataset paths and model hyperparameters
- `data_preprocess.py`: Main data preprocessing pipeline
- `vis_latent.py`: t-SNE visualization of VAE latent space

**Data Preprocessing:**
- `Data_Preprocess/IDS_2017_TO_Mong.py`: Preprocesses CIC-IDS2017 PCAP files into MongoDB format
- `Data_Preprocess/IDS_2018_TO_Mong.py`: Preprocesses CSE-CIC-IDS2018 PCAP files into MongoDB format
- `Data_Preprocess/TON_IOT/IOT_to_Mong.py`: Converts TON_IoT PCAP files to MongoDB collections
- `Data_Preprocess/TON_IOT/benign_pcap_processor.py`: Specialized processor for benign TON_IoT traffic

**Model Components:**
- `model/vae.py`: Core CorrelatedGaussianVAE implementation with multiple training modes
- `model/packet.py`: PacketEncoder, SessionEncoder, and FactorizedDecoder implementations
- `model/cnn.py`: CNN backbone components including temporal convolutional networks
- `model/losses.py`: Custom loss functions for VAE training

**Training Infrastructure:**
- `training/trainer.py`: VAETrainer class handling training loops and anomaly score computation
- `training/callbacks.py`: Training callbacks (EarlyStopping, ModelCheckpoint, etc.)

**Evaluation & Visualization:**
- `evaluation/metrics.py`: Evaluation metrics computation including threshold selection
- `evaluation/anomaly_detection.py`: Anomaly detection methods with different scoring approaches
- `evaluation/visualization.py`: Visualization utilities for results and latent space analysis

**Utilities:**
- `utils/dataloader.py`: PyTorch Dataset and DataLoader implementations
- `utils/data_utils.py`: Data processing utilities including train/validation splitting
- `utils/general_utils.py`: General utilities including logging and experiment management

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- MongoDB (for data preprocessing)

---

## Usage

### Training (main.py)

Train the CAVAD model on benign traffic using `main.py`:

```bash
# Train on CIC-IDS2017 with correlated mode
python main.py --dataset cicids2017 --training_mode correlated

# Train on CSE-CIC-IDS2018
python main.py --dataset cicids2018 --training_mode correlated

# Train on TON_IoT
python main.py --dataset ton_iot --training_mode correlated
```

**Key Arguments:**
- `--dataset`: Dataset name (cicids2017, cicids2018, ton_iot)
- `--training_mode`: VAE mode (diagonal, correlated, gmm, ae)
- `--latent_dim`: Latent space dimension (default: 64)
- `--batch_size`: Batch size (default: 128)
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate (default: 1e-5)
- `--output_dir`: Output directory for checkpoints and logs

The script will automatically load dataset configurations from `config.py`, create an experiment directory, and save checkpoints when validation loss improves.

### Evaluation (test.py)

Evaluate trained models on test data using `test.py`:

```bash
# Basic evaluation
python test.py \
    --dataset cicids2017 \
    --model_path outputs/your_model/checkpoints/best_epoch_45.pth \
    --output_dir test_results/cicids2017

# Evaluate with custom FPR target
python test.py \
    --dataset cicids2017 \
    --model_path outputs/your_model/checkpoints/best_epoch_45.pth \
    --fpr_target 0.01 \
    --output_dir test_results/cicids2017_fpr001
```

**Key Arguments:**
- `--dataset`: Dataset name (same as training)
- `--model_path`: Path to trained model checkpoint
- `--methods`: Anomaly scoring methods (reconstruction, mahalanobis, whitened_l2, combined, kl)
- `--fpr_target`: Target false positive rate (default: 0.1)
- `--output_dir`: Output directory for evaluation results

The evaluation script computes multiple anomaly scores and generates detailed reports with visualizations including ROC curves, confusion matrices, and per-category analysis.

---

## Data Preprocessing

The data preprocessing approach in this repository is based on the methodology from the **PBCNN project**: [https://github.com/sspku-2021/PBCNN](https://github.com/sspku-2021/PBCNN)

We use the preprocessing ideas from PBCNN for converting raw PCAP files to packet byte sequences and session-based representations for CICIDS2017 and 2018.

Other you can used the preprocessed data in ......

## Datasets

The experiments are conducted on three public intrusion-detection datasets:

**CIC-IDS2017**
- Network traffic dataset with benign and common attack scenarios
- Attack types: Brute Force, Heartbleed, Botnet, DoS, DDoS, Web Attack, Infiltration

**CSE-CIC-IDS2018**
- Collaborative network intrusion detection dataset
- Attack types: Brute Force, DoS, DDoS, Web Attack, Infiltration, Botnet

**TON_IoT**
- IoT and IIoT network traffic dataset
- Attack types: DDoS, DoS, Ransomware, Backdoor, Injection, XSS, Password, Scanning, MITM

---


