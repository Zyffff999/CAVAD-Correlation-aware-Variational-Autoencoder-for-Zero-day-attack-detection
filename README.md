# CAVAD: Correlation-Aware Variational Autoencoder for Zero-Day Network Anomaly Detection

**CoNEXT Conference Presentation**

This repository contains the reference implementation of **CAVAD** (Correlation-Aware Variational Autoencoder with Cross-Attention) for **zero-day network anomaly detection** from raw packet bytes.

CAVAD is an unsupervised framework: it is trained only on benign traffic and detects attacks as **out-of-distribution (OOD)** samples in a learned latent space.

---

## Table of Contents

- [Key Ideas](#key-ideas)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Datasets](#datasets)
- [Citation](#citation)

---

## Key Ideas

- **Raw packet modeling**
  Works directly on packet bytes instead of hand-crafted flow features, preserving fine-grained header and payload information.

- **Correlation-aware VAE**
  Uses a **full-covariance Gaussian** posterior in the latent space to capture dependencies between latent dimensions, instead of assuming independence.

- **Header–payload cross-attention**
  A **bidirectional cross-attention module** enforces semantic consistency between protocol headers and payloads, helping to detect masquerading traffic.

- **Mahalanobis-based anomaly score**
  Combines reconstruction error with a **Mahalanobis distance** in the latent space to better separate benign and malicious traffic.

---

## Project Structure

```
CAVAD/
├── Data_Preprocess/           # Data preprocessing scripts
│   ├── IDS_2017_TO_Mong.py   # CIC-IDS2017 data preprocessing
│   ├── IDS_2018_TO_Mong.py   # CSE-CIC-IDS2018 data preprocessing
│   └── TON_IOT/               # TON_IoT dataset preprocessing
│       ├── IOT_to_Mong.py
│       └── benign_pcap_processor.py
├── model/                     # CAVAD model components
│   ├── vae.py                # Main VAE architecture
│   ├── packet.py             # Packet encoder/decoder
│   ├── cnn.py                # CNN backbone
│   └── losses.py             # Loss functions
├── training/                  # Training utilities
│   ├── trainer.py            # VAE trainer class
│   └── callbacks.py          # Training callbacks
├── evaluation/                # Evaluation utilities
│   ├── metrics.py            # Evaluation metrics
│   ├── anomaly_detection.py  # Anomaly scoring methods
│   └── visualization.py      # Result visualization
├── utils/                     # Helper utilities
│   ├── data_utils.py         # Data processing utilities
│   ├── dataloader.py         # PyTorch dataloaders
│   └── general_utils.py      # General utilities
├── main.py                    # Main training script
├── test.py                    # Model evaluation script
├── vis_latent.py             # Latent space visualization
├── data_preprocess.py        # Main data preprocessing script
└── config.py                 # Configuration for all datasets
```

---

## File Descriptions

### Core Training & Evaluation

- **`main.py`**
  Main training script for CAVAD model. Supports multiple datasets (CIC-IDS2017, CSE-CIC-IDS2018, TON_IoT) and training modes (diagonal, correlated, GMM, autoencoder). Includes KL annealing, early stopping, and learning rate scheduling.

- **`test.py`**
  Comprehensive evaluation script for trained models. Computes multiple anomaly scores (reconstruction error, Mahalanobis distance, combined score, KL divergence) and generates detailed reports with per-category analysis and visualizations.

- **`config.py`**
  Centralized configuration file containing dataset paths, model architectures, and training hyperparameters for all three datasets. Uses dataclass-based configurations for clarity and maintainability.

### Data Preprocessing

- **`data_preprocess.py`**
  Main data preprocessing pipeline for TON_IoT dataset. Implements stratified sampling, benign/attack separation, and session construction from MongoDB collections. Generates NPZ files with packet headers, payloads, and labels.

- **`Data_Preprocess/IDS_2017_TO_Mong.py`**
  Preprocesses CIC-IDS2017 PCAP files into MongoDB format. Extracts packet headers and payloads, creates sessions, and stores metadata.

- **`Data_Preprocess/IDS_2018_TO_Mong.py`**
  Preprocesses CSE-CIC-IDS2018 PCAP files into MongoDB format with similar functionality to IDS_2017_TO_Mong.py.

- **`Data_Preprocess/TON_IOT/IOT_to_Mong.py`**
  Converts TON_IoT PCAP files to MongoDB collections with session-based organization.

- **`Data_Preprocess/TON_IOT/benign_pcap_processor.py`**
  Specialized processor for benign TON_IoT traffic with quality filtering and validation.

### Model Components

- **`model/vae.py`**
  Core CorrelatedGaussianVAE implementation supporting multiple training modes (diagonal, correlated, GMM, autoencoder). Implements full-covariance Gaussian posterior with Cholesky decomposition and analytical KL divergence.

- **`model/packet.py`**
  Implements PacketEncoder (TCN-based with cross-attention), SessionEncoder (temporal aggregation), and FactorizedDecoder (low-rank reconstruction) for processing packet sequences.

- **`model/cnn.py`**
  CNN backbone components including temporal convolutional networks (TCN) with SE blocks for feature extraction.

- **`model/losses.py`**
  Custom loss functions including reconstruction losses, KL divergence variants, and regularization terms.

### Training Infrastructure

- **`training/trainer.py`**
  VAETrainer class handling training loops, validation, anomaly score computation (reconstruction, Mahalanobis, whitened L2, combined, KL), and reference statistics calculation.

- **`training/callbacks.py`**
  Training callbacks including EarlyStopping, ModelCheckpoint, LearningRateMonitor, GradientMonitor, MetricTracker, ProgressBar, and WarmupScheduler.

### Evaluation & Visualization

- **`evaluation/metrics.py`**
  Evaluation metrics computation including optimal threshold selection, FPR-based thresholding, and performance metrics (accuracy, precision, recall, F1, AUC).

- **`evaluation/anomaly_detection.py`**
  Anomaly detection methods implementing different scoring approaches.

- **`evaluation/visualization.py`**
  Visualization utilities for results, distributions, and latent space analysis.

- **`vis_latent.py`**
  Standalone script for t-SNE visualization of VAE latent space. Generates publication-quality plots showing separation between benign and attack traffic.

### Utilities

- **`utils/dataloader.py`**
  PyTorch Dataset and DataLoader implementations for efficient loading of preprocessed NPZ files with optional data augmentation.

- **`utils/data_utils.py`**
  Data processing utilities including train/validation splitting, normalization, and session construction.

- **`utils/general_utils.py`**
  General utilities including random seed setting, experiment directory management, logging setup, and GPU memory monitoring.

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- MongoDB (for data preprocessing)

### Install Dependencies

Create a virtual environment (recommended) and install packages:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 1.12.0
- NumPy
- scikit-learn
- pymongo (for data preprocessing)
- tqdm
- matplotlib
- seaborn
- pandas

---

## Usage

### 1. Data Preprocessing

First, preprocess your raw PCAP files into the required format. Update the dataset paths in `config.py` to match your local setup.

```bash
# For TON_IoT dataset
python data_preprocess.py

# For CIC-IDS2017 dataset
python Data_Preprocess/IDS_2017_TO_Mong.py

# For CSE-CIC-IDS2018 dataset
python Data_Preprocess/IDS_2018_TO_Mong.py
```

The preprocessing scripts will:
1. Read raw PCAP files or MongoDB collections
2. Extract packet headers and payloads
3. Construct sessions (sequences of packets)
4. Apply padding/truncation to fixed lengths
5. Generate NPZ files containing:
   - `headers`: Packet headers (shape: [N, num_packets, header_size])
   - `payloads`: Packet payloads (shape: [N, num_packets, payload_size])
   - `payload_masks`: Valid payload indicators
   - `labels`: Traffic labels (0 for benign, >0 for attacks)
   - `stats_features`: Statistical features

### 2. Training

Train the CAVAD model on benign traffic:

```bash
# Train on CIC-IDS2017 with correlated mode (recommended)
python main.py --dataset cicids2017 --training_mode correlated

# Train on CSE-CIC-IDS2018
python main.py --dataset cicids2018 --training_mode correlated

# Train on TON_IoT
python main.py --dataset ton_iot --training_mode correlated

# Other training modes
python main.py --dataset cicids2017 --training_mode diagonal    # Diagonal covariance
python main.py --dataset cicids2017 --training_mode gmm         # Gaussian Mixture Model
python main.py --dataset cicids2017 --training_mode ae          # Autoencoder (no KL)
```

**Key Arguments:**
- `--dataset`: Dataset name (cicids2017, cicids2018, ton_iot)
- `--training_mode`: VAE mode (diagonal, correlated, gmm, ae)
- `--latent_dim`: Latent space dimension (default: 64)
- `--batch_size`: Batch size (default: 128)
- `--num_epochs`: Number of epochs (default: 50-80 depending on dataset)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--output_dir`: Output directory for checkpoints and logs
- `--experiment_name`: Custom experiment name

The training script will:
- Automatically load dataset configurations from `config.py`
- Create an experiment directory with timestamps
- Save checkpoints periodically and when validation loss improves
- Log training metrics (loss, reconstruction error, KL divergence)
- Apply KL annealing and early stopping

**Training Output:**
```
outputs/
└── vae_cicids2017_correlated_20231204_120000/
    ├── checkpoints/
    │   ├── best_epoch_45.pth
    │   └── epoch_50.pth
    ├── logs/
    │   ├── training.log
    │   ├── metrics.json
    │   └── lrs.txt
    └── config.json
```

### 3. Evaluation

Evaluate trained models on test data with multiple anomaly scoring methods:

```bash
# Basic evaluation
python test.py \
    --dataset cicids2017 \
    --model_path outputs/vae_cicids2017_correlated_20231204_120000/checkpoints/best_epoch_45.pth \
    --output_dir test_results/cicids2017

# Evaluate with custom FPR target
python test.py \
    --dataset cicids2017 \
    --model_path outputs/vae_cicids2017_correlated_20231204_120000/checkpoints/best_epoch_45.pth \
    --fpr_target 0.01 \
    --output_dir test_results/cicids2017_fpr001

# Evaluate specific anomaly scoring methods
python test.py \
    --dataset cicids2017 \
    --model_path path/to/checkpoint.pth \
    --methods reconstruction mahalanobis combined
```

**Available Anomaly Scoring Methods:**
- `reconstruction`: Reconstruction error (MSE between input and output)
- `mahalanobis`: Mahalanobis distance in latent space
- `whitened_l2`: L2 distance in whitened latent space
- `combined`: Weighted combination of reconstruction and Mahalanobis
- `kl`: KL divergence between posterior and prior

**Evaluation Output:**
```
test_results/cicids2017/
├── evaluation_report.txt              # Summary report
├── evaluation_results.json            # Detailed JSON results
├── method_comparison.csv              # Method comparison table
├── method_reconstruction_fpr_sweep.csv # FPR sweep for reconstruction
├── method_mahalanobis_fpr_sweep.csv   # FPR sweep for Mahalanobis
├── reconstruction/                     # Per-method results
│   ├── scores_distribution.png
│   ├── per_category_distribution.png
│   ├── per_category_accuracy.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── score_timeline.png
│   └── threshold_analysis.png
└── mahalanobis/
    └── ...
```

### 4. Latent Space Visualization

Visualize the learned latent space using t-SNE:

```bash
python vis_latent.py \
    --dataset cicids2017 \
    --model_path outputs/vae_cicids2017_correlated_20231204_120000/checkpoints/best_epoch_45.pth \
    --n_samples 1000 \
    --output_file latent_tsne/cicids2017/latent_space.png
```

This generates a t-SNE plot showing how benign and attack traffic are separated in the latent space.

---

## Data Preprocessing

### Acknowledgment

The IDS series data preprocessing approach in this repository is **inspired by and adapted from** the PBCNN project:

**PBCNN Repository**: [https://github.com/sspku-2021/PBCNN](https://github.com/sspku-2021/PBCNN)

We acknowledge the PBCNN authors for their pioneering work on packet-level preprocessing for network intrusion detection. Our preprocessing pipeline builds upon their ideas of:
- Converting raw PCAP files to packet byte sequences
- Separating packet headers and payloads
- Session-based traffic organization
- Fixed-length packet and session representations

**Key Differences in Our Implementation:**
1. **Enhanced session construction** with stratified sampling for balanced datasets
2. **MongoDB integration** for efficient storage and querying of large-scale packet data
3. **Multi-dataset support** (CIC-IDS2017, CSE-CIC-IDS2018, TON_IoT) with dataset-specific optimizations
4. **Preprocessing configuration** through Python dataclasses for reproducibility
5. **Statistical feature extraction** alongside raw packet bytes

### Preprocessing Pipeline

The preprocessing consists of three main stages:

**Stage 1: PCAP to MongoDB**
- Parse raw PCAP files using libraries like `scapy` or `dpkt`
- Extract packet headers (IP, TCP/UDP headers)
- Extract packet payloads
- Store in MongoDB collections with metadata (timestamps, protocols, labels)

**Stage 2: Session Construction**
- Group packets into bidirectional flows (5-tuple: src_ip, dst_ip, src_port, dst_port, protocol)
- Create sessions with fixed number of packets (default: 16 packets per session)
- Apply padding/truncation to standardize session lengths
- Separate benign and attack traffic based on labels

**Stage 3: NPZ Export**
- Convert sessions to NumPy arrays
- Apply stratified sampling for balanced training/test splits
- Export to compressed NPZ format for efficient loading
- Generate label mappings and metadata files

### Data Format Specifications

**Input Format (PCAP):**
- Standard PCAP/PCAPNG files
- Ground truth labels from dataset documentation

**Intermediate Format (MongoDB):**
```json
{
  "_id": ObjectId("..."),
  "key": {
    "src_ip": "192.168.1.1",
    "dst_ip": "10.0.0.1",
    "src_port": 443,
    "dst_port": 52341,
    "proto": "TCP"
  },
  "p_header": ["020000...", "020000...", ...],    // Hex strings
  "p_payload": ["474554...", "485454...", ...],   // Hex strings
  "packet_count": 16,
  "payload_length": 1024,
  "label": "Benign",
  "timestamp": ISODate("...")
}
```

**Output Format (NPZ):**
```python
{
  'headers': np.ndarray,        # shape: [N, 16, 128], dtype: uint8
  'payloads': np.ndarray,       # shape: [N, 16, 128], dtype: uint8
  'payload_masks': np.ndarray,  # shape: [N, 16], dtype: bool
  'labels': np.ndarray,         # shape: [N,], dtype: int32
  'stats_features': np.ndarray  # shape: [N, 20], dtype: float32
}
```

Where:
- `N`: Number of sessions
- `16`: Number of packets per session
- `128`: Bytes per packet header/payload
- `20`: Number of statistical features

---

## Datasets

The experiments in this repository are conducted on three public intrusion-detection datasets:

### 1. CIC-IDS2017

**Source**: Canadian Institute for Cybersecurity
**Description**: Network traffic dataset containing benign and common attack scenarios
**Attack Types**: Brute Force, Heartbleed, Botnet, DoS, DDoS, Web Attack, Infiltration
**Traffic Period**: 5 days (Monday to Friday)
**Total Flows**: ~2.8 million labeled flows

**Configuration in `config.py`:**
- Training: Benign traffic only
- Testing: Mixed benign and attack traffic
- Session format: 16 packets × (128-byte header + 128-byte payload)
- Latent dimension: 64

### 2. CSE-CIC-IDS2018

**Source**: Communications Security Establishment & Canadian Institute for Cybersecurity
**Description**: Collaborative network intrusion detection dataset
**Attack Types**: Brute Force, DoS, DDoS, Web Attack, Infiltration, Botnet
**Traffic Period**: 10 days
**Total Flows**: ~16 million labeled flows

**Configuration in `config.py`:**
- Training: Benign traffic only
- Testing: Mixed benign and attack traffic
- Session format: 16 packets × (128-byte header + 128-byte payload)
- Latent dimension: 64

### 3. TON_IoT

**Source**: University of New South Wales (UNSW)
**Description**: IoT and IIoT network traffic dataset
**Attack Types**: DDoS, DoS, Ransomware, Backdoor, Injection, XSS, Password, Scanning, MITM
**Network Type**: IoT devices, network services
**Total Records**: ~300 GB of raw data

**Configuration in `config.py`:**
- Training: Pure benign traffic from dedicated benign collection
- Testing: Benign + filtered attack traffic
- Session format: 16 packets × (128-byte header + 128-byte payload)
- Latent dimension: 32 (smaller due to IoT traffic characteristics)

### Dataset Access

Due to size constraints, **preprocessed datasets are not included in this repository**.

**📦 Processed Data Availability**

The preprocessed NPZ files for all three datasets will be made available upon request or after paper acceptance. The processed data includes:

- ✅ Training sets (benign traffic only)
- ✅ Test sets (mixed benign and attack traffic)
- ✅ Label mappings and metadata
- ✅ Reference statistics for anomaly detection

**To obtain the processed data:**
1. Contact the authors at [your_email@domain.com] or
2. Download from [data repository link - to be added] after publication

**Expected data structure:**
```
data/
├── IDS_2017_processed/
│   └── sessions_16_h128_p128/
│       ├── train_benign_ready.npz
│       ├── test_mixed_ready.npz
│       ├── label_map.json
│       └── metadata.json
├── IDS_2018_processed/
│   └── Session-16-h128-p128/
│       ├── train.npz
│       ├── test.npz
│       ├── label_map.json
│       └── metadata.json
└── TON_IOT/
    └── Session-16-h128-p128/
        ├── train.npz
        ├── test.npz
        ├── label_map.json
        └── metadata.json
```

**If you wish to preprocess the datasets yourself:**
1. Download raw datasets from official sources:
   - CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
   - CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
   - TON_IoT: https://research.unsw.edu.au/projects/toniot-datasets
2. Set up MongoDB instance
3. Run preprocessing scripts as described in [Usage](#usage) section
4. Update paths in `config.py`

---

## Model Architecture

CAVAD consists of three main components:

### 1. Packet Encoder
- **Input**: Raw packet bytes (header + payload)
- **Architecture**:
  - Byte embedding layer (vocab_size=257, dim=128)
  - Parallel TCN branches for headers and payloads
  - Bidirectional cross-attention between header and payload features
  - Spatial reduction via adaptive pooling

### 2. Session Encoder
- **Input**: Sequence of packet-level features
- **Architecture**:
  - Temporal convolution across packet sequence
  - Global aggregation (mean/max pooling)
  - Projection to latent space parameters (μ, Σ)

### 3. Decoder
- **Input**: Latent vector z
- **Architecture**:
  - Low-rank factorized decoder for efficiency
  - Residual blocks
  - Reconstruction of flattened packet sequence

**Training Modes:**
- **Diagonal**: Independent Gaussian dimensions (β-VAE style)
- **Correlated**: Full-covariance Gaussian (CAVAD)
- **GMM**: Gaussian Mixture Model posterior
- **AE**: Autoencoder mode (no KL divergence)

---

## Performance

**Results on CIC-IDS2017 (Zero-day detection):**

| Method | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| CAVAD (Correlated) | 94.2% | 93.8% | 94.6% | 94.2% | 0.982 |
| CAVAD (Diagonal) | 92.1% | 91.5% | 92.8% | 92.1% | 0.971 |
| Baseline AE | 88.4% | 87.2% | 89.7% | 88.4% | 0.952 |

**Per-Attack Type Detection Rates (CAVAD Correlated):**
- DoS/DDoS: 98.3%
- Web Attack: 95.7%
- Brute Force: 93.4%
- Infiltration: 91.2%
- Botnet: 89.8%

*Note: Detailed results on all three datasets available in the paper.*

---

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@inproceedings{cavad2024,
  title={CAVAD: Correlation-Aware Variational Autoencoder for Zero-Day Network Anomaly Detection},
  author={[Author Names]},
  booktitle={Proceedings of the International Conference on emerging Networking EXperiments and Technologies (CoNEXT)},
  year={2024}
}
```

---

## Acknowledgments

- Data preprocessing approach inspired by **PBCNN**: [https://github.com/sspku-2021/PBCNN](https://github.com/sspku-2021/PBCNN)
- Dataset providers: CIC (UNB), UNSW
- PyTorch and scientific Python community

---

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

---

## Contact

For questions, issues, or collaboration opportunities:
- **Email**: [your_email@domain.com]
- **GitHub Issues**: [https://github.com/your-username/CAVAD/issues](https://github.com/your-username/CAVAD/issues)

---

## Troubleshooting

### Common Issues

**Q: MongoDB connection error during preprocessing**
- Ensure MongoDB is running: `sudo systemctl start mongod`
- Check connection string in `config.py` or environment variable `MONGO_URI`

**Q: CUDA out of memory during training**
- Reduce `--batch_size` (try 64 or 32)
- Reduce `--latent_dim` or model capacity
- Enable gradient checkpointing (modify `training/trainer.py`)

**Q: Poor detection performance**
- Ensure model is trained on benign-only data
- Check dataset balance (benign vs attack ratio in test set)
- Try different `--fpr_target` values for evaluation
- Experiment with different training modes (correlated vs diagonal)

**Q: Preprocessing script hangs or is very slow**
- MongoDB indexing: Create index on `label` field
- Reduce `--n_samples` for faster preprocessing
- Use `allowDiskUse=True` for large aggregation queries

---

**Last Updated**: December 2024
**Version**: 1.0.0
**Status**: CoNEXT Conference Submission
