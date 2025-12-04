import argparse
import sys
from pathlib import Path
import warnings
import os
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from model.vae import CorrelatedGaussianVAE
from utils.dataloader import ByteDataset
from config import VAE_DATASETS

import json
import logging

# ------------------------------------------------------
# Model loading
# ------------------------------------------------------

def apply_dataset_cfg_to_args(args: argparse.Namespace, ds_cfg) -> argparse.Namespace:
    # --- data paths ---
    args.train_data_path = ds_cfg.train_data_path

    # --- model shape ---
    args.latent_dim = getattr(ds_cfg.model, "latent_dim", args.latent_dim)
    # these two are only used inside model / dataloader; keep them on args
    args.num_packets = getattr(ds_cfg.model, "num_packets", getattr(args, "num_packets", None))
    args.packet_len = getattr(ds_cfg.model, "packet_len", 256)

    # --- training hyper-parameters / regularizers ---
    t = ds_cfg.train
    args.batch_size = t.batch_size
    args.num_epochs = t.num_epochs
    args.learning_rate = t.lr
    args.weight_decay = t.weight_decay

    args.use_kl_annealing = t.use_kl_annealing
    args.kld_weight_min = t.kld_weight_min
    args.kld_weight_max = t.kld_weight_max
    args.kl_anneal_period = t.kl_anneal_period
    args.kld_warmup_epochs = t.kld_warmup_epochs
    args.free_bits = t.free_bits
    args.capacity = t.capacity
    args.capacity_weight = t.capacity_weight

    args.cov_offdiag_weight = t.cov_offdiag_weight
    args.ortho_weight = t.ortho_weight

    # --- data loader / scheduler / ES ---
    args.val_split = t.val_split
    args.num_workers = t.num_workers
    args.use_scheduler = t.use_scheduler
    args.scheduler_patience = t.scheduler_patience
    args.scheduler_factor = t.scheduler_factor
    args.min_lr = t.min_lr

    args.use_early_stopping = t.use_early_stopping
    args.early_stopping_patience = t.early_stopping_patience
    args.early_stopping_min_delta = t.early_stopping_min_delta

    return args

def load_model(args: argparse.Namespace, device: torch.device, dataset_cfg: Optional[object] = None, ):
    encoder_cfg = decoder_cfg = None
    if dataset_cfg is not None:
        m = dataset_cfg.model
        encoder_cfg = {
            "packet_latent_dim": getattr(m, "packet_latent_dim", 128),
            "d_model": getattr(m, "session_d_model", 128),
            "latent_dim": args.latent_dim,
        }
        decoder_cfg = {
            "latent_dim": args.latent_dim,
            "output_size": getattr(m, "num_packets", 16) * getattr(m, "packet_len", 256),
            "rank": getattr(m, "decoder_rank", args.hidden_dim),
            "num_deep_layers": getattr(m, "decoder_num_layers", 1),
        }

    model = CorrelatedGaussianVAE(
        latent_dim=args.latent_dim,
        training_mode=args.training_mode,
        n_components=args.n_components,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout_rate,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters.")
    return model


# def load_model(args: argparse.Namespace, device: torch.device) -> CorrelatedGaussianVAE:
#     print(f"Loading model from {args.model_path}")
#     model = CorrelatedGaussianVAE(
#         latent_dim=args.latent_dim,
#         training_mode=args.training_mode,
#         n_components=5,
#         hidden_dim=256,  # keep as in your original script
#     )
#     checkpoint = torch.load(args.model_path, map_location=device)
#     if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['model_state_dict'])
#     else:
#         model.load_state_dict(checkpoint)
#
#     model.to(device)
#     model.eval()
#     print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters.")
#     return model

# ------------------------------------------------------
# Latent extraction
# ------------------------------------------------------
def extract_latent_vectors(
    model: CorrelatedGaussianVAE,
    dataloader: DataLoader,
    device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    all_mus = []
    all_labels = []
    with torch.no_grad():
        for i, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            _, mu = model.get_features(data)
            all_mus.append(mu.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            if i % 100 == 0:
                print(f"  Extracted features from batch {i + 1}/{len(dataloader)}")

    return np.concatenate(all_mus, axis=0), np.concatenate(all_labels, axis=0)


# ------------------------------------------------------
# Attack mapping + style
# ------------------------------------------------------
def load_attack_mapping(mapping_path: Optional[str]) -> Dict[int, str]:
    default_mapping = {
        0: "Benign",
        1: "ransomware",
        2: "mitm",
        3: "injection",
        4: "password",
        5: "ddos",
        6: "xss",
        7: "backdoor",
        8: "scanning",
        9: "dos"
    }

    if not mapping_path or not os.path.exists(mapping_path):
        print(f"Attack mapping not found at {mapping_path}, using default")
        return default_mapping

    try:
        with open(mapping_path, 'r') as f:
            m = json.load(f)

        # 1. Unwrap if nested (handle previous format)
        if "id_to_label" in m:
            m = m["id_to_label"]
        elif "label_to_id" in m:
            m = m["label_to_id"]

        if not m:
            return default_mapping

        # 2. Smart Detect: Check the first key
        first_key = next(iter(m))
        # SCENARIO A: Keys are Labels (e.g., "Benign": 0)
        # This matches the JSON you just pasted.
        if not str(first_key).isdigit():
            print(f"Detected Label-to-ID format (key='{first_key}'); swapping keys/values.")
            # We swap k and v.
            # k was "Benign", v was 0 -> New dict is {0: "Benign"}
            m_int = {int(v): k for k, v in m.items()}

        # SCENARIO B: Keys are IDs (e.g., "0": "Benign")
        # This matches the original code expectation.
        else:
            m_int = {int(k): v for k, v in m.items()}

        print(f"Loaded {len(m_int)} attack categories")
        return m_int

    except Exception as e:
        print(f"Error loading attack mapping: {e}")
        return default_mapping


def label_to_name(lbl: int, benign_label: int, mapping: dict[int, str]) -> str:
    if lbl == benign_label:
        return "Benign"
    if lbl in mapping:
        return mapping[lbl]
    # fallback
    return f"Attack_{lbl}"


def set_plot_style():
    base_style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot"
    plt.style.use(base_style)
    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.grid": False,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.frameon": False,
    })


# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize VAE Latent Space with t-SNE")

    # dataset from config
    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(VAE_DATASETS.keys()),
        default='cicids2018',
        help='Dataset key from VAE_DATASETS in config.py'
    )

    # model / paths; defaults can be overwritten by config
    parser.add_argument('--model_path', type=str, default=r"D:\OOD_detect\outputs\vae_cicids2018_correlated_20251129_003227\checkpoints\best_epoch_20.pth",
                        help='Path to trained VAE checkpoint (.pth)')
    parser.add_argument('--train_data_path', type=str, default=None)
    parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save the output plot (PNG)')

    # visualization
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Max number of samples per class to visualize')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Latent dimension of the model (if not found in config/checkpoint)')
    parser.add_argument('--training_mode', type=str, default='ae',
                        choices=['diagonal', 'correlated', 'gmm'],
                        help='VAE training mode')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--n_components', type=int, default=5)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    args = parser.parse_args()

    # ----------------- apply VAE_DATASETS config -----------------
    ds_cfg = VAE_DATASETS[args.dataset]

    # data paths
    if args.test_data_path is None:
        args.test_data_path = ds_cfg.test_data_path
    if args.train_data_path is None:
        args.train_data_path = ds_cfg.train_data_path

    # latent dim
    if args.latent_dim is None:
        args.latent_dim = ds_cfg.model.latent_dim

    # output path
    if args.output_file is None:
        out_dir = Path("./latent_tsne") / ds_cfg.name
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_file = str(out_dir / "latent_space_tsne-80-ae.pdf")

    # benign label + mapping
    benign_label = ds_cfg.test_benign_label
    mapping_path = getattr(ds_cfg, "attack_mapping_path", None)
    attack_mapping = load_attack_mapping(mapping_path)

    # ----------------- setup -----------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset} | benign_label={benign_label}")

    # 1. Load model
    args = apply_dataset_cfg_to_args(args, ds_cfg)
    model = load_model(args, device, ds_cfg)

    # 2. Load data
    print(f"Loading test dataset from: {args.test_data_path}")
    test_dataset = ByteDataset(args.test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Extract Latent Vectors
    print("\nExtracting latent vectors from test data...")
    test_mus, test_labels = extract_latent_vectors(model, test_loader, device)

    # 4. Balanced sampling per class (benign + each attack class)
    unique_labels = np.unique(test_labels)
    print(f"\nFound {len(unique_labels)} classes in test labels: {unique_labels}")

    all_indices = []
    all_desc_labels = []

    for lbl in unique_labels:
        idx = np.where(test_labels == lbl)[0]
        if idx.size == 0:
            continue
        k = min(args.n_samples, idx.size)
        if lbl == benign_label:
            k = 1000
        sampled_idx = np.random.choice(idx, size=k, replace=False)
        all_indices.append(sampled_idx)

        name = label_to_name(int(lbl), benign_label, attack_mapping)
        all_desc_labels.extend([name] * k)

        print(f"  Class {lbl} ({name}): using {k} / {idx.size} samples for t-SNE")

    if not all_indices:
        print("Error: no samples found for any class.")
        return

    all_indices = np.concatenate(all_indices)
    all_mus = test_mus[all_indices]
    all_desc_labels = np.array(all_desc_labels)

    # 5. Run t-SNE
    print("\nRunning t-SNE... (this may take some time)")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        random_state=args.seed,
        n_jobs=-1,
        init="pca",
        learning_rate="auto",
    )
    latent_2d = tsne.fit_transform(all_mus)

    # 6. Plot
    print("Creating plot...")
    df = pd.DataFrame(latent_2d, columns=['tsne-1', 'tsne-2'])
    df['label'] = all_desc_labels
    df['is_attack'] = df['label'] != 'Benign'

    set_plot_style()
    plt.figure(figsize=(14, 10))

    # color by class name; marker style by benign vs attack
    palette = sns.color_palette("tab20", n_colors=df['label'].nunique())
    sns.scatterplot(
        x="tsne-1",
        y="tsne-2",
        hue="label",
        style="is_attack",
        data=df,
        palette=palette,
        alpha=0.5,  # 1. [降低] 透明度，让重叠区域更明显
        s=30,  # 2. [减小] 点的大小，减少重叠
        linewidth=0,  # 3. [移除] 描边
        edgecolor=None,  # (和 linewidth=0 效果一样)
    )

    plt.title('t-SNE Visualization of VAE Latent Space', fontsize=20, pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=22, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=22, fontweight='bold')

    # put legend outside for many classes
    leg = plt.legend(
        # title='Class',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.0,
        fontsize=25,
        markerscale=3.0  # <--- 在这里添加这一行
    )
    plt.setp(leg.get_title(), fontsize=12)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"\nVisualization saved successfully to: {output_path}")
    plt.show()

if __name__ == '__main__':
    main()
