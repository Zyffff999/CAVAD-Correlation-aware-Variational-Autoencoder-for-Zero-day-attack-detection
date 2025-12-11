import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)

import matplotlib.pyplot as plt
import seaborn as sns

from model.vae import CorrelatedGaussianVAE
from training.trainer import VAETrainer

from utils.dataloader import ByteDataset
from evaluation.metrics import compute_threshold_fpr, compute_optimal_threshold

from config import VAE_DATASETS


class EnhancedVAEEvaluator:
    def __init__(self, device: torch.device, output_dir: str, args: argparse.Namespace):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.args = args
        self.benign_label = args.test_benign_label
        self.fpr_target = getattr(args, "fpr_target", 0.01)
        self._setup_logging()

    def _setup_logging(self):
        import logging
        log_file = self.output_dir / "evaluation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)

    def _set_plot_style(self):
        base_style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot"
        plt.style.use(base_style)
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
        })
        # ⬇️ global colorblind-safe palette
        sns.set_palette("colorblind")

    def _get_color(self, index: int, total: int):
        # deterministic color from tab10 (works well up to ~10)
        cmap = plt.cm.get_cmap("tab10", total)
        return cmap(index)

    def _build_fpr_sweep_table(self,
                               scores: np.ndarray,
                               labels: np.ndarray,
                               targets: List[float]) -> pd.DataFrame:
        """
        为若干 FPR 目标（如 0.01, 0.05, 0.1, ...）构建一个表，
        每行包含该目标 FPR 下的 threshold / FPR / TPR / 各种指标。
        """
        rows = []
        for t in targets:
            res = compute_threshold_fpr(
                scores,
                labels,
                self.benign_label,
                fpr_target=t,
            )
            rows.append({
                'FPR_target': t,
                'threshold': res.get('threshold'),
                'FPR': res.get('fpr'),
                'TPR': res.get('tpr'),
                'accuracy': res.get('accuracy'),
                'precision': res.get('precision'),
                'recall': res.get('recall'),
                'f1_score': res.get('f1_score'),
                'roc_auc': res.get('roc_auc', None),
                'pr_auc': res.get('pr_auc', None),
                'used_fallback': res.get('used_fallback', False),
            })
        df = pd.DataFrame(rows)
        return df

    # ---------- model / data helpers ----------

    def load_attack_mapping(self, mapping_path: Optional[str]) -> Dict[int, str]:
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
            self.logger.warning(f"Attack mapping not found at {mapping_path}, using default")
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
                self.logger.info(f"Detected Label-to-ID format (key='{first_key}'); swapping keys/values.")
                # We swap k and v.
                # k was "Benign", v was 0 -> New dict is {0: "Benign"}
                m_int = {int(v): k for k, v in m.items()}

            # SCENARIO B: Keys are IDs (e.g., "0": "Benign")
            # This matches the original code expectation.
            else:
                m_int = {int(k): v for k, v in m.items()}

            self.logger.info(f"Loaded {len(m_int)} attack categories")
            return m_int

        except Exception as e:
            self.logger.error(f"Error loading attack mapping: {e}")
            return default_mapping

    # def load_model(self, model_path: str, args: argparse.Namespace, dataset_cfg: Optional[object] = None,):
    #     self.logger.info(f"Loading model from {model_path}")
    #     # model = CorrelatedGaussianVAE(
    #     #     latent_dim=config.get('latent_dim', self.args.latent_dim),
    #     #     training_mode=config.get('training_mode', self.args.training_mode),
    #     #     n_components=config.get('n_components', self.args.n_components),
    #     #     hidden_dim=config.get('hidden_dim', 256),
    #     # ).to(self.device)
    #     encoder_cfg = decoder_cfg = None
    #     if dataset_cfg is not None:
    #         m = dataset_cfg.model
    #         encoder_cfg = {
    #             "packet_latent_dim": getattr(m, "packet_latent_dim", 128),
    #             "d_model": getattr(m, "session_d_model", 128),
    #             "latent_dim": args.latent_dim,
    #         }
    #         decoder_cfg = {
    #             "latent_dim": args.latent_dim,
    #             "output_size": getattr(m, "num_packets", 16) * getattr(m, "packet_len", 256),
    #             "rank": getattr(m, "decoder_rank", args.hidden_dim),
    #             "num_deep_layers": getattr(m, "decoder_num_layers", 1),
    #         }
    #
    #     model = CorrelatedGaussianVAE(
    #         latent_dim=args.latent_dim,
    #         training_mode=args.training_mode,
    #         n_components=args.n_components,
    #         hidden_dim=args.hidden_dim,
    #         dropout=args.dropout_rate,
    #         encoder_cfg=encoder_cfg,
    #         decoder_cfg=decoder_cfg,
    #     ).to(self.device)
    #
    #     trainer = VAETrainer(model, self.device)
    #     ckpt = torch.load(model_path, map_location=self.device)
    #     if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    #         model.load_state_dict(ckpt['model_state_dict'])
    #     elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
    #         model.load_state_dict(ckpt['state_dict'])
    #     else:
    #         model.load_state_dict(ckpt)
    #     self.logger.info(f"Model loaded | params: {sum(p.numel() for p in model.parameters()):,}")
    #     return trainer
    def load_model(self, model_path: str, args: argparse.Namespace, dataset_cfg: Optional[object] = None):
        self.logger.info(f"Loading model from {model_path}")

        # Construct Configs based on Dataset Defaults + CLI Overrides
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

        # Initialize Model
        model = CorrelatedGaussianVAE(
            latent_dim=args.latent_dim,
            training_mode=args.training_mode,
            n_components=args.n_components,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout_rate,
            encoder_cfg=encoder_cfg,
            decoder_cfg=decoder_cfg,
        ).to(self.device)

        # Load Weights
        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()

        self.logger.info(f"Model loaded successfully. Params: {sum(p.numel() for p in model.parameters()):,}")

        # Return wrapped in Trainer for easy access to utility methods
        return VAETrainer(model, self.device)

    def compute_reference_statistics(self, trainer, train_loader: DataLoader, cache_path: Optional[str] = None) -> Dict:
        if cache_path and os.path.exists(cache_path):
            self.logger.info(f"Loading cached reference statistics from {cache_path}")
            return torch.load(cache_path, map_location=self.device)
        self.logger.info("Computing reference statistics from training data...")
        stats = trainer.compute_reference_statistics(train_loader)
        if cache_path:
            torch.save(stats, cache_path)
            self.logger.info(f"Cached reference statistics to {cache_path}")
        return stats

    # ---------- evaluation ----------

    def evaluate_method(self, trainer, test_loader: DataLoader, test_labels: np.ndarray,
                        reference_stats: Dict, method: str, attack_mapping: Dict) -> Dict:

        self.logger.info(f"Evaluating {method} ...")
        trainer.model.eval()
        with torch.no_grad():
            scores = trainer.compute_anomaly_scores(test_loader, method, reference_stats)
        scores = np.asarray(scores).reshape(-1)
        self.logger.info(f"{method}: collected {scores.shape[0]} scores")

        # ⬇️ save scores for later FPR-sweep
        method_dir = self.output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        np.save(method_dir / f"{method}_scores.npy", scores)

        # ---- optimal thresholds (F1-based) ----
        threshold_results = compute_optimal_threshold(scores, test_labels, self.benign_label)
        best_key = max(threshold_results.keys(), key=lambda k: threshold_results[k]['f1_score'])
        best_threshold = threshold_results[best_key]['threshold']

        # ---- FPR-based threshold ----
        fpr_result = compute_threshold_fpr(
            scores,
            test_labels,
            self.benign_label,
            fpr_target=self.fpr_target,
        )

        binary_labels = (test_labels != self.benign_label).astype(int)
        predictions = (scores >= best_threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(binary_labels, predictions),
            'precision': precision_score(binary_labels, predictions, zero_division=0),
            'recall': recall_score(binary_labels, predictions, zero_division=0),
            'f1_score': f1_score(binary_labels, predictions, zero_division=0),
            'roc_auc': roc_auc_score(binary_labels, scores) if len(np.unique(binary_labels)) > 1 else 0.0,
            'fpr_target': self.fpr_target,
            'fpr_at_target': fpr_result.get('fpr', None),
            'threshold_fpr': fpr_result.get('threshold', None),
        }

        per_category = self._analyze_per_category(scores, test_labels, predictions, best_threshold, attack_mapping)

        # visualizations
        self._create_visualizations(scores, test_labels, best_threshold, attack_mapping, method, method_dir)

        return {
            'method': method,
            'best_threshold': float(best_threshold),
            'metrics': metrics,
            'per_category': per_category,
            'fpr_result': fpr_result,
            'confusion_matrix': confusion_matrix(binary_labels, predictions).tolist()
        }

    def evaluate_all_methods(self, trainer: VAETrainer, test_loader: DataLoader, test_labels: np.ndarray,
                             reference_stats: Dict, attack_mapping: Dict, methods: List[str]) -> Dict:
        results = {}
        methods = methods or ['reconstruction']
        for m in methods:
            try:
                r = self.evaluate_method(trainer, test_loader, test_labels, reference_stats, m, attack_mapping)
                f1 = r['metrics']['f1_score']
                auc_ = r['metrics']['roc_auc']
                fpr_at = r['metrics'].get('fpr_at_target')
                msg = f"{m.upper()} - F1: {f1:.4f}, AUC: {auc_:.4f}"
                if fpr_at is not None:
                    msg += f", FPR@{self.fpr_target:.3f}: {fpr_at:.4f}"
                self.logger.info(msg)
                results[m] = r
            except Exception as e:
                self.logger.error(f"{m.upper()} failed: {e}", exc_info=True)
        return results

    def _analyze_per_category(self, scores: np.ndarray, labels: np.ndarray, predictions: np.ndarray,
                              threshold: float, attack_mapping: Dict) -> Dict:
        out = {}
        for label in np.unique(labels):
            mask = labels == label
            if not np.any(mask):
                continue
            name = attack_mapping.get(int(label), f"Category_{label}")
            cat_scores = scores[mask]
            cat_preds = predictions[mask]
            acc = np.mean(cat_preds == (0 if label == self.benign_label else 1))
            out[name] = {
                'label': int(label),
                'sample_count': int(mask.sum()),
                'accuracy': float(acc),
                'detection_rate': float(cat_preds.mean()),
                'mean_score': float(cat_scores.mean()),
                'std_score': float(cat_scores.std()),
                'min_score': float(cat_scores.min()),
                'max_score': float(cat_scores.max()),
                'median_score': float(np.median(cat_scores)),
            }
        return out

    # ---------- visualizations ----------

    def _create_visualizations(self, scores: np.ndarray, labels: np.ndarray, threshold: float,
                               attack_mapping: Dict, method: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self._set_plot_style()

        # 1) Score distributions
        fig, ax = plt.subplots(figsize=(7, 5))
        self._plot_score_distributions(ax, scores, labels, threshold, attack_mapping)
        fig.suptitle(f'{method.capitalize()} - Score Distributions', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_dir / f'scores_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 2) Per-category distribution
        fig, ax = plt.subplots(figsize=(7, 5))
        self._plot_box_plots(ax, scores, labels, threshold, attack_mapping)
        fig.suptitle(f'{method.capitalize()} - Per-Category Distribution', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_dir / f'per_category_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 3) Per-category accuracy
        fig, ax = plt.subplots(figsize=(7, 5))
        self._plot_category_accuracy(ax, scores, labels, threshold, attack_mapping)
        fig.suptitle(f'{method.capitalize()} - Per-Category Accuracy', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_dir / f'per_category_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 4) Confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        self._plot_confusion_matrix(ax, labels, scores >= threshold)
        fig.suptitle(f'{method.capitalize()} - Confusion Matrix', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_dir / f'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 5) Score timeline
        fig, ax = plt.subplots(figsize=(8, 4))
        self._plot_score_timeline(ax, scores, labels, threshold, attack_mapping)
        fig.suptitle(f'{method.capitalize()} - Score Timeline', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_dir / f'score_timeline.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 6) Threshold vs metrics
        fig, ax = plt.subplots(figsize=(7, 5))
        self._plot_threshold_analysis(ax, scores, labels)
        fig.suptitle(f'{method.capitalize()} - Threshold Analysis', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_dir / f'threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 7) ROC curve
        binary_labels = (labels != self.benign_label).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, linewidth=2.5, label=f'ROC (AUC={roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{method.capitalize()} - ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_score_distributions(self, ax, scores, labels, threshold, attack_mapping):
        uniq = np.unique(labels)
        n_cat = len(uniq)

        for i, lab in enumerate(uniq):
            mask = labels == lab
            vals = scores[mask]
            if vals.size == 0:
                continue

            color = self._get_color(i, n_cat)
            name = attack_mapping.get(int(lab), f"Cat_{lab}")
            label_txt = f"{name} (n={mask.sum()})"

            if vals.size > 1:
                sns.kdeplot(
                    vals,
                    ax=ax,
                    fill=True,
                    alpha=0.35,
                    linewidth=1.6,
                    label=label_txt,
                    color=color,
                )
            else:
                ax.scatter(vals, [0], color=color, s=30, label=label_txt)

        ax.axvline(threshold, color="black", ls="--", lw=1.8, label=f"Threshold = {threshold:.3f}")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.set_title("Score Distributions by Category")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    def _plot_box_plots(self, ax, scores, labels, threshold, attack_mapping):
        data, labs = [], []
        for lab in np.unique(labels):
            m = labels == lab
            if m.sum() > 0:
                data.append(scores[m])
                labs.append(attack_mapping.get(int(lab), f"Cat_{lab}"))

        if not data:
            ax.set_visible(False)
            return

        vp = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
        colors = plt.cm.tab20(np.linspace(0, 1, len(data)))
        for body, c in zip(vp['bodies'], colors):
            body.set_facecolor(c)
            body.set_edgecolor("black")
            body.set_alpha(0.7)

        ax.set_xticks(np.arange(1, len(labs) + 1))
        ax.set_xticklabels(labs, rotation=35, ha='right')
        ax.axhline(threshold, color='red', ls='--', lw=2, alpha=0.7)
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Per-Category Score Distribution')
        ax.grid(True, alpha=0.3)

    def _plot_category_accuracy(self, ax, scores, labels, threshold, attack_mapping):
        preds = (scores >= threshold).astype(int)
        cats, accs = [], []

        for lab in np.unique(labels):
            m = labels == lab
            if m.sum() == 0:
                continue
            name = attack_mapping.get(int(lab), f"Cat_{lab}")
            target = 0 if lab == self.benign_label else 1
            acc = np.mean(preds[m] == target)
            cats.append(name)
            accs.append(acc)

        if not cats:
            ax.set_visible(False)
            return

        y_pos = np.arange(len(cats))
        bars = []
        for i, (y, a) in enumerate(zip(y_pos, accs)):
            color = self._get_color(i, len(cats))
            b = ax.barh(y, a, color=color, alpha=0.9)
            bars.append(b)
            ax.text(a + 0.01, y, f"{a:.2f}", va="center", ha="left")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(cats)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Detection Accuracy")
        ax.set_title("Per-Category Detection Accuracy")
        ax.grid(True, axis="x", alpha=0.3)

    def _plot_confusion_matrix(self, ax, labels, predictions):
        y = (labels != self.benign_label).astype(int)
        cm = confusion_matrix(y, predictions.astype(int))
        with np.errstate(all='ignore'):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_norm[i, j] * 100 if cm.sum(axis=1)[i] > 0 else 0.0
                annot[i, j] = f"{count}\n{pct:.1f}%"

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",  # <-- grayscale-friendly
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Proportion"},
        )
        ax.set_xticklabels(["Benign", "Malicious"])
        ax.set_yticklabels(["Benign", "Malicious"], rotation=0)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

    def _plot_score_timeline(self, ax, scores, labels, threshold, attack_mapping):
        uniq = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(uniq)))
        for i, lab in enumerate(uniq):
            m = labels == lab
            idx = np.where(m)[0]
            if idx.size == 0:
                continue
            ax.scatter(
                idx, scores[m], s=10, alpha=0.6, color=colors[i],
                label=attack_mapping.get(int(lab), f"Cat_{lab}")
            )

        # highlight top anomalies
        k = min(30, len(scores))
        if k > 0:
            top_idx = np.argsort(scores)[-k:]
            ax.scatter(
                top_idx, scores[top_idx],
                s=40, facecolors='none', edgecolors='red', linewidths=0.8, label='Top anomalies'
            )

        ax.axhline(threshold, color='red', ls='--', lw=2, alpha=0.7, label=f'Th={threshold:.3f}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Anomaly Scores Timeline')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    def _plot_threshold_analysis(self, ax, scores, labels):
        y = (labels != self.benign_label).astype(int)
        ts = np.linspace(scores.min(), scores.max(), 120)
        accs, f1s = [], []
        for t in ts:
            p = (scores >= t).astype(int)
            accs.append(accuracy_score(y, p))
            f1s.append(f1_score(y, p, zero_division=0))

        accs = np.asarray(accs)
        f1s = np.asarray(f1s)

        ax.plot(ts, accs, label='Accuracy', lw=2)
        ax.plot(ts, f1s, label='F1', lw=2)

        best_idx = int(np.argmax(f1s))
        ax.scatter(ts[best_idx], f1s[best_idx], color='red', s=80, zorder=5)
        ax.axvline(ts[best_idx], color='red', ls='--', alpha=0.5)
        ax.text(ts[best_idx], f1s[best_idx] + 0.02, f'Best F1 @ {ts[best_idx]:.3f}',
                ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Performance vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ---------- report ----------
    def generate_report(self, results: Dict, attack_mapping: Dict, test_labels: np.ndarray):
        self.logger.info("Generating evaluation report...")
        if not results:
            self.logger.error("No results to generate report from")
            return None, None

        best = max(results.keys(), key=lambda k: results[k]['metrics']['f1_score'])

        rows = [{
            'Method': m,
            'Accuracy': r['metrics']['accuracy'],
            'Precision': r['metrics']['precision'],
            'Recall': r['metrics']['recall'],
            'F1-Score': r['metrics']['f1_score'],
            'ROC-AUC': r['metrics']['roc_auc'],
            'Best Threshold (F1)': r['best_threshold'],
            'Threshold (FPR)': r['metrics'].get('threshold_fpr', None),
            f'FPR@{self.fpr_target:.3f}': r['metrics'].get('fpr_at_target', None),
        } for m, r in results.items()]
        df = pd.DataFrame(rows).sort_values('F1-Score', ascending=False)
        df.to_csv(self.output_dir / 'method_comparison.csv', index=False)

        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # ===== FPR sweep for ALL methods =====
        fpr_targets = [0.01, 0.05, 0.10, 0.15, 0.20]

        for method_name, r in results.items():
            try:
                scores_path = self.output_dir / method_name / f"{method_name}_scores.npy"
                if scores_path.exists():
                    scores = np.load(scores_path)
                    fpr_df = self._build_fpr_sweep_table(scores, test_labels, fpr_targets)

                    # 每个 method 单独一个 FPR sweep 表
                    fpr_csv_path = self.output_dir / f'method_{method_name}_fpr_sweep.csv'
                    fpr_df.to_csv(fpr_csv_path, index=False)

                    self.logger.info(
                        f"Saved FPR sweep table for method '{method_name}' to {fpr_csv_path}"
                    )
                else:
                    self.logger.warning(
                        f"No scores file found for method '{method_name}' at {scores_path}; "
                        "skip FPR sweep table."
                    )
            except Exception as e:
                self.logger.error(
                    f"Failed to build FPR sweep table for method '{method_name}': {e}",
                    exc_info=True
                )

        # ===== text report (same as before, but now with correct FPR values) =====
        lines = ["="*70, "ENHANCED VAE ANOMALY DETECTION EVALUATION REPORT", "="*70, ""]
        lines += [
            f"Benign Label: {self.benign_label}",
            f"Best Method: {best.upper()}",
            f"Best F1-Score: {results[best]['metrics']['f1_score']:.4f}",
            "",
            "Method Comparison:",
            df.to_string(index=False),
            "",
            "Dataset Composition:",
        ]
        uniq, cnt = np.unique(test_labels, return_counts=True)
        for lab, c in zip(uniq, cnt):
            cat = attack_mapping.get(int(lab), f"Category_{lab}")
            pct = c / len(test_labels) * 100
            mark = " (BENIGN)" if lab == self.benign_label else ""
            lines.append(f"  {cat}{mark}: {int(c)} samples ({pct:.1f}%)")

        lines += ["", f"Per-Category Results ({best.upper()}):", "-"*70]
        for cat, st in results[best]['per_category'].items():
            lines.append(
                f"{cat}: Accuracy={st['accuracy']:.3f}, "
                f"Mean Score={st['mean_score']:.3f}±{st['std_score']:.3f}"
            )

        with open(self.output_dir / 'evaluation_report.txt', 'w') as f:
            f.write("\n".join(lines))
        self.logger.info(f"Report saved to {self.output_dir}")
        return df, best

def _apply_dataset_cfg_to_args(args: argparse.Namespace, ds_cfg) -> argparse.Namespace:
    """
    Fill training hyper-parameters from VAEDatasetConfig into argparse args.

    CLI flags still exist; after calling this, args becomes the single
    source of truth and will be written to config.json.
    """
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


def make_loader(dataset: ByteDataset, batch_size: int, device: torch.device, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced VAE Model Evaluation for Anomaly Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(VAE_DATASETS.keys()),
        default='cicids2018',
        help='Dataset key from VAE_DATASETS in config.py'
    )

    # Required
    parser.add_argument('--model_path', type=str, required=False,
                        default=r"D:\OOD_detect\outputs\CICIDS2018-64\checkpoints\best_epoch_18.pth")
    parser.add_argument('--test_data_path', type=str, required=False, default=None)

    # Data for reference stats
    parser.add_argument('--train_data_path', type=str, required=False, default=None)
    parser.add_argument('--reference_stats_path', type=str, default=None)
    parser.add_argument('--attack_mapping_path', type=str, default=None)

    # Benign label
    parser.add_argument('--test_benign_label', type=int, default=0)

    # Model cfg (fallbacks if config files missing)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--training_mode', type=str, default='correlated', choices=['diagonal', 'correlated', 'gmm'])


    parser.add_argument('--n_components', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    # (legacy args kept for compatibility; not strongly used)
    parser.add_argument('--use_covariance', action='store_true', default=True)


    parser.add_argument('--use_simplified', action='store_true', default=False)




    parser.add_argument('--use_embedding', action='store_true', default=True)
    parser.add_argument('--packet_length', type=int, default=256)
    parser.add_argument('--num_packets', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--data_normalization', action='store_true', default=False)
    parser.add_argument('--embed_dim', type=int, default=256)

    # Evaluation
    parser.add_argument('--methods', nargs='+',
                        default=['reconstruction','mahalanobis',"whitened_l2", 'combined','kl'])
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=r"D:\OOD_detect\test_results\ids2018-to-2017")

    # FPR target
    parser.add_argument('--fpr_target', type=float, default=0.1,
                        help='Target FPR used in compute_threshold_fpr')

    # System
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # ----- apply VAE_DATASETS config -----
    if args.dataset not in VAE_DATASETS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Available: {list(VAE_DATASETS.keys())}")
    ds_cfg = VAE_DATASETS[args.dataset]
    print(f"Using dataset '{args.dataset}'")
    # data paths
    if args.test_data_path is None:
        args.test_data_path = ds_cfg.test_data_path
    if args.train_data_path is None:
        args.train_data_path = ds_cfg.train_data_path
    if args.attack_mapping_path is None:
        args.attack_mapping_path = ds_cfg.attack_mapping_path

    # labels
    if args.test_benign_label is None:
        args.test_benign_label = ds_cfg.test_benign_label

    # model dims
    if args.latent_dim is None:
        args.latent_dim = ds_cfg.model.latent_dim

    # batch size
    if args.batch_size is None:
        args.batch_size = ds_cfg.train.batch_size

    # output dir
    if args.output_dir is None:
        args.output_dir = f'./test_results/{ds_cfg.name}'

    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device} | benign label: {args.test_benign_label}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    evaluator = EnhancedVAEEvaluator(device, args.output_dir, ds_cfg)

    # ---- load model ----
    args = _apply_dataset_cfg_to_args(args, ds_cfg)
    trainer = evaluator.load_model(args.model_path, args, ds_cfg)

    # ---- data ----
    evaluator.logger.info(f"Loading test data from {args.test_data_path}")
    test_dataset = ByteDataset(args.test_data_path)
    test_loader = make_loader(test_dataset, args.batch_size, device, args.num_workers)

    if hasattr(test_dataset, "y") and isinstance(test_dataset.y, torch.Tensor):
        test_labels = test_dataset.y.cpu().numpy()
    else:
        tmp = []
        for _, labels in test_loader:
            tmp.extend(labels.numpy())
        test_labels = np.array(tmp)
    evaluator.logger.info(f"Loaded {len(test_labels)} test samples")

    attack_mapping = evaluator.load_attack_mapping(args.attack_mapping_path)

    # ---- reference stats ----
    if args.reference_stats_path and os.path.exists(args.reference_stats_path):
        reference_stats = torch.load(args.reference_stats_path, map_location=device)
        evaluator.logger.info(f"Loaded pre-computed reference stats from {args.reference_stats_path}")
    else:
        if not args.train_data_path:
            raise ValueError("Either --reference_stats_path or --train_data_path must be provided")
        evaluator.logger.info(f"Building reference stats from {args.train_data_path}")
        train_dataset = ByteDataset(args.train_data_path)
        train_loader = make_loader(train_dataset, args.batch_size, device, args.num_workers)
        cache_path = args.reference_stats_path or (Path(args.output_dir) / 'reference_stats.pth')
        reference_stats = evaluator.compute_reference_statistics(trainer, train_loader, cache_path)

    # ---- eval ----
    results = evaluator.evaluate_all_methods(
        trainer, test_loader, test_labels, reference_stats, attack_mapping, args.methods
    )
    summary_df, best_method = evaluator.generate_report(results, attack_mapping, test_labels)

    print("\n" + "="*70)
    print("ENHANCED VAE EVALUATION COMPLETED")
    print("="*70)
    print(f"Best method: {best_method.upper() if best_method else 'None'}")
    print(f"Results saved to: {args.output_dir}")
    if summary_df is not None:
        print("\nSummary:")
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
