import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import warnings
import os

warnings.filterwarnings('ignore')

def plot_score_distributions(scores: np.ndarray,
                             labels: np.ndarray,
                             category_mapping: Dict[int, str],
                             threshold: Optional[float] = None,
                             method_name: str = '',
                             output_dir: Optional[str] = None,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot score distributions for each category.

    Args:
        scores: Anomaly scores
        labels: True labels
        category_mapping: Mapping from label to category name
        threshold: Detection threshold to plot
        method_name: Name of the scoring method
        output_dir: Directory to save plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    unique_labels = np.unique(labels)
    n_categories = len(unique_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, n_categories))

    # Plot 1: Overlapping histograms
    ax = axes[0, 0]
    for i, label in enumerate(unique_labels):
        category_name = category_mapping.get(int(label), f"Category_{label}")
        mask = labels == label
        category_scores = scores[mask]

        if len(category_scores) > 0:
            ax.hist(category_scores, bins=30, alpha=0.6,
                    label=f"{category_name} (n={np.sum(mask)})",
                    color=colors[i], density=True)

    if threshold is not None:
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold={threshold:.3f}')

    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title(f'Score Distributions by Category - {method_name}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Box plots
    ax = axes[0, 1]
    score_data = []
    category_names = []

    for label in unique_labels:
        category_name = category_mapping.get(int(label), f"Category_{label}")
        mask = labels == label
        if np.sum(mask) > 0:
            score_data.append(scores[mask])
            category_names.append(category_name)

    bp = ax.boxplot(score_data, labels=category_names, patch_artist=True)

    # Color box plots
    for patch, color in zip(bp['boxes'], colors[:len(score_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    if threshold is not None:
        ax.axhline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold={threshold:.3f}')

    ax.set_ylabel('Anomaly Score')
    ax.set_title(f'Score Distribution Box Plots - {method_name}')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    # Plot 3: Violin plots
    ax = axes[1, 0]
    data_for_violin = []
    labels_for_violin = []

    for label in unique_labels:
        category_name = category_mapping.get(int(label), f"Category_{label}")
        mask = labels == label
        if np.sum(mask) > 0:
            data_for_violin.extend(scores[mask])
            labels_for_violin.extend([category_name] * np.sum(mask))

    if data_for_violin:
        df = pd.DataFrame({'Score': data_for_violin, 'Category': labels_for_violin})
        sns.violinplot(data=df, x='Category', y='Score', ax=ax, inner='quartile')

        if threshold is not None:
            ax.axhline(threshold, color='red', linestyle='--', linewidth=2)

    ax.set_title(f'Score Distribution Violin Plots - {method_name}')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    # Plot 4: Detection accuracy by category
    ax = axes[1, 1]
    predictions = (scores >= threshold).astype(int) if threshold is not None else np.zeros_like(scores)

    accuracies = []
    categories = []

    for label in unique_labels:
        category_name = category_mapping.get(int(label), f"Category_{label}")
        mask = labels == label
        if np.sum(mask) > 0:
            category_predictions = predictions[mask]

            # For normal (label 0), correct is prediction 0; for anomalies, correct is prediction 1
            if label == 0:
                accuracy = np.mean(category_predictions == 0)
            else:
                accuracy = np.mean(category_predictions == 1)

            accuracies.append(accuracy)
            categories.append(category_name)

    bars = ax.bar(categories, accuracies, color=colors[:len(categories)], alpha=0.7)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')

    ax.set_ylabel('Detection Accuracy')
    ax.set_title(f'Per-Category Detection Accuracy - {method_name}')
    ax.set_ylim(0, 1.1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        save_path = Path(output_dir) / 'score_distributions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None

    return fig


def plot_roc_and_pr_curves(fpr: np.ndarray,
                           tpr: np.ndarray,
                           roc_auc: float,
                           recall: np.ndarray,
                           precision: np.ndarray,
                           pr_auc: float,
                           method_name: str = '',
                           output_dir: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot ROC and Precision-Recall curves.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: ROC AUC score
        recall: Recall values
        precision: Precision values
        pr_auc: PR AUC score
        method_name: Name of the method
        output_dir: Directory to save plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ROC Curve
    ax1.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    # Find optimal point (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    ax1.scatter(fpr[optimal_idx], tpr[optimal_idx],
                color='red', s=100, zorder=5,
                label=f'Optimal (FPR={fpr[optimal_idx]:.3f}, TPR={tpr[optimal_idx]:.3f})')

    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve - {method_name}')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Precision-Recall Curve
    ax2.plot(recall, precision, linewidth=2, label=f'PR curve (AUC = {pr_auc:.4f})')

    # Baseline (random classifier)
    positive_rate = np.sum(tpr) / len(tpr)  # Approximate positive rate
    ax2.axhline(y=positive_rate, color='k', linestyle='--', linewidth=1, label='Random')

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Precision-Recall Curve - {method_name}')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    if output_dir:
        save_path = Path(output_dir) / 'roc_pr_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None

    return fig


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: List[str],
                          method_name: str = '',
                          output_dir: Optional[str] = None,
                          figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix
        class_names: Names for each class
        method_name: Name of the method
        output_dir: Directory to save plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize for percentage display
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)

    # Add percentage annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.texts[i * len(class_names) + j]
            text.set_text(f'{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)')

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - {method_name}')

    plt.tight_layout()

    if output_dir:
        save_path = Path(output_dir) / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None

    return fig


def create_detailed_visualizations(scores: np.ndarray,
                                   labels: np.ndarray,
                                   threshold: float,
                                   attack_mapping: Dict[int, str],
                                   method_name: str,
                                   output_dir: str):
    """
    Create comprehensive visualizations for evaluation results.

    Args:
        scores: Anomaly scores
        labels: True labels
        threshold: Detection threshold
        attack_mapping: Category mapping
        method_name: Name of the method
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Score distributions
    plot_score_distributions(scores, labels, attack_mapping, threshold,
                             method_name, output_dir)

    # Additional specialized plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Score vs Index (time series view)
    ax = axes[0, 0]
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        indices = np.where(mask)[0]
        ax.scatter(indices, scores[mask], alpha=0.6, s=10,
                   label=attack_mapping.get(int(label), f"Category_{label}"),
                   color=colors[i])

    ax.axhline(threshold, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Anomaly Scores Over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 2: Score percentiles by category
    ax = axes[0, 1]
    percentiles = [5, 25, 50, 75, 95]

    for label in unique_labels:
        category_name = attack_mapping.get(int(label), f"Category_{label}")
        mask = labels == label
        if np.sum(mask) > 0:
            category_scores = scores[mask]
            percentile_values = np.percentile(category_scores, percentiles)
            ax.plot(percentiles, percentile_values, marker='o',
                    label=category_name, linewidth=2)

    ax.axhline(threshold, color='red', linestyle='--', linewidth=2,
               label='Threshold')
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Score Value')
    ax.set_title('Score Percentiles by Category')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Detection performance vs threshold
    ax = axes[1, 0]
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    tprs = []
    fprs = []

    binary_labels = (labels > 0).astype(int)

    for t in thresholds:
        predictions = (scores >= t).astype(int)
        tp = np.sum((predictions == 1) & (binary_labels == 1))
        fp = np.sum((predictions == 1) & (binary_labels == 0))
        tn = np.sum((predictions == 0) & (binary_labels == 0))
        fn = np.sum((predictions == 0) & (binary_labels == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tprs.append(tpr)
        fprs.append(fpr)

    ax.plot(thresholds, tprs, label='TPR (Sensitivity)', linewidth=2)
    ax.plot(thresholds, fprs, label='FPR (1-Specificity)', linewidth=2)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label='Selected Threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Rate')
    ax.set_title('TPR and FPR vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Category-wise detection matrix
    ax = axes[1, 1]
    predictions = (scores >= threshold).astype(int)

    # Create detection matrix
    categories = [attack_mapping.get(int(l), f"Cat_{l}") for l in unique_labels]
    detection_matrix = np.zeros((len(categories), 2))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if np.sum(mask) > 0:
            detection_matrix[i, 0] = np.mean(predictions[mask] == 0)  # Not detected
            detection_matrix[i, 1] = np.mean(predictions[mask] == 1)  # Detected

    im = ax.imshow(detection_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Add text annotations
    for i in range(len(categories)):
        for j in range(2):
            text = ax.text(j, i, f'{detection_matrix[i, j]:.2f}',
                           ha='center', va='center', color='black')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Detected', 'Detected'])
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_title('Detection Rates by Category')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rate')

    plt.tight_layout()
    save_path = Path(output_dir) / 'detailed_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_latent_space_2d(latent_vectors: np.ndarray,
                         labels: np.ndarray,
                         category_mapping: Optional[Dict[int, str]] = None,
                         method: str = 'tsne',
                         output_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot 2D visualization of latent space.

    Args:
        latent_vectors: Latent space representations [n_samples, latent_dim]
        labels: Sample labels
        category_mapping: Mapping from label to category name
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        output_path: Path to save plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # Reduce to 2D if needed
    if latent_vectors.shape[1] > 2:
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            latent_2d = reducer.fit_transform(latent_vectors)
        elif method == 'pca':
            reducer = PCA(n_components=2)
            latent_2d = reducer.fit_transform(latent_vectors)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                latent_2d = reducer.fit_transform(latent_vectors)
            except ImportError:
                print("UMAP not installed, falling back to t-SNE")
                reducer = TSNE(n_components=2, random_state=42)
                latent_2d = reducer.fit_transform(latent_vectors)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        latent_2d = latent_vectors[:, :2]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if category_mapping:
            label_name = category_mapping.get(int(label), f"Category_{label}")
        else:
            label_name = f"Class {label}"

        ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                   alpha=0.6, c=[colors[i]], label=label_name, s=30)

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(f'Latent Space Visualization ({method.upper()})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None

    return fig


def plot_reconstruction_comparison(original: np.ndarray,
                                   reconstructed: np.ndarray,
                                   num_samples: int = 5,
                                   output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison between original and reconstructed samples.

    Args:
        original: Original samples
        reconstructed: Reconstructed samples
        num_samples: Number of samples to plot
        output_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    num_samples = min(num_samples, len(original))
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Original
        axes[i, 0].imshow(original[i].reshape(-1, 1), cmap='viridis', aspect='auto')
        axes[i, 0].set_title(f'Original Sample {i + 1}')
        axes[i, 0].set_ylabel('Features')

        # Reconstructed
        axes[i, 1].imshow(reconstructed[i].reshape(-1, 1), cmap='viridis', aspect='auto')
        axes[i, 1].set_title(f'Reconstructed Sample {i + 1}')

        # Compute MSE
        mse = np.mean((original[i] - reconstructed[i]) ** 2)
        axes[i, 1].text(0.02, 0.98, f'MSE: {mse:.4f}',
                        transform=axes[i, 1].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None

    return fig


def plot_calibration_curve(scores: np.ndarray,
                           labels: np.ndarray,
                           n_bins: int = 10,
                           output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot calibration curve to assess score reliability.

    Args:
        scores: Anomaly scores (should be normalized to [0, 1])
        labels: True binary labels
        n_bins: Number of calibration bins
        output_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    # Normalize scores if needed
    if scores.max() > 1 or scores.min() < 0:
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    else:
        scores_norm = scores

    binary_labels = (labels > 0).astype(int)

    # Compute calibration
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        bin_mask = (scores_norm >= bin_boundaries[i]) & (scores_norm < bin_boundaries[i + 1])
        if i == n_bins - 1:
            bin_mask = (scores_norm >= bin_boundaries[i]) & (scores_norm <= bin_boundaries[i + 1])

        if np.sum(bin_mask) > 0:
            bin_accuracy = np.mean(binary_labels[bin_mask])
            bin_confidence = np.mean(scores_norm[bin_mask])

            bin_centers.append(bin_confidence)
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(np.sum(bin_mask))

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration plot
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.scatter(bin_centers, bin_accuracies, s=np.array(bin_counts) * 2,
                alpha=0.7, label='Actual', zorder=5)
    ax1.plot(bin_centers, bin_accuracies, 'b-', alpha=0.7)

    ax1.set_xlabel('Mean Predicted Score')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Calibration Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Histogram of predictions
    ax2.hist(scores_norm[binary_labels == 0], bins=30, alpha=0.5,
             label='Normal', density=True, color='blue')
    ax2.hist(scores_norm[binary_labels == 1], bins=30, alpha=0.5,
             label='Anomaly', density=True, color='red')

    ax2.set_xlabel('Predicted Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Predicted Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None

    return fig


def create_summary_dashboard(results: Dict[str, Dict],
                             output_path: str,
                             title: str = "VAE Evaluation Summary"):
    """
    Create a summary dashboard with key metrics from all methods.

    Args:
        results: Dictionary with results for each method
        output_path: Path to save dashboard
        title: Dashboard title
    """
    num_methods = len(results)
    fig = plt.figure(figsize=(16, 10))

    # Title
    fig.suptitle(title, fontsize=16, y=0.98)

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Method comparison bar chart
    ax1 = fig.add_subplot(gs[0, :2])
    methods = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    x = np.arange(len(methods))
    width = 0.15

    for i, metric in enumerate(metrics):
        values = [results[m]['binary_metrics'].get(metric, 0) for m in methods]
        ax1.bar(x + i * width - 2 * width, values, width, label=metric.upper())

    ax1.set_xlabel('Method')
    ax1.set_ylabel('Score')
    ax1.set_title('Method Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # 2. Best method details
    ax2 = fig.add_subplot(gs[0, 2])
    best_method = max(results.keys(),
                      key=lambda k: results[k]['binary_metrics']['f1_score'])
    best_metrics = results[best_method]['binary_metrics']

    metrics_text = f"Best Method: {best_method}\n\n"
    for metric, value in best_metrics.items():
        if isinstance(value, float):
            metrics_text += f"{metric}: {value:.4f}\n"

    ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes,
             verticalalignment='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax2.axis('off')

    # 3. Per-category performance heatmap
    ax3 = fig.add_subplot(gs[1, :])

    # Collect category names and methods
    all_categories = set()
    for method_results in results.values():
        if 'per_category_analysis' in method_results:
            all_categories.update(method_results['per_category_analysis']['per_category'].keys())

    categories = sorted(list(all_categories))
    accuracy_matrix = np.zeros((len(methods), len(categories)))

    for i, method in enumerate(methods):
        for j, category in enumerate(categories):
            if 'per_category_analysis' in results[method]:
                cat_data = results[method]['per_category_analysis']['per_category'].get(category, {})
                accuracy_matrix[i, j] = cat_data.get('accuracy', 0)

    im = ax3.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks(range(len(categories)))
    ax3.set_xticklabels(categories, rotation=45, ha='right')
    ax3.set_yticks(range(len(methods)))
    ax3.set_yticklabels(methods)
    ax3.set_title('Detection Accuracy by Method and Category')

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(categories)):
            text = ax3.text(j, i, f'{accuracy_matrix[i, j]:.2f}',
                            ha='center', va='center', color='black')

    plt.colorbar(im, ax=ax3, label='Accuracy')

    # 4. Threshold analysis
    ax4 = fig.add_subplot(gs[2, :])

    for method, method_results in results.items():
        if 'threshold_analysis' in method_results:
            thresholds = []
            f1_scores = []

            for threshold_name, threshold_data in method_results['threshold_analysis'].items():
                thresholds.append(threshold_data['threshold'])
                f1_scores.append(threshold_data['f1_score'])

            # Sort by threshold
            sorted_indices = np.argsort(thresholds)
            thresholds = np.array(thresholds)[sorted_indices]
            f1_scores = np.array(f1_scores)[sorted_indices]

            ax4.plot(thresholds, f1_scores, marker='o', label=method)

    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Score vs Threshold for Different Methods')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()