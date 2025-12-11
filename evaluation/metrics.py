from typing import Dict, Tuple, List, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
from scipy import stats
from typing import Dict, Optional
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, f1_score, accuracy_score

def compute_binary_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           scores: Optional[np.ndarray] = None) -> Dict[str, float]:
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if scores is not None and np.unique(y_true).size > 1 and np.unique(scores).size > 1:
        try:
            out['roc_auc'] = float(roc_auc_score(y_true, scores))
            P, R, _ = precision_recall_curve(y_true, scores)
            out['pr_auc'] = float(auc(R, P))
        except Exception:
            out['roc_auc'] = 0.0
            out['pr_auc'] = 0.0
    return out


def compute_threshold_fpr(scores: np.ndarray,
                          labels: np.ndarray,
                          test_benign_label: int,
                          fpr_target: float = 0.01,
                          fallback_percentile: float = 99.5) -> Dict[str, float]:
    """
    返回一个阈值，使 FPR ≈ fpr_target（默认 1%）。
    - 若能基于 ROC 曲线找到接近目标 FPR 的点，则使用该点；
    - 若无法构造 ROC（类别或分数不够丰富），则退回到“良性分数分位数”阈值。
    返回示例字段：
        {
            'method': 'fpr@1.00%',
            'threshold': ...,
            'fpr': ... 或 None,
            'tpr': ... 或 None,
            'accuracy': ...,
            'precision': ...,
            'recall': ...,
            'f1_score': ...,
            'roc_auc': ...,
            'pr_auc': ...,
            'used_fallback': True/False,
        }
    """
    scores = scores.astype(float)
    # 0 = benign, 1 = attack
    y = (labels != test_benign_label).astype(int)

    result: Dict[str, float] = {
        'method': f'fpr@{fpr_target*100:.2f}%',
        'used_fallback': False,
    }

    can_curve = (np.unique(y).size > 1) and (np.unique(scores).size > 1)

    if can_curve:
        fpr, tpr, th = roc_curve(y, scores)   # thresholds correspond to scores (>= positive)

        # 找到 FPR <= target 的最后一个点；若不存在，则选择最接近 target 的点
        idx_ok = np.where(fpr <= fpr_target)[0]
        if idx_ok.size > 0:
            j = idx_ok[-1]
        else:
            j = int(np.argmin(np.abs(fpr - fpr_target)))

        thr = float(th[j])
        y_pred = (scores >= thr).astype(int)
        metrics = compute_binary_metrics(y, y_pred, scores)

        result.update({
            'threshold': thr,
            'fpr': float(fpr[j]),
            'tpr': float(tpr[j]),
            **metrics,
        })
    else:
        result['used_fallback'] = True

    # ---- 回退路径：仅在无法构建 ROC 时启用 ----
    if not can_curve:
        # 注意：这里用 test_benign_label，而不是硬编码 0
        normal_scores = scores[labels == test_benign_label]
        base = normal_scores if normal_scores.size > 0 else scores

        thr_fb = float(np.percentile(base, fallback_percentile))
        y_pred_fb = (scores >= thr_fb).astype(int)
        metrics_fb = compute_binary_metrics(y, y_pred_fb, scores)

        result.update({
            'threshold': thr_fb,
            'fpr': None,
            'tpr': None,
            **metrics_fb,
        })

    return result


def compute_optimal_threshold(scores: np.ndarray,
                              labels: np.ndarray,
                              test_benign_label: int,
                              num_thresholds: int = 10000,
                              percentiles: List[float] = [90, 95, 99]) -> Dict[str, Dict]:
    """
    Compute optimal thresholds using various methods.

    Args:
        scores: Anomaly scores (higher = more anomalous)
        labels: True labels (0 = normal, >0 = anomaly)
        num_thresholds: Number of thresholds to evaluate
        percentiles: Percentiles of normal scores to use as thresholds

    Returns:
        Dictionary with threshold results for each method
    """
    # Convert to binary labels
    binary_labels = (labels != test_benign_label).astype(int)
    normal_scores = scores[labels == test_benign_label]
    threshold_results = {}

    # Method 1: Percentile-based thresholds
    for p in percentiles:
        if len(normal_scores) > 0:
            threshold = np.percentile(normal_scores, p)
        else:
            threshold = np.percentile(scores, p)

        predictions = (scores >= threshold).astype(int)
        metrics = compute_binary_metrics(binary_labels, predictions, scores)
        threshold_results[f'{p}th_percentile'] = {
            'threshold': float(threshold),
            **metrics
        }

    # Method 2: ROC-based optimal threshold
    if len(np.unique(binary_labels)) > 1:
        fpr, tpr, roc_thresholds = roc_curve(binary_labels, scores)
        optimal_idx = np.argmax(tpr - fpr)
        roc_optimal_threshold = roc_thresholds[optimal_idx]

        predictions = (scores >= roc_optimal_threshold).astype(int)
        metrics = compute_binary_metrics(binary_labels, predictions, scores)

        threshold_results['roc_optimal'] = {
            'threshold': float(roc_optimal_threshold),
            **metrics
        }

    min_score = np.min(scores)
    max_score = np.max(scores)
    threshold_range = np.linspace(min_score, max_score, num_thresholds)

    best_f1 = 0
    best_f1_threshold = min_score
    best_accuracy = 0
    best_accuracy_threshold = min_score

    for threshold in threshold_range:
        predictions = (scores >= threshold).astype(int)

        if len(np.unique(predictions)) > 1:
            f1 = f1_score(binary_labels, predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_f1_threshold = threshold

            accuracy = accuracy_score(binary_labels, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_threshold = threshold


    predictions = (scores >= best_accuracy_threshold).astype(int)
    metrics_accuracy = compute_binary_metrics(binary_labels, predictions, scores)

    threshold_results['acc'] = {
        'threshold': float(best_accuracy),
        **metrics_accuracy
    }

    predictions = (scores >= best_f1_threshold).astype(int)
    metrics = compute_binary_metrics(binary_labels, predictions, scores)

    threshold_results['f1_optimal'] = {
        'threshold': float(best_f1_threshold),
        **metrics
    }

    return threshold_results



# def compute_binary_metrics(true_labels: np.ndarray,
#                            predictions: np.ndarray,
#                            scores: Optional[np.ndarray] = None) -> Dict[str, float]:
#     """
#     Compute binary classification metrics.
#
#     Args:
#         true_labels: True binary labels
#         predictions: Binary predictions
#         scores: Continuous scores (for AUC computation)
#
#     Returns:
#         Dictionary of metrics
#     """
#     metrics = {
#         'accuracy': float(accuracy_score(true_labels, predictions)),
#         'precision': float(precision_score(true_labels, predictions, zero_division=0)),
#         'recall': float(recall_score(true_labels, predictions, zero_division=0)),
#         'f1_score': float(f1_score(true_labels, predictions, zero_division=0))
#     }
#
#     # Add AUC if scores provided
#     if scores is not None and len(np.unique(true_labels)) > 1:
#         try:
#             metrics['roc_auc'] = float(roc_auc_score(true_labels, scores))
#             precision, recall, _ = precision_recall_curve(true_labels, scores)
#             metrics['pr_auc'] = float(auc(recall, precision))
#         except:
#             metrics['roc_auc'] = 0.0
#             metrics['pr_auc'] = 0.0
#
#     return metrics


def analyze_per_category_performance(scores: np.ndarray,
                                     labels: np.ndarray,
                                     threshold: float,
                                     category_mapping: Dict[int, str]) -> Dict:
    """
    Analyze performance for each category/attack type.

    Args:
        scores: Anomaly scores
        labels: True labels (multi-class)
        threshold: Detection threshold
        category_mapping: Mapping from label to category name

    Returns:
        Dictionary with per-category analysis
    """
    predictions = (scores >= threshold).astype(int)
    binary_labels = (labels != 0).astype(int)

    # Overall binary metrics
    overall_metrics = compute_binary_metrics(binary_labels, predictions, scores)

    # Per-category analysis
    unique_labels = np.unique(labels)
    category_results = {}

    for label in unique_labels:
        category_name = category_mapping.get(int(label), f"Category_{label}")
        mask = labels == label

        if np.sum(mask) == 0:
            continue

        category_scores = scores[mask]
        category_predictions = predictions[mask]

        # For normal class (label 0), correct prediction is 0 (not anomalous)
        # For anomaly classes (label > 0), correct prediction is 1 (anomalous)
        if label == 0:
            correct_predictions = (category_predictions == 0)
        else:
            correct_predictions = (category_predictions == 1)

        category_accuracy = np.mean(correct_predictions)

        # Compute statistics
        category_results[category_name] = {
            'label': int(label),
            'sample_count': int(np.sum(mask)),
            'accuracy': float(category_accuracy),
            'detection_rate': float(np.mean(category_predictions)),
            'mean_score': float(np.mean(category_scores)),
            'std_score': float(np.std(category_scores)),
            'min_score': float(np.min(category_scores)),
            'max_score': float(np.max(category_scores)),
            'median_score': float(np.median(category_scores))
        }

    return {
        'overall': overall_metrics,
        'per_category': category_results
    }


def compute_detection_delay(scores: np.ndarray,
                            labels: np.ndarray,
                            threshold: float,
                            window_size: int = 10) -> Dict[str, float]:
    """
    Compute detection delay metrics for time-series anomaly detection.

    Args:
        scores: Anomaly scores over time
        labels: True labels over time
        threshold: Detection threshold
        window_size: Window size for computing delays

    Returns:
        Dictionary with delay metrics
    """
    predictions = (scores >= threshold).astype(int)

    # Find anomaly segments
    anomaly_starts = []
    detection_delays = []

    in_anomaly = False
    anomaly_start = None

    for i in range(len(labels)):
        if labels[i] > 0 and not in_anomaly:
            # Anomaly starts
            in_anomaly = True
            anomaly_start = i
            anomaly_starts.append(i)

            # Find when it's detected
            for j in range(i, min(i + window_size, len(predictions))):
                if predictions[j] == 1:
                    detection_delays.append(j - i)
                    break
            else:
                # Not detected within window
                detection_delays.append(window_size)

        elif labels[i] == 0 and in_anomaly:
            # Anomaly ends
            in_anomaly = False

    if len(detection_delays) > 0:
        avg_delay = np.mean(detection_delays)
        median_delay = np.median(detection_delays)
        detected_ratio = np.sum(np.array(detection_delays) < window_size) / len(detection_delays)
    else:
        avg_delay = median_delay = detected_ratio = 0.0

    return {
        'average_delay': float(avg_delay),
        'median_delay': float(median_delay),
        'detection_rate': float(detected_ratio),
        'num_anomaly_segments': len(anomaly_starts)
    }


def compute_stability_metrics(scores_list: List[np.ndarray],
                              labels: np.ndarray,
                              threshold: float) -> Dict[str, float]:
    """
    Compute stability metrics across multiple runs.

    Args:
        scores_list: List of score arrays from multiple runs
        labels: True labels
        threshold: Detection threshold

    Returns:
        Dictionary with stability metrics
    """
    # Compute metrics for each run
    accuracies = []
    f1_scores = []
    aucs = []

    binary_labels = (labels > 0).astype(int)

    for scores in scores_list:
        predictions = (scores >= threshold).astype(int)

        accuracies.append(accuracy_score(binary_labels, predictions))
        f1_scores.append(f1_score(binary_labels, predictions, zero_division=0))

        if len(np.unique(binary_labels)) > 1:
            try:
                aucs.append(roc_auc_score(binary_labels, scores))
            except:
                pass

    return {
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'f1_mean': float(np.mean(f1_scores)),
        'f1_std': float(np.std(f1_scores)),
        'auc_mean': float(np.mean(aucs)) if aucs else 0.0,
        'auc_std': float(np.std(aucs)) if aucs else 0.0,
        'num_runs': len(scores_list)
    }


def compute_confidence_intervals(scores: np.ndarray,
                                 labels: np.ndarray,
                                 threshold: float,
                                 confidence_level: float = 0.95,
                                 n_bootstrap: int = 1000) -> Dict[str, Tuple[float, float]]:
    """
    Compute confidence intervals for metrics using bootstrap.

    Args:
        scores: Anomaly scores
        labels: True labels
        threshold: Detection threshold
        confidence_level: Confidence level (e.g., 0.95)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with confidence intervals for each metric
    """
    binary_labels = (labels > 0).astype(int)
    predictions = (scores >= threshold).astype(int)

    n_samples = len(labels)
    metrics_bootstrap = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc': []
    }

    # Bootstrap sampling
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)

        sample_labels = binary_labels[indices]
        sample_predictions = predictions[indices]
        sample_scores = scores[indices]

        # Compute metrics
        metrics_bootstrap['accuracy'].append(
            accuracy_score(sample_labels, sample_predictions)
        )
        metrics_bootstrap['precision'].append(
            precision_score(sample_labels, sample_predictions, zero_division=0)
        )
        metrics_bootstrap['recall'].append(
            recall_score(sample_labels, sample_predictions, zero_division=0)
        )
        metrics_bootstrap['f1_score'].append(
            f1_score(sample_labels, sample_predictions, zero_division=0)
        )

        if len(np.unique(sample_labels)) > 1:
            try:
                metrics_bootstrap['roc_auc'].append(
                    roc_auc_score(sample_labels, sample_scores)
                )
            except:
                pass

    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    confidence_intervals = {}
    for metric, values in metrics_bootstrap.items():
        if values:
            lower = np.percentile(values, lower_percentile)
            upper = np.percentile(values, upper_percentile)
            confidence_intervals[metric] = (float(lower), float(upper))
        else:
            confidence_intervals[metric] = (0.0, 0.0)

    return confidence_intervals


def compute_calibration_metrics(scores: np.ndarray,
                                labels: np.ndarray,
                                n_bins: int = 10) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute calibration metrics to assess score reliability.

    Args:
        scores: Anomaly scores (normalized to [0, 1])
        labels: True binary labels
        n_bins: Number of calibration bins

    Returns:
        Dictionary with calibration metrics
    """
    # Normalize scores to [0, 1] if not already
    if scores.max() > 1 or scores.min() < 0:
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    else:
        scores_norm = scores

    binary_labels = (labels > 0).astype(int)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    # Compute calibration
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        bin_mask = (scores_norm >= bin_boundaries[i]) & (scores_norm < bin_boundaries[i + 1])

        if i == n_bins - 1:  # Include maximum value in last bin
            bin_mask = (scores_norm >= bin_boundaries[i]) & (scores_norm <= bin_boundaries[i + 1])

        if np.sum(bin_mask) > 0:
            bin_accuracy = np.mean(binary_labels[bin_mask])
            bin_confidence = np.mean(scores_norm[bin_mask])
            bin_count = np.sum(bin_mask)

            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)

    if bin_accuracies:
        # Expected Calibration Error (ECE)
        ece = np.sum(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)) *
                     np.array(bin_counts)) / np.sum(bin_counts)

        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
    else:
        ece = mce = 0.0

    return {
        'expected_calibration_error': float(ece),
        'maximum_calibration_error': float(mce),
        'bin_accuracies': np.array(bin_accuracies),
        'bin_confidences': np.array(bin_confidences),
        'bin_counts': np.array(bin_counts)
    }


def compute_statistical_tests(normal_scores: np.ndarray,
                              anomaly_scores: np.ndarray) -> Dict[str, Dict]:
    """
    Perform statistical tests to compare normal and anomaly score distributions.

    Args:
        normal_scores: Scores for normal samples
        anomaly_scores: Scores for anomaly samples

    Returns:
        Dictionary with test results
    """
    results = {}

    # Mann-Whitney U test (non-parametric)
    try:
        statistic, p_value = stats.mannwhitneyu(
            normal_scores, anomaly_scores, alternative='less'
        )
        results['mann_whitney_u'] = {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
    except:
        results['mann_whitney_u'] = {'error': 'Test failed'}

    # Kolmogorov-Smirnov test
    try:
        statistic, p_value = stats.ks_2samp(normal_scores, anomaly_scores)
        results['kolmogorov_smirnov'] = {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
    except:
        results['kolmogorov_smirnov'] = {'error': 'Test failed'}

    # Effect size (Cohen's d)
    try:
        mean_diff = np.mean(anomaly_scores) - np.mean(normal_scores)
        pooled_std = np.sqrt((np.var(normal_scores) + np.var(anomaly_scores)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        results['effect_size'] = {
            'cohens_d': float(cohens_d),
            'interpretation': interpret_cohens_d(cohens_d)
        }
    except:
        results['effect_size'] = {'error': 'Calculation failed'}

    return results


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compute_multi_class_metrics(true_labels: np.ndarray,
                                predictions: np.ndarray,
                                class_names: Optional[List[str]] = None) -> Dict:
    """
    Compute multi-class classification metrics.

    Args:
        true_labels: True multi-class labels
        predictions: Predicted labels
        class_names: Names for each class

    Returns:
        Dictionary with multi-class metrics
    """
    # Basic metrics
    accuracy = accuracy_score(true_labels, predictions)

    # Per-class metrics
    precision_per_class = precision_score(true_labels, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)

    # Averaged metrics
    macro_precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)

    weighted_precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    weighted_recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    weighted_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Classification report
    if class_names:
        report = classification_report(true_labels, predictions,
                                       target_names=class_names,
                                       output_dict=True)
    else:
        report = classification_report(true_labels, predictions, output_dict=True)

    return {
        'accuracy': float(accuracy),
        'macro_metrics': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1_score': float(macro_f1)
        },
        'weighted_metrics': {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1_score': float(weighted_f1)
        },
        'per_class_metrics': {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1_score': f1_per_class.tolist()
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }