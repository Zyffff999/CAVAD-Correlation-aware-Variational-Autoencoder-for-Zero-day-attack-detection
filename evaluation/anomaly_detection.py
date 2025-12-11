import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from scipy.stats import multivariate_normal, chi2
import logging

class AnomalyDetector:
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MahalanobisDetector(AnomalyDetector):
    def __init__(self, contamination: float = 0.1,
                 regularization: float = 1e-4):

        super().__init__(contamination)
        self.regularization = regularization
        self.mean = None
        self.cov_inv = None
        self.threshold = None

    def fit(self, X: np.ndarray):
        """
        Fit the detector on normal data.

        Args:
            X: Training data [n_samples, n_features]
        """
        self.mean = np.mean(X, axis=0)

        # Compute covariance with regularization
        cov = np.cov(X.T)
        cov_reg = cov + self.regularization * np.eye(cov.shape[0])

        # Compute inverse
        try:
            self.cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            logging.warning("Using pseudo-inverse for singular covariance matrix")
            self.cov_inv = np.linalg.pinv(cov_reg)

        # Compute threshold based on contamination
        distances = self._compute_distances(X)
        self.threshold = np.percentile(distances, (1 - self.contamination) * 100)

        self.is_fitted = True

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances."""
        diff = X - self.mean
        distances = np.sqrt(np.sum(diff @ self.cov_inv * diff, axis=1))
        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Args:
            X: Data to predict [n_samples, n_features]

        Returns:
            Labels (-1 for anomaly, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        distances = self._compute_distances(X)
        return np.where(distances > self.threshold, -1, 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.

        Args:
            X: Data to score [n_samples, n_features]

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring")

        return self._compute_distances(X)


class ReconstructionDetector(AnomalyDetector):
    """
    Reconstruction error-based anomaly detector.
    """

    def __init__(self, model: torch.nn.Module,
                 contamination: float = 0.1,
                 device: torch.device = None):
        """
        Args:
            model: Trained autoencoder model
            contamination: Expected proportion of anomalies
            device: PyTorch device
        """
        super().__init__(contamination)
        self.model = model
        self.device = device or torch.device('cpu')
        self.threshold = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        """
        Fit the detector on normal data.

        Args:
            X: Training data [n_samples, ...]
        """
        # Compute reconstruction errors on training data
        errors = self._compute_reconstruction_errors(X)

        # Fit scaler on errors
        errors_reshaped = errors.reshape(-1, 1)
        self.scaler.fit(errors_reshaped)

        # Compute threshold
        self.threshold = np.percentile(errors, (1 - self.contamination) * 100)

        self.is_fitted = True

    def _compute_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction errors."""
        self.model.eval()
        errors = []

        # Convert to tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).to(self.device)
        else:
            X_tensor = X.to(self.device)

        with torch.no_grad():
            # Process in batches if needed
            batch_size = 256
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size]

                # Get reconstruction
                if hasattr(self.model, 'forward'):
                    # Assume VAE-like model
                    z, mu, latent_param, recon, embedded = self.model(batch)
                    error = torch.nn.functional.mse_loss(
                        recon, embedded, reduction='none'
                    ).mean(dim=(1, 2, 3))
                else:
                    # Generic autoencoder
                    recon = self.model(batch)
                    error = torch.nn.functional.mse_loss(
                        recon, batch, reduction='none'
                    ).mean(dim=tuple(range(1, batch.dim())))

                errors.append(error.cpu().numpy())

        return np.concatenate(errors)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        errors = self._compute_reconstruction_errors(X)
        return np.where(errors > self.threshold, -1, 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring")

        errors = self._compute_reconstruction_errors(X)

        # Normalize scores
        errors_reshaped = errors.reshape(-1, 1)
        scores = self.scaler.transform(errors_reshaped).flatten()

        return scores


class EnsembleDetector(AnomalyDetector):
    """
    Ensemble of multiple anomaly detectors.
    """

    def __init__(self, detectors: List[AnomalyDetector],
                 combination: str = 'average',
                 weights: Optional[List[float]] = None):
        """
        Args:
            detectors: List of anomaly detectors
            combination: How to combine scores ('average', 'maximum', 'vote')
            weights: Weights for weighted average
        """
        super().__init__()
        self.detectors = detectors
        self.combination = combination
        self.weights = weights

        if weights is not None:
            assert len(weights) == len(detectors)
            self.weights = np.array(weights) / np.sum(weights)

    def fit(self, X: np.ndarray):
        """Fit all detectors."""
        for detector in self.detectors:
            detector.fit(X)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble."""
        if not self.is_fitted:
            raise ValueError("Detectors must be fitted before prediction")

        if self.combination == 'vote':
            # Majority voting
            predictions = np.array([d.predict(X) for d in self.detectors])
            return np.sign(np.sum(predictions, axis=0))
        else:
            # Score-based prediction
            scores = self.score_samples(X)
            threshold = np.median(scores)  # Simple threshold
            return np.where(scores > threshold, -1, 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute ensemble scores."""
        if not self.is_fitted:
            raise ValueError("Detectors must be fitted before scoring")

        scores = []
        for detector in self.detectors:
            detector_scores = detector.score_samples(X)
            # Normalize scores to [0, 1]
            min_score = np.min(detector_scores)
            max_score = np.max(detector_scores)
            if max_score > min_score:
                normalized = (detector_scores - min_score) / (max_score - min_score)
            else:
                normalized = np.zeros_like(detector_scores)
            scores.append(normalized)

        scores = np.array(scores)

        if self.combination == 'average':
            if self.weights is not None:
                return np.average(scores, axis=0, weights=self.weights)
            else:
                return np.mean(scores, axis=0)
        elif self.combination == 'maximum':
            return np.max(scores, axis=0)
        else:
            raise ValueError(f"Unknown combination method: {self.combination}")


def create_baseline_detectors(contamination: float = 0.1) -> Dict[str, AnomalyDetector]:
    """
    Create baseline anomaly detectors for comparison.

    Args:
        contamination: Expected proportion of anomalies

    Returns:
        Dictionary of detectors
    """
    detectors = {}

    # Isolation Forest
    detectors['isolation_forest'] = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )

    # Local Outlier Factor
    detectors['lof'] = LocalOutlierFactor(
        contamination=contamination,
        novelty=True,
        n_neighbors=20
    )

    # Elliptic Envelope (Robust Covariance)
    detectors['elliptic_envelope'] = EllipticEnvelope(
        contamination=contamination,
        random_state=42
    )

    return detectors


def compute_anomaly_metrics(scores: np.ndarray,
                            labels: np.ndarray,
                            pos_label: int = 1) -> Dict[str, float]:
    """
    Compute comprehensive anomaly detection metrics.

    Args:
        scores: Anomaly scores
        labels: True labels (0 for normal, pos_label for anomaly)
        pos_label: Label for positive (anomaly) class

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_recall_curve, roc_curve
    )

    # Convert to binary labels
    binary_labels = (labels == pos_label).astype(int)

    metrics = {}

    # AUC scores
    try:
        metrics['roc_auc'] = roc_auc_score(binary_labels, scores)
        metrics['pr_auc'] = average_precision_score(binary_labels, scores)
    except:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0

    # Find optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(binary_labels, scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Metrics at optimal threshold
    predictions = (scores >= optimal_threshold).astype(int)
    tp = np.sum((predictions == 1) & (binary_labels == 1))
    fp = np.sum((predictions == 1) & (binary_labels == 0))
    tn = np.sum((predictions == 0) & (binary_labels == 0))
    fn = np.sum((predictions == 0) & (binary_labels == 1))

    metrics['optimal_threshold'] = float(optimal_threshold)
    metrics['accuracy'] = (tp + tn) / (tp + fp + tn + fn)
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / \
                    (metrics['precision'] + metrics['recall']) \
        if (metrics['precision'] + metrics['recall']) > 0 else 0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0

    return metrics


def adaptive_threshold(scores: np.ndarray,
                       contamination: float,
                       method: str = 'percentile') -> float:
    """
    Compute adaptive threshold for anomaly detection.

    Args:
        scores: Anomaly scores
        contamination: Expected contamination rate
        method: Thresholding method

    Returns:
        Threshold value
    """
    if method == 'percentile':
        # Simple percentile-based
        threshold = np.percentile(scores, (1 - contamination) * 100)

    elif method == 'gaussian':
        # Assume Gaussian distribution
        mean = np.mean(scores)
        std = np.std(scores)
        z_score = np.abs(np.percentile(np.random.randn(100000),
                                       (1 - contamination) * 100))
        threshold = mean + z_score * std

    elif method == 'iqr':
        # Interquartile range
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr

    elif method == 'mad':
        # Median Absolute Deviation
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        threshold = median + 3 * mad

    else:
        raise ValueError(f"Unknown method: {method}")

    return threshold


def detect_concept_drift(reference_scores: np.ndarray,
                         current_scores: np.ndarray,
                         test: str = 'ks',
                         alpha: float = 0.05) -> Dict[str, Union[bool, float]]:
    """
    Detect concept drift between reference and current score distributions.

    Args:
        reference_scores: Reference (training) scores
        current_scores: Current scores
        test: Statistical test to use ('ks', 'mmd', 'chi2')
        alpha: Significance level

    Returns:
        Dictionary with drift detection results
    """
    from scipy import stats

    results = {}

    if test == 'ks':
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(reference_scores, current_scores)
        results['statistic'] = statistic
        results['p_value'] = p_value
        results['drift_detected'] = p_value < alpha

    elif test == 'mmd':
        # Maximum Mean Discrepancy
        # Simplified implementation
        def gaussian_kernel(x, y, gamma=1.0):
            return np.exp(-gamma * np.sum((x - y) ** 2))

        n_ref = len(reference_scores)
        n_curr = len(current_scores)

        # Sample if too large
        if n_ref > 1000:
            idx = np.random.choice(n_ref, 1000, replace=False)
            reference_scores = reference_scores[idx]
            n_ref = 1000
        if n_curr > 1000:
            idx = np.random.choice(n_curr, 1000, replace=False)
            current_scores = current_scores[idx]
            n_curr = 1000

        # Compute MMD
        mmd = 0
        for i in range(n_ref):
            for j in range(n_ref):
                mmd += gaussian_kernel(reference_scores[i], reference_scores[j])
        mmd /= n_ref * n_ref

        for i in range(n_curr):
            for j in range(n_curr):
                mmd += gaussian_kernel(current_scores[i], current_scores[j])
        mmd /= n_curr * n_curr

        for i in range(n_ref):
            for j in range(n_curr):
                mmd -= 2 * gaussian_kernel(reference_scores[i], current_scores[j])
        mmd /= n_ref * n_curr

        # Simple threshold
        threshold = 0.05
        results['mmd'] = mmd
        results['drift_detected'] = mmd > threshold

    elif test == 'chi2':
        # Chi-square test on binned data
        n_bins = 20
        bins = np.linspace(
            min(reference_scores.min(), current_scores.min()),
            max(reference_scores.max(), current_scores.max()),
            n_bins + 1
        )

        ref_hist, _ = np.histogram(reference_scores, bins=bins)
        curr_hist, _ = np.histogram(current_scores, bins=bins)

        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()

        # Chi-square test
        chi2_stat = np.sum((ref_hist - curr_hist) ** 2 /
                           (ref_hist + curr_hist + 1e-10))
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=n_bins - 1)

        results['chi2_statistic'] = chi2_stat
        results['p_value'] = p_value
        results['drift_detected'] = p_value < alpha

    else:
        raise ValueError(f"Unknown test: {test}")

    return results


def calibrate_scores(scores: np.ndarray,
                     labels: np.ndarray,
                     method: str = 'platt') -> Callable:
    """
    Calibrate anomaly scores to probabilities.

    Args:
        scores: Anomaly scores
        labels: True labels
        method: Calibration method ('platt', 'isotonic')

    Returns:
        Calibration function
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    binary_labels = (labels > 0).astype(int)

    if method == 'platt':
        # Platt scaling (logistic regression)
        lr = LogisticRegression()
        lr.fit(scores.reshape(-1, 1), binary_labels)

        def calibration_func(x):
            return lr.predict_proba(x.reshape(-1, 1))[:, 1]

    elif method == 'isotonic':
        # Isotonic regression
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(scores, binary_labels)

        def calibration_func(x):
            return ir.transform(x)

    else:
        raise ValueError(f"Unknown method: {method}")

    return calibration_func


def compute_uncertainty(model: torch.nn.Module,
                        X: torch.Tensor,
                        n_samples: int = 10,
                        method: str = 'dropout') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute prediction uncertainty for anomaly detection.

    Args:
        model: Trained model
        X: Input data
        n_samples: Number of forward passes
        method: Uncertainty method ('dropout', 'ensemble')

    Returns:
        Tuple of (mean_scores, uncertainty)
    """
    if method == 'dropout':
        # Enable dropout at test time
        model.train()

        scores_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                # Get scores
                if hasattr(model, 'compute_anomaly_scores'):
                    scores = model.compute_anomaly_scores(X)
                else:
                    # Reconstruction-based
                    output = model(X)
                    if isinstance(output, tuple):
                        recon = output[3]  # Assuming VAE
                        embedded = output[4]
                        scores = torch.nn.functional.mse_loss(
                            recon, embedded, reduction='none'
                        ).mean(dim=(1, 2, 3)).cpu().numpy()
                    else:
                        scores = torch.nn.functional.mse_loss(
                            output, X, reduction='none'
                        ).mean(dim=tuple(range(1, X.dim()))).cpu().numpy()

                scores_list.append(scores)

        scores_array = np.array(scores_list)
        mean_scores = np.mean(scores_array, axis=0)
        uncertainty = np.std(scores_array, axis=0)

    else:
        raise ValueError(f"Unknown method: {method}")

    return mean_scores, uncertainty