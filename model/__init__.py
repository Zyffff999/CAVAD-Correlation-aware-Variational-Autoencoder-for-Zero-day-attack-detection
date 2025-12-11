from .vae import CorrelatedGaussianVAE
from .packet import PacketEncoder, FactorizedDecoder
from .losses import vae_loss_full, vae_elbo, compute_reconstruction_loss, compute_kl_divergence

__all__ = [
    'CorrelatedGaussianVAE',
    'PacketEncoder',
    'vae_loss_full',
    'vae_elbo',
    'compute_reconstruction_loss',
    'compute_kl_divergence'
]

# training/__init__.py
"""
Training package containing trainer and callbacks
"""

from training.trainer import VAETrainer
from training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    GradientMonitor,
    MetricTracker,
    ProgressBar,
    WarmupScheduler,
    CallbackList
)

__all__ = [
    'VAETrainer',
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateMonitor',
    'GradientMonitor',
    'MetricTracker',
    'ProgressBar',
    'WarmupScheduler',
    'CallbackList'
]

# evaluation/__init__.py
"""
Evaluation package containing metrics, visualization, and anomaly detection
"""

from evaluation.metrics import (
    compute_optimal_threshold,
    compute_binary_metrics,
    analyze_per_category_performance,
    compute_detection_delay,
    compute_stability_metrics,
    compute_confidence_intervals,
    compute_calibration_metrics,
    compute_statistical_tests,
    compute_multi_class_metrics
)

from evaluation.visualization import (
    plot_score_distributions,
    plot_roc_and_pr_curves,
    plot_confusion_matrix,
    create_detailed_visualizations,
    plot_latent_space_2d,
    plot_reconstruction_comparison,
    plot_calibration_curve,
    create_summary_dashboard
)

from evaluation.anomaly_detection import (
    AnomalyDetector,
    MahalanobisDetector,
    ReconstructionDetector,
    EnsembleDetector,
    create_baseline_detectors,
    compute_anomaly_metrics,
    adaptive_threshold,
    detect_concept_drift,
    calibrate_scores,
    compute_uncertainty
)

__all__ = [
    # Metrics
    'compute_optimal_threshold',
    'compute_binary_metrics',
    'analyze_per_category_performance',
    'compute_detection_delay',
    'compute_stability_metrics',
    'compute_confidence_intervals',
    'compute_calibration_metrics',
    'compute_statistical_tests',
    'compute_multi_class_metrics',

    # Visualization
    'plot_score_distributions',
    'plot_roc_and_pr_curves',
    'plot_confusion_matrix',
    'create_detailed_visualizations',
    'plot_latent_space_2d',
    'plot_reconstruction_comparison',
    'plot_calibration_curve',
    'create_summary_dashboard',

    # Anomaly Detection
    'AnomalyDetector',
    'MahalanobisDetector',
    'ReconstructionDetector',
    'EnsembleDetector',
    'create_baseline_detectors',
    'compute_anomaly_metrics',
    'adaptive_threshold',
    'detect_concept_drift',
    'calibrate_scores',
    'compute_uncertainty'
]

# utils/__init__.py
"""
Utilities package containing data and general utilities
"""

from utils.data_utils import (
    create_train_val_loaders,
    create_test_loader,
    create_streaming_loader,

)

from utils.general_utils import (
    set_seed,
    setup_training_environment,
    save_config,
    load_config,
    make_serializable,
    create_experiment_directory,
    setup_logger,
    get_gpu_memory_usage,
    count_parameters,
    save_model_architecture,
    create_backup,
    format_time,
    get_learning_rate,
    save_checkpoint_atomic,
    load_checkpoint_safe,
    get_model_size,
    AverageMeter,
    Timer
)

__all__ = [
    'create_train_val_loaders',
    'create_test_loader',
    'load_attack_mapping',
    'create_subset_loader',
    'split_by_label',
    'create_balanced_loader',
    'compute_dataset_statistics',
    'create_streaming_loader',
    'save_data_split',
    'load_data_split',
    'DataPrefetcher',

    'set_seed',
    'setup_training_environment',
    'save_config',
    'load_config',
    'make_serializable',
    'create_experiment_directory',
    'setup_logger',
    'get_gpu_memory_usage',
    'count_parameters',
    'save_model_architecture',
    'create_backup',
    'format_time',
    'get_learning_rate',
    'save_checkpoint_atomic',
    'load_checkpoint_safe',
    'get_model_size',
    'AverageMeter',
    'Timer'
]