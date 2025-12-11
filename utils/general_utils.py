import torch
import numpy as np
import random
import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import shutil
import sys
from datetime import datetime
import psutil
import GPUtil

import argparse
from typing import Union, Dict

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to {seed}")

def setup_training_environment(device: Optional[str] = None) -> torch.device:
    """
    Setup training environment and return device.

    Args:
        device: Device string ('cuda', 'cpu', 'cuda:0', etc.)

    Returns:
        PyTorch device object
    """
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device_obj = torch.device(device)
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")

    if device_obj.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")

        torch.cuda.set_per_process_memory_fraction(0.95)
        if hasattr(torch.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")

    return device_obj


def save_config(config: Union[argparse.Namespace, Dict],
                output_dir: str,
                filename: str = 'config.json'):
    """
    Save configuration to JSON file.

    Args:
        config: Configuration object or dictionary
        output_dir: Directory to save config
        filename: Config filename
    """
    # Convert Namespace to dict if needed
    if hasattr(config, '__dict__'):
        config_dict = vars(config)
    else:
        config_dict = config

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert non-serializable objects
    config_dict = make_serializable(config_dict)

    # Save to JSON
    config_path = os.path.join(output_dir, filename)
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, sort_keys=True)

    logging.info(f"Configuration saved to {config_path}")

    # Also save as YAML for readability
    yaml_path = config_path.replace('.json', '.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from file.

    Args:
        config_path: Path to config file (JSON or YAML)

    Returns:
        Configuration dictionary
    """
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path}")

    return config


def make_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable format.

    Args:
        obj: Object to convert

    Returns:
        Serializable object
    """
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif hasattr(obj, '__dict__'):
        return make_serializable(obj.__dict__)
    else:
        return obj


def create_experiment_directory(base_dir: str,
                                experiment_name: Optional[str] = None,
                                timestamp: bool = True) -> str:
    """
    Create directory for experiment outputs.

    Args:
        base_dir: Base output directory
        experiment_name: Experiment name
        timestamp: Whether to add timestamp

    Returns:
        Path to experiment directory
    """
    if experiment_name is None:
        experiment_name = "experiment"

    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{experiment_name}_{timestamp_str}"

    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Create subdirectories
    subdirs = ['checkpoints', 'logs', 'plots', 'results']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)

    logging.info(f"Created experiment directory: {experiment_dir}")

    return experiment_dir


def setup_logger(log_file: Optional[str] = None,
                 log_level: str = 'INFO',
                 log_format: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_file: Path to log file (None for console only)
        log_level: Logging level
        log_format: Custom log format

    Returns:
        Logger instance
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger


def get_gpu_memory_usage() -> Dict[str, float]:

    if not torch.cuda.is_available():
        return {}

    # PyTorch CUDA memory
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9

    # System GPU memory
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assume using first GPU
        total = gpu.memoryTotal / 1e3
        used = gpu.memoryUsed / 1e3
        free = gpu.memoryFree / 1e3
    else:
        total = used = free = 0

    return {
        'allocated': allocated,
        'reserved': reserved,
        'total': total,
        'used': used,
        'free': free,
        'utilization': (used / total * 100) if total > 0 else 0
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def save_model_architecture(model: torch.nn.Module,
                            save_path: str,
                            input_shape: Optional[tuple] = None):
    """
    Save model architecture summary.

    Args:
        model: PyTorch model
        save_path: Path to save summary
        input_shape: Input shape for torchsummary
    """
    with open(save_path, 'w') as f:
        # Basic info
        f.write("Model Architecture\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model class: {model.__class__.__name__}\n")

        # Parameter count
        param_count = count_parameters(model)
        f.write(f"\nParameters:\n")
        f.write(f"  Total: {param_count['total']:,}\n")
        f.write(f"  Trainable: {param_count['trainable']:,}\n")
        f.write(f"  Non-trainable: {param_count['non_trainable']:,}\n")

        # Model structure
        f.write(f"\nModel Structure:\n")
        f.write(str(model))

        # Detailed summary if possible
        if input_shape:
            try:
                from torchsummary import summary
                f.write(f"\n\nDetailed Summary:\n")
                # Redirect stdout to capture summary
                import io
                from contextlib import redirect_stdout

                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    summary(model, input_shape)
                f.write(buffer.getvalue())
            except:
                pass


def create_backup(source_dir: str,
                  backup_dir: str,
                  include_patterns: List[str] = None,
                  exclude_patterns: List[str] = None):
    """
    Create backup of code and configurations.

    Args:
        source_dir: Source directory
        backup_dir: Backup destination
        include_patterns: Patterns to include
        exclude_patterns: Patterns to exclude
    """
    if include_patterns is None:
        include_patterns = ['*.py', '*.yaml', '*.json', '*.txt']

    if exclude_patterns is None:
        exclude_patterns = ['__pycache__', '*.pyc', '.git']

    os.makedirs(backup_dir, exist_ok=True)

    # Copy files
    for root, dirs, files in os.walk(source_dir):
        # Filter directories
        dirs[:] = [d for d in dirs if not any(
            d.match(pattern) for pattern in exclude_patterns
        )]

        for file in files:
            file_path = os.path.join(root, file)

            # Check if file should be included
            should_include = any(
                Path(file).match(pattern) for pattern in include_patterns
            )
            should_exclude = any(
                Path(file).match(pattern) for pattern in exclude_patterns
            )

            if should_include and not should_exclude:
                # Compute relative path
                rel_path = os.path.relpath(file_path, source_dir)
                dest_path = os.path.join(backup_dir, rel_path)

                # Create destination directory
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Copy file
                shutil.copy2(file_path, dest_path)

    logging.info(f"Created backup in: {backup_dir}")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0

def save_checkpoint_atomic(state: Dict,
                           filepath: str,
                           is_best: bool = False,
                           backup: bool = True):
    """
    Save checkpoint atomically to prevent corruption.

    Args:
        state: State dictionary to save
        filepath: Target file path
        is_best: Whether this is the best model
        backup: Whether to keep backup of previous checkpoint
    """
    # Save to temporary file first
    temp_filepath = filepath + '.tmp'
    torch.save(state, temp_filepath)

    # Backup existing checkpoint if requested
    if backup and os.path.exists(filepath):
        backup_filepath = filepath + '.backup'
        shutil.copy2(filepath, backup_filepath)

    # Atomic rename
    os.replace(temp_filepath, filepath)
    if is_best:
        best_filepath = filepath.replace('.pth', '_best.pth')
        shutil.copy2(filepath, best_filepath)

    logging.info(f"Checkpoint saved: {filepath}")


def load_checkpoint_safe(filepath: str,
                         map_location: Optional[Union[str, torch.device]] = None) -> Dict:
    try:
        checkpoint = torch.load(filepath, map_location=map_location)
        return checkpoint
    except Exception as e:
        logging.warning(f"Failed to load checkpoint: {e}")
        backup_filepath = filepath + '.backup'
        if os.path.exists(backup_filepath):
            logging.info("Attempting to load backup checkpoint...")
            return torch.load(backup_filepath, map_location=map_location)
        else:
            raise


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:

    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return {
        'param_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024,
        'total_size_mb': size_mb
    }


class AverageMeter:
    def __init__(self, name: str = 'Meter'):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'{self.name}: {self.val:.4f} (avg: {self.avg:.4f})'

class Timer:
    def __init__(self, name: str = 'Timer', verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = 0

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.verbose:
            print(f"{self.name} took {format_time(self.elapsed)}")