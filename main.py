import os
import json
import argparse
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

import torch
import torch.optim as optim

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from model.vae import CorrelatedGaussianVAE
from config import VAE_DATASETS  # use unified VAE dataset config
from utils.data_utils import create_train_val_loaders
from utils.general_utils import (
    set_seed,
    create_experiment_directory,
    setup_logger,
    save_config,
    get_gpu_memory_usage,
)
from training.trainer import VAETrainer
from training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    GradientMonitor,
    MetricTracker,
    ProgressBar,
    WarmupScheduler,
)

# ========= dataset config helpers =========
def _select_dataset_cfg(name: str):
    key = name.lower()
    if key in {"ton-iot", "toniot"}:
        key = "ton_iot"
    elif key in {"cicids-2017", "cic"}:
        key = "cicids2017"
    elif key in {"cicids-2018"}:
        key = "cicids2018"

    if key not in VAE_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(VAE_DATASETS.keys())}")
    return VAE_DATASETS[key]


def _apply_dataset_cfg_to_args(args: argparse.Namespace, ds_cfg) -> argparse.Namespace:
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


def kl_weight_for_epoch(epoch: int, args: argparse.Namespace) -> float:
    if not args.use_kl_annealing:
        return args.kld_weight_max
    if epoch < args.kld_warmup_epochs:
        return args.kld_weight_min
    progress = (epoch - args.kld_warmup_epochs) / max(1, args.kl_anneal_period)
    return args.kld_weight_min + (args.kld_weight_max - args.kld_weight_min) * max(0.0, min(1.0, progress))


def _log_loader_summaries(loaders: Dict[str, Any]) -> None:
    for name, loader in loaders.items():
        ds = loader.dataset
        ds_name = type(ds).__name__
        if hasattr(ds, "dataset"):
            ds_name = f"Subset({type(ds.dataset).__name__})"
        logging.info(
            f"{name}: ds={ds_name}, N={len(ds)}, bs={loader.batch_size}, "
            f"steps/epoch={len(loader)}, sampler={type(loader.sampler).__name__}, "
            f"drop_last={loader.drop_last}, workers={loader.num_workers}, pin={loader.pin_memory}"
        )

def _safe(v, default=0.0):
    try:
        import math
        return default if (v is None or math.isnan(v) or math.isinf(v)) else float(v)
    except Exception:
        return default


def _build_metrics(train_m: dict, val_m: dict, kl_w: float) -> dict:
    m = {
        "loss":       _safe(train_m.get("total_loss")),
        "recon_loss": _safe(train_m.get("recon_loss")),
        "kld_loss":   _safe(train_m.get("kld_loss")),
        # "reg_loss":   _safe(train_m.get("reg_loss")),
    }

    val_loss = _safe(val_m.get("val_loss") or val_m.get("total_loss"))
    v_recon  = _safe(val_m.get("v_recon") or val_m.get("recon_loss") or val_m.get("val_recon_loss"))
    v_kld    = _safe(val_m.get("v_kld")   or val_m.get("kld_loss")   or val_m.get("val_kld_loss"))
    # v_reg    = _safe(val_m.get("v_reg")   or val_m.get("reg_loss")   or val_m.get("val_reg_loss"))

    m.update(
        val_loss=val_loss,
        v_recon=v_recon,
        v_kld=v_kld,
        # v_reg=v_reg,
        # val_recon_loss=v_recon,
        # val_kld_loss=v_kld,
        # val_reg_loss=v_reg,
        kl_weight=float(kl_w),
    )
    return m

# ========= build model & optimizer =========
def build_model_and_optim(
    args: argparse.Namespace,
    device: torch.device,
    dataset_cfg: Optional[object] = None,
) -> Tuple[torch.nn.Module, optim.Optimizer, Any, VAETrainer]:
    # Optional encoder/decoder structural config from dataset_cfg
    encoder_cfg = decoder_cfg = None
    if dataset_cfg is not None:
        m = dataset_cfg.model
        encoder_cfg = {
            "packet_latent_dim": getattr(m, "packet_latent_dim", 128),
            "d_model": getattr(m, "session_d_model", 128),
            "latent_dim": args.latent_dim,
            "spatial_reduction_size": getattr(m, "spatial_reduction_size", 8),
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

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=args.scheduler_patience,
            factor=args.scheduler_factor,
            min_lr=args.min_lr,
        )

    trainer = VAETrainer(model, device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        f"Model: {model.__class__.__name__} | mode={args.training_mode} | "
        f"params: total={total:,}, trainable={trainable:,}"
    )
    return model, optimizer, scheduler, trainer

# ========= training main loop =========

def train(args: argparse.Namespace):
    # 1) pick dataset config & push into args
    ds_cfg = _select_dataset_cfg(args.dataset)
    args = _apply_dataset_cfg_to_args(args, ds_cfg)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)

    exp_name = args.experiment_name or f"vae_{args.dataset}_{args.training_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = create_experiment_directory(args.output_dir, exp_name, timestamp=False)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    setup_logger(log_file=os.path.join(out_dir, "logs", "training.log"), log_level="INFO")
    logging.info(f"Experiment dir: {out_dir}")
    try:
        logging.info(f"GPU mem: {get_gpu_memory_usage()}")
    except Exception:
        pass

    # Save resolved config (CLI + dataset defaults)
    save_config(args, out_dir, filename="config.json")

    # DataLoader
    train_loader, val_loader = create_train_val_loaders(
        data_path=args.train_data_path,
        args=args,
        val_split=args.val_split,
        seed=args.seed,
        augment=args.use_augmentation,
    )
    logging.info("Data loaders:")
    _log_loader_summaries({"train": train_loader, "val": val_loader})

    # Model & optim
    model, optimizer, scheduler, trainer = build_model_and_optim(args, device, ds_cfg)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        if args.resume_epoch is not None:
            start_epoch = args.resume_epoch
        logging.info(f"Resumed from {args.resume}, start_epoch={start_epoch}")

    # Callbacks
    warmup = WarmupScheduler(
        optimizer,
        args.warmup_epochs,
        args.warmup_initial_lr,
        args.learning_rate,
        args.warmup_method,
    ) if args.warmup_epochs > 0 else None

    es = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        mode="min",
        restore_best_weights=True,
        verbose=True,
    ) if args.use_early_stopping else None

    ckpt_best = ModelCheckpoint(
        filepath=os.path.join(out_dir, "checkpoints", "best_epoch_{epoch}.pth"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=True,
    )
    ckpt_periodic = ModelCheckpoint(
        filepath=os.path.join(out_dir, "checkpoints", "epoch_{epoch}.pth"),
        monitor="val_loss",
        mode="min",
        save_best_only=False,
        save_weights_only=False,
        verbose=False,
    ) if (not args.save_best_only and args.save_interval > 0) else None

    lr_monitor = LearningRateMonitor(log_file=os.path.join(out_dir, "logs", "lrs.txt"))
    grad_monitor = GradientMonitor(log_interval=max(1, args.grad_log_interval), verbose=True)
    metric_tracker = MetricTracker(save_path=os.path.join(out_dir, "logs", "metrics.json"))
    pbar = ProgressBar(total_epochs=args.num_epochs, width=50)

    best_val = float("inf")

    for epoch in range(start_epoch, args.num_epochs):
        if warmup:
            warmup(epoch)
        kl_w = kl_weight_for_epoch(epoch, args)

        # train
        train_m = trainer.train_epoch(
            train_loader,
            optimizer,
            kld_weight=kl_w,
            free_bits=args.free_bits,
            cov_offdiag_weight=args.cov_offdiag_weight,
            ortho_weight=args.ortho_weight,
            capacity=args.capacity,
            capacity_weight=args.capacity_weight,
        )
        grad_monitor(model)

        # val
        val_m = trainer.evaluate(
            val_loader,
            kld_weight=kl_w,
            free_bits=args.free_bits,
            cov_offdiag_weight=args.cov_offdiag_weight,
            ortho_weight=args.ortho_weight,
            capacity=args.capacity,
            capacity_weight=args.capacity_weight,
        )

        trainer.run_diagnostics(train_loader, val_loader, epoch)

        if scheduler is not None:
            scheduler.step(_safe(val_m.get("val_loss") or val_m.get("total_loss")))
        lr_monitor(optimizer, epoch)

        metrics = _build_metrics(train_m, val_m, kl_w)
        metric_tracker(epoch, metrics)
        pbar(epoch, metrics)
        if es is not None:
            if es(metrics["val_loss"], model, epoch=epoch):
                logging.info("Early stopping triggered.")
                break

        # NOTE: keep call signature consistent with ModelCheckpoint.__call__
        ckpt_best(epoch, model, metrics, optimizer)
        if ckpt_periodic is not None and (epoch + 1) % args.save_interval == 0:
            ckpt_periodic(epoch, model, metrics, optimizer)

        best_val = min(best_val, metrics["val_loss"])

    logging.info(f"Training finished. Best val loss={best_val:.4f}. Outputs at: {out_dir}")
    return trainer, out_dir

# ========= Argument Parser =========
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train CorrelatedGaussianVAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # dataset / mode
    p.add_argument("--dataset", type=str, choices=["ton_iot", "cicids2017", "cicids2018"], default="cicids2017")
    p.add_argument("--training_mode", type=str, choices=["diagonal", "correlated", "gmm", "ae"], default="correlated")

    # model / optim（部分值会被 dataset-config 覆盖）
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_components", type=int, default=5)
    p.add_argument("--dropout_rate", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_epochs", type=int, default=60)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)

    # KL / regularizers
    p.add_argument("--use_kl_annealing", action="store_true", default=True)
    p.add_argument("--kld_weight_min", type=float, default=0.1)
    p.add_argument("--kld_weight_max", type=float, default=1.0)
    p.add_argument("--kl_anneal_period", type=int, default=10)
    p.add_argument("--kld_warmup_epochs", type=int, default=2)
    p.add_argument("--free_bits", type=float, default=0.0)
    p.add_argument("--cov_offdiag_weight", type=float, default=1e-4)
    p.add_argument("--ortho_weight", type=float, default=1e-4)
    p.add_argument("--capacity", type=float, default=40.0)
    p.add_argument("--capacity_weight", type=float, default=1.0)

    # scheduler / ES / warmup
    p.add_argument("--use_scheduler", action="store_true", default=False)
    p.add_argument("--scheduler_patience", type=int, default=2)
    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-6)

    p.add_argument("--use_early_stopping", action="store_true", default=True)
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--early_stopping_min_delta", type=float, default=1e-6)

    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--warmup_initial_lr", type=float, default=1e-4)
    p.add_argument("--warmup_method", type=str, default="linear", choices=["linear", "exponential"])

    # augmentation（透传到 ByteDataset）
    p.add_argument("--use_augmentation", action="store_true", default=False)
    p.add_argument("--corruption_prob", type=float, default=0.01)
    p.add_argument("--packet_drop_prob", type=float, default=0.001)
    p.add_argument("--mask_prob", type=float, default=0.0)
    p.add_argument("--noise_std", type=float, default=0.0)
    p.add_argument("--num_packets", type=int, default=None)
    p.add_argument("--val_split", type=float, default=0.1)

    # system / IO
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--experiment_name", type=str, default=None)
    p.add_argument("--save_interval", type=int, default=5)
    p.add_argument("--save_best_only", action="store_true", default=False)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_log_interval", type=int, default=50)

    # resume
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--resume_epoch", type=int, default=45)

    return p

def main():
    parser = build_argparser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
