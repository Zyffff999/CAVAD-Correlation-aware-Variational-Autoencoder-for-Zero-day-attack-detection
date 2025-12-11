from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class VAEModelConfig:
    # 词表 & embedding
    vocab_size: int = 257
    emb_dim: int = 128

    # Packet / Session 结构
    packet_latent_dim: int = 256      # SessionEncoder(packet_latent_dim)
    session_d_model: int = 256        # SessionEncoder(d_model)
    latent_dim: int = 32              # 最终 VAE latent 维度，和 VAE 里的 latent_dim 一致

    # Packet TCN
    header_tcn_channels: List[int] = field(default_factory=lambda: [256, 192, 128])
    payload_tcn_channels: List[int] = field(default_factory=lambda: [256, 192, 128])
    header_strides: List[int] = field(default_factory=lambda: [2, 1, 1])  # 128 -> 64
    payload_strides: List[int] = field(default_factory=lambda: [2, 1, 1])
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.1
    tcn_use_se: bool = True
    tcn_causal: bool = True

    # cross attention
    cross_attn_heads: int = 4
    cross_attn_dropout: float = 0.1

    # 空间压缩
    spatial_reduction_size: int = 8  # adaptive_avg_pool1d 输出长度

    # Session 级 TCN
    num_packets: int = 16
    packet_len: int = 256
    session_tcn_channels: List[int] = field(default_factory=lambda: [192, 128, 32])
    session_tcn_causal: bool = True
    session_aggregator: str = "mean"   # 预留：mean / mean_max / attn

    # Decoder
    decoder_rank: int = 256
    decoder_num_layers: int = 1        # ResidualBlock1 层数


@dataclass
class VAETrainConfig:
    # 训练基本超参
    batch_size: int = 128
    num_epochs: int = 60
    lr: float = 1e-5
    weight_decay: float = 1e-5

    # KL / 退火
    use_kl_annealing: bool = True
    kld_weight_min: float = 0.1
    kld_weight_max: float = 1.0
    kl_anneal_period: int = 10
    kld_warmup_epochs: int = 2
    free_bits: float = 0.0
    capacity: float = 40.0
    capacity_weight: float = 1.0

    # 正则
    cov_offdiag_weight: float = 1e-4
    ortho_weight: float = 1e-4

    # 其他
    val_split: float = 0.1
    num_workers: int = 0
    seed: int = 42

    # Scheduler
    use_scheduler: bool = False
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-6


@dataclass
class VAEDatasetConfig:
    name: str
    train_data_path: str
    test_data_path: str
    attack_mapping_path: str
    test_benign_label: int

    model: VAEModelConfig
    train: VAETrainConfig


CICIDS2017_VAE = VAEDatasetConfig(
    name="cicids2017",
    train_data_path=r"D:\OOD_detect\data\IDS_2017_processed\sessions_16_h128_p128\train_benign_ready.npz",
    # test_data_path=r"D:\OOD_detect\data\IDS_2018-1\Session-16-h128-p128\test.npz",
    # attack_mapping_path=r"D:\OOD_detect\data\IDS_2018-1\Session-16-h128-p128\label_map.json",
    # test_benign_label=0,
    test_data_path=r"D:\OOD_detect\data\IDS_2017_processed\sessions_16_h128_p128\test_mixed_ready.npz",
    attack_mapping_path=r"D:\OOD_detect\data\IDS_2017_processed\sessions_16_h128_p128\test_mixed_label_map.json",
    test_benign_label=2,


    model=VAEModelConfig(
        vocab_size=257,
        emb_dim=128,
        packet_latent_dim=256,
        session_d_model=256,
        latent_dim=64,
        header_tcn_channels=[256, 192, 128],
        payload_tcn_channels=[256, 192, 128],
        header_strides=[2, 1, 1],
        payload_strides=[2, 1, 1],
        spatial_reduction_size=8,
        num_packets=16,
        packet_len=256,
        session_tcn_channels=[192, 128, 64],
        session_tcn_causal=True,
        decoder_rank=256,
        decoder_num_layers=1,
    ),

    train=VAETrainConfig(
        batch_size=128,
        num_epochs=50,
        lr=1e-5,
        weight_decay=1e-5,
        use_kl_annealing=True,
        kld_weight_min=0.1,
        kld_weight_max=0.7,
        kl_anneal_period=10,
        kld_warmup_epochs=5,
        free_bits=0.0,
        capacity=40.0,
        capacity_weight=1.0,
        cov_offdiag_weight=1e-4,
        ortho_weight=1e-4,
        val_split=0.1,
        num_workers=0,
        use_scheduler=False,
        use_early_stopping=True,
        early_stopping_patience=5,
        early_stopping_min_delta=1e-6,
    ),
)

CICIDS_2018_VAE = VAEDatasetConfig(
    name="cicids2018",
    train_data_path=r"D:\OOD_detect\data\IDS_2018-1\Session-16-h128-p128\train.npz",
    # test_data_path=r"D:\OOD_detect\data\IDS_2018-1\Session-16-h128-p128\test.npz",
    # attack_mapping_path=r"D:\OOD_detect\data\IDS_2018-1\Session-16-h128-p128\label_map.json",
    test_data_path=r"D:\OOD_detect\data\IDS_2017_processed\sessions_16_h128_p128\test_mixed_ready.npz",
    attack_mapping_path=r"D:\OOD_detect\data\IDS_2017_processed\sessions_16_h128_p128\test_mixed_label_map.json",
    test_benign_label=2,
    # test_benign_label=0,

    model=VAEModelConfig(
        vocab_size=257,
        emb_dim=128,
        packet_latent_dim=256,
        session_d_model=256,
        latent_dim=64,
        header_tcn_channels=[256, 192, 128],
        payload_tcn_channels=[256, 192, 128],
        header_strides=[2, 1, 1],
        payload_strides=[2, 1, 1],
        spatial_reduction_size=64,
        num_packets=16,
        packet_len=256,
        session_tcn_channels=[192, 128, 64],
        session_tcn_causal=True,
        decoder_rank=256,
        decoder_num_layers=1,
    ),

    train=VAETrainConfig(
        batch_size=128,
        num_epochs=80,
        lr=2e-5,
        weight_decay=1e-5,
        use_kl_annealing=True,
        kld_weight_min=0.02,
        kld_weight_max=0.5,
        kl_anneal_period=20,
        kld_warmup_epochs=5,
        free_bits=0.5,
        capacity=40.0,
        capacity_weight=0.0,
        cov_offdiag_weight=5e-5,
        ortho_weight=5e-5,
        val_split=0.1,
        num_workers=0,
        use_scheduler=True,
        scheduler_patience=4,
        scheduler_factor=0.5,
        min_lr=3e-6,
        use_early_stopping=True,
        early_stopping_patience=8,
        early_stopping_min_delta=1e-5,
    ),
)

# ========= TON_IoT VAE 配置 =========
TONIOT_VAE = VAEDatasetConfig(
    name="ton_iot",
    train_data_path=r"data/NOT-IOT/Session-16-h128-p128/train.npz",
    test_data_path=r"data/NOT-IOT/Session-16-h128-p128/test.npz",
    attack_mapping_path=r"data/NOT-IOT/Session-16-h128-p128/label_map.json",
    # test_data_path=r"D:\OOD_detect\data\IDS_2018-1\Session-16-h128-p128\test.npz",
    # attack_mapping_path=r"D:\OOD_detect\data\IDS_2018-1\Session-16-h128-p128\label_map.json",
    test_benign_label=0,

    model=VAEModelConfig(
        vocab_size=257,
        emb_dim=128,
        packet_latent_dim=128,
        session_d_model=128,
        latent_dim=32,

        header_tcn_channels=[196, 156, 96],
        payload_tcn_channels=[196, 156, 96],
        header_strides=[2, 1, 1],
        payload_strides=[2, 1, 1],

        spatial_reduction_size=64,
        num_packets=16,
        packet_len=256,

        session_tcn_channels=[128, 96, 64],
        session_tcn_causal=True,

        decoder_rank=256,
        decoder_num_layers=1,
    ),

    train=VAETrainConfig(
        batch_size=128,
        num_epochs=80,
        lr=1e-5,  # slightly higher is ok
        weight_decay=1e-5,

        use_kl_annealing=True,
        kld_weight_min=0.01,
        kld_weight_max=0.3,  # lower than 0.3
        kl_anneal_period=20,
        kld_warmup_epochs=5,

        free_bits=0.0,  # small; 0.4–0.6 is big for dim=32
        capacity=0.0,  # disable
        capacity_weight=0.0,

        cov_offdiag_weight=0.0,
        ortho_weight=0.0,

        val_split=0.1,
        use_scheduler=True,
        scheduler_patience=6,
        scheduler_factor=0.5,
        min_lr=3e-6,

        use_early_stopping=True,
        early_stopping_patience=12,
    )
)

VAE_DATASETS: Dict[str, VAEDatasetConfig] = {
    CICIDS2017_VAE.name: CICIDS2017_VAE,
    TONIOT_VAE.name: TONIOT_VAE,
    CICIDS_2018_VAE.name: CICIDS_2018_VAE
}