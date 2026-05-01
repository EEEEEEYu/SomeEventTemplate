"""OmegaConf-backed structured config for training runs.

Loaded by train.py at startup. The schema is the single source of truth for
which fields a config file may contain — unrecognised fields raise.

The original template's `file_name`/`class_name` dynamic-import pattern has been
replaced with a `name` + `args` pair on DATA and MODEL: train.py owns the
explicit registry of {name -> class}. This keeps imports static and refactor-safe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf, MISSING


def _register_resolvers_once():
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
        OmegaConf.register_new_resolver("div", lambda x, y: x / y)
        OmegaConf.register_new_resolver("add", lambda x, y: x + y)
        OmegaConf.register_new_resolver("subtract", lambda x, y: x - y)


_register_resolvers_once()


@dataclass
class TrainingConfig:
    seed: int = 42
    max_epochs: int = 50
    deterministic: bool = False
    inference_mode: bool = False
    num_sanity_val_steps: int = 2
    use_compile: bool = False
    # `torch.set_float32_matmul_precision`. "highest" is PyTorch's default
    # (full precision matmul); "high" enables TF32 / TensorFloat-32 on Ampere+
    # and L40S, ~10–20% speedup on tensor-core ops with no accuracy regression
    # for our use case (the difflogic relaxation is already a stochastic
    # estimator). Lightning warns to set this when running on tensor-core GPUs.
    matmul_precision: str = "highest"
    # `torch.compile` the LightningModule after construction. Fuses
    # `bin_op_s`'s 16-op Python loop into a single CUDA kernel. First-step cold
    # start adds 20–60 s; leave false on short debug runs.
    compile_model: bool = False


@dataclass
class DiagnosticsConfig:
    # Per-layer-group gradient-norm logger. Models declare `layer_groups` as a
    # dict of prefix lists; the callback emits `gradnorm/<group>` and
    # `gradnorm/__total__`. Off by default; useful when an encoder/decoder
    # split exists.
    gradient_norm_logger: bool = False
    gradient_norm_log_every_n_steps: int = 50


@dataclass
class DistributedConfig:
    accelerator: str = "auto"
    devices: Any = 1
    num_nodes: int = 1
    strategy: str = "auto"


@dataclass
class DataloaderConfig:
    batch_size: int = 128
    test_batch_size: Optional[int] = None
    num_workers: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    multiprocessing_context: Optional[str] = None
    drop_last: bool = False
    shuffle_train: bool = True
    shuffle_val: bool = False
    shuffle_test: bool = False


@dataclass
class DataConfig:
    name: str = MISSING
    args: Dict[str, Any] = field(default_factory=dict)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)


@dataclass
class ModelConfig:
    name: str = MISSING
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GradientClipConfig:
    enabled: bool = False
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"


@dataclass
class GradientAccumulationConfig:
    enabled: bool = False
    scheduling: Dict[int, int] = field(default_factory=dict)


@dataclass
class SWAConfig:
    enabled: bool = False
    swa_lrs: float = 1e-2


@dataclass
class OptimizerConfig:
    name: str = "Adam"
    arguments: Dict[str, Any] = field(default_factory=lambda: {"lr": 1e-3})
    gradient_clip: GradientClipConfig = field(default_factory=GradientClipConfig)
    gradient_accumulation: GradientAccumulationConfig = field(default_factory=GradientAccumulationConfig)
    stochastic_weight_averaging: SWAConfig = field(default_factory=SWAConfig)


@dataclass
class LearningRateSchedulerConfig:
    enabled: bool = False
    name: str = "CosineAnnealingLR"
    arguments: Dict[str, Any] = field(default_factory=dict)
    interval: str = "step"


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    monitor: str = "val_loss_epoch"
    mode: str = "min"
    patience: int = 5
    min_delta: float = 1e-5


@dataclass
class SchedulerConfig:
    learning_rate: LearningRateSchedulerConfig = field(default_factory=LearningRateSchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class LoggerConfig:
    log_dir_root: str = "lightning_logs"
    experiment_name: str = "run"


@dataclass
class CheckpointConfig:
    enabled: bool = True
    every_n_epochs: int = 1
    monitor: str = "val_loss_epoch"
    mode: str = "min"
    filename: str = "best-{epoch:03d}-{val_loss_epoch:.5f}"
    save_top_k: int = 1
    save_last: bool = True


@dataclass
class AppConfig:
    TRAINING: TrainingConfig = field(default_factory=TrainingConfig)
    DISTRIBUTED: DistributedConfig = field(default_factory=DistributedConfig)
    DATA: DataConfig = field(default_factory=DataConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    OPTIMIZER: OptimizerConfig = field(default_factory=OptimizerConfig)
    SCHEDULER: SchedulerConfig = field(default_factory=SchedulerConfig)
    LOGGER: LoggerConfig = field(default_factory=LoggerConfig)
    CHECKPOINT: CheckpointConfig = field(default_factory=CheckpointConfig)
    DIAGNOSTICS: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


def _validate(cfg: AppConfig) -> List[str]:
    """Structural validation. MODEL.name / DATA.name presence is enforced at the
    registry-lookup site in train.py — which is skipped on --dry-run, so base.yaml
    (defaults file with empty names) parses fine here."""
    errors: List[str] = []
    if cfg.DATA.dataloader.batch_size <= 0:
        errors.append("DATA.dataloader.batch_size must be positive.")
    if cfg.TRAINING.max_epochs <= 0:
        errors.append("TRAINING.max_epochs must be positive.")
    if not cfg.OPTIMIZER.name:
        errors.append("OPTIMIZER.name must be specified.")
    if not cfg.OPTIMIZER.arguments:
        errors.append("OPTIMIZER.arguments must provide keyword args.")
    lr_sched = cfg.SCHEDULER.learning_rate
    if lr_sched.enabled and not lr_sched.name:
        errors.append("SCHEDULER.learning_rate.name required when enabled.")
    return errors


def load_config(path: str) -> AppConfig:
    user_cfg = OmegaConf.load(path)
    merged = OmegaConf.merge(OmegaConf.structured(AppConfig), user_cfg)
    cfg: AppConfig = OmegaConf.to_object(merged)
    errors = _validate(cfg)
    if errors:
        bullet = "\n - ".join(errors)
        raise ValueError(f"Configuration validation failed:\n - {bullet}")
    return cfg


def config_to_dict(cfg: AppConfig) -> Dict[str, Any]:
    return asdict(cfg)
