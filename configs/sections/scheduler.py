from dataclasses import dataclass, field
from typing import Any, Dict
from configs.config_tracker import TrackedConfigMixin

@dataclass
class LearningRateSchedulerConfig(TrackedConfigMixin):
    enabled: bool = False
    name: str = "CosineAnnealingLR"
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EarlyStoppingConfig(TrackedConfigMixin):
    enabled: bool = False
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 10
    min_delta: float = 0.0


@dataclass
class SchedulerConfig(TrackedConfigMixin):
    learning_rate: LearningRateSchedulerConfig = field(default_factory=LearningRateSchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
