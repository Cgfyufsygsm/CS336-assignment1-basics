from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    train_data: str = ""
    val_data: str = ""
    dtype: str = "uint16"
    device: str = "cuda"


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    n_layers: int = 4   
    n_heads: int = 16
    d_ff: int = 1344
    dropout: float = 0.0
    tie_embeddings: bool = True


@dataclass
class OptimConfig:
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    eps: float = 1e-8
    grad_clip: float = 1.0


@dataclass
class ScheduleConfig:
    warmup_iters: int = 1000
    cosine_cycle_iters: int = 20000
    min_lr: float = 3e-5


@dataclass
class TrainConfig:
    batch_size: int = 64
    max_iters: int = 20000
    eval_every: int = 500
    eval_iters: int = 100
    log_every: int = 50
    seed: int = 42
    experiment_name: str = "toy_tinystories"


@dataclass
class CheckpointConfig:
    every: int = 1000
    resume: bool = False


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_project: str = "cs336_lab1"
    wandb_run_name: str = ""


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def set_nested_attr(obj: object, path: str, value: object) -> None:
    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
