from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_pre_trained_model: str
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path
    params_early_stopping_monitor: str
    params_early_stopping_min_delta: float
    params_early_stopping_patience: int
    params_early_stopping_mode: str
    params_early_stopping_is_restore_best_weight: bool


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_dir: Path
    training_graphs_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_graphs_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    evaluation_metrics_dir: Path
    params_image_size: list
    params_batch_size: int
