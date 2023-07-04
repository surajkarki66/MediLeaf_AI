import os
import time
import numpy as np
import tensorflow as tf

from pathlib import Path

from MediLeaf_AI.entity.config_entity import PrepareCallbacksConfig, PrepareBaseModelConfig, TrainingConfig


class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig, prepare_base_config: PrepareBaseModelConfig, training_config: TrainingConfig):
        self.config = config
        self.prepare_base_config = prepare_base_config
        self.training_config = training_config
        self.decay_rate = self.prepare_base_config.params_learning_rate / \
            self.training_config.params_epochs

    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath.joinpath(
                Path(self.prepare_base_config.params_pre_trained_model)),
            save_best_only=True
        )

    @property
    def _create_early_stopping_callbacks(self):
        return tf.keras.callbacks.EarlyStopping(monitor=self.config.params_early_stopping_monitor,
                                                min_delta=self.config.params_early_stopping_min_delta,
                                                patience=self.config.params_early_stopping_patience,
                                                mode=self.config.params_early_stopping_mode,
                                                restore_best_weights=self.config.params_early_stopping_is_restore_best_weight)

    @property
    def _create_lr_schedular_callbacks(self):
        def exp_decay(epoch):
            lrate = self.prepare_base_config.params_learning_rate * \
                np.exp(-self.decay_rate*epoch)
            return lrate
        return tf.keras.callbacks.LearningRateScheduler(exp_decay)

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks,
            self._create_early_stopping_callbacks,
            self._create_lr_schedular_callbacks
        ]
