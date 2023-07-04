import bentoml
import tensorflow as tf

from pathlib import Path
from MediLeaf_AI.entity.config_entity import DeploymentConfig, PrepareBaseModelConfig


class Deployment:
    def __init__(self, config: DeploymentConfig, prepare_base_model_config: PrepareBaseModelConfig):
        self.config = config
        self.prepare_base_model_config = prepare_base_model_config

    def _save_to_bento(self, model:  tf.keras.Model):
        bentoml.tensorflow.save_model(self.config.model_tag, model)

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def deploy(self):
        model = self.load_model(self.config.path_of_model.joinpath(
            Path(self.prepare_base_model_config.params_pre_trained_model)))
        self._save_to_bento(model)
