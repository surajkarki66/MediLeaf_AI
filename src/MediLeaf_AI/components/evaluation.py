import tensorflow as tf
from pathlib import Path
from MediLeaf_AI.entity.config_entity import EvaluationConfig, PrepareBaseModelConfig, TrainingConfig
from MediLeaf_AI.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig,  prepare_base_model_config: PrepareBaseModelConfig, training_config: TrainingConfig):
        self.config = config
        self.prepare_base_model_config = prepare_base_model_config
        self.training_config = training_config

    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        model = self.load_model(self.training_config.trained_model_path.joinpath(
            Path(self.prepare_base_model_config.params_pre_trained_model)))
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
