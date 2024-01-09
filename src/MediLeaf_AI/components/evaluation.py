import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report
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
        model = self.load_model(os.path.join(self.training_config.trained_model_path, Path(self.prepare_base_model_config.params_pre_trained_model) ))
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)
        self.predictions = model.predict(self.valid_generator)
        self.predicted_labels = np.argmax(self.predictions, axis=1)
        self.actual_labels = self.valid_generator.classes
        self.confusion_mtx = confusion_matrix(
            self.actual_labels, self.predicted_labels)
        self.class_labels = list(self.valid_generator.class_indices.keys())
        self.plot_confusion_matrix()

    def save_score(self):
        scores = {"loss": self.score[0], "top1_accuracy": self.score[1], "top5_accuracy": self.score[2],
                   "precision": self.score[3], "recall": self.score[4],
                   "auc": self.score[5]}
        save_json(path=Path("evaluation_scores.json"), data=scores)

    def plot_confusion_matrix(self):
        plt.clf()
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.confusion_mtx, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_labels, yticklabels=self.class_labels, cbar=False)
        plt.title(f'Confusion Matrix ({self.prepare_base_model_config.params_pre_trained_model})')
        plt.xlabel('Predicted Labels')
        plt.ylabel('Actual Labels')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.evaluation_metrics_dir, Path(self.prepare_base_model_config.params_pre_trained_model)))
