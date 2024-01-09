import os
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path

from MediLeaf_AI.entity.config_entity import TrainingConfig, PrepareBaseModelConfig
from MediLeaf_AI.utils.common import save_json


class Training:
    def __init__(self, config: TrainingConfig, prepare_base_model_config: PrepareBaseModelConfig):
        self.config = config
        self.prepare_base_model_config = prepare_base_model_config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            os.path.join(self.config.updated_base_model_path, Path(self.prepare_base_model_config.params_pre_trained_model)),
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
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

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        self.plot_training_history(self.history, "top1_accuracy", os.path.join(self.config.training_graphs_path, Path(self.prepare_base_model_config.params_pre_trained_model + "_top1_accuracy")), "Training and Validation Top-1 Accuracy")
        self.plot_training_history(self.history, "top5_accuracy", os.path.join(self.config.training_graphs_path, Path(self.prepare_base_model_config.params_pre_trained_model + "_top5_accuracy")), "Training and Validation Top-5 Accuracy")
        self.plot_training_history(self.history, "loss", os.path.join(self.config.training_graphs_path, Path(self.prepare_base_model_config.params_pre_trained_model + "_loss")), "Training and Validation Loss")
        self.plot_training_history(self.history, "precision", os.path.join(self.config.training_graphs_path, Path(self.prepare_base_model_config.params_pre_trained_model + "_precision")), "Training and Validation Precision")
        self.plot_training_history(self.history, "recall", os.path.join(self.config.training_graphs_path, Path(self.prepare_base_model_config.params_pre_trained_model + "_recall")), "Training and Validation Recall")
        self.plot_training_history(self.history, "auc", os.path.join(self.config.training_graphs_path, Path(self.prepare_base_model_config.params_pre_trained_model + "_auc")), "Training and Validation AUC Score")

        self.save_model(
            path=os.path.join(self.config.trained_model_path, Path(self.prepare_base_model_config.params_pre_trained_model)),
            model=self.model
        )

    def plot_training_history(self, history, title, plot_path, graph_title):
        plt.clf()
        plt.plot(history.history[title])
        plt.plot(history.history["val_" + title])
        plt.title(graph_title+" "+"("+self.prepare_base_model_config.params_pre_trained_model+")")
        plt.xlabel("epochs")
        plt.ylabel(title)
        plt.legend([title, "val_" + title])
        plt.savefig(plot_path)

    def save_score(self):
        scores = {"loss": self.history.history['loss'][self.config.params_epochs-1],
                  "top1_accuracy": self.history.history['top1_accuracy'][self.config.params_epochs-1],
                  "top5_accuracy": self.history.history['top5_accuracy'][self.config.params_epochs-1],
                  "precision": self.history.history['precision'][self.config.params_epochs-1],
                  "recall": self.history.history['recall'][self.config.params_epochs-1],
                  "auc": self.history.history['auc'][self.config.params_epochs-1]}
        save_json(path=Path("training_scores.json"), data=scores)
