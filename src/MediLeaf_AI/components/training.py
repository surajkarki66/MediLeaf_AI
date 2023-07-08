import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path

from MediLeaf_AI.entity.config_entity import TrainingConfig, PrepareBaseModelConfig


class Training:
    def __init__(self, config: TrainingConfig, prepare_base_model_config: PrepareBaseModelConfig):
        self.config = config
        self.prepare_base_model_config = prepare_base_model_config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path.joinpath(
                Path(self.prepare_base_model_config.params_pre_trained_model))
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

        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        self.plot_training_history(history, "accuracy", self.config.training_metrics_path.joinpath(
            Path(self.prepare_base_model_config.params_pre_trained_model + "_accuracy")))
        self.plot_training_history(history, "loss", self.config.training_metrics_path.joinpath(
            Path(self.prepare_base_model_config.params_pre_trained_model + "_loss")))
        self.plot_training_history(history, "auc", self.config.training_metrics_path.joinpath(
            Path(self.prepare_base_model_config.params_pre_trained_model + "_auc")))
        
        self.save_model(
            path=self.config.trained_model_path.joinpath(
                Path(self.prepare_base_model_config.params_pre_trained_model)),
            model=self.model
        )

    def plot_training_history(self, history, title, plot_path):
        plt.clf()
        plt.plot(history.history[title])
        plt.plot(history.history["val_" + title])
        plt.title(self.prepare_base_model_config.params_pre_trained_model.upper())
        plt.xlabel("epochs")
        plt.ylabel(title)
        plt.legend([title, "val_" + title])
        plt.savefig(plot_path)
