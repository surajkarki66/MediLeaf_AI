import os
import sys
import tensorflow as tf

from pathlib import Path

from MediLeaf_AI.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        if self.config.params_pre_trained_model == 'inceptionv3':
            self.model = tf.keras.applications.InceptionV3(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )
        elif self.config.params_pre_trained_model == 'mobilenet':
            self.model = tf.keras.applications.MobileNet(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )
        elif self.config.params_pre_trained_model == 'mobilenetv2':
            self.model = tf.keras.applications.MobileNetV2(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )
        elif self.config.params_pre_trained_model == 'mobilenetv3':
            self.model = tf.keras.applications.MobileNetV3(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )

        elif self.config.params_pre_trained_model == 'nasnetmobile':
            self.model = tf.keras.applications.NASNetMobile(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )
            
        elif self.config.params_pre_trained_model == 'densenet201':
            self.model = tf.keras.applications.DenseNet201(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )

        else:
            print("Provided model is not available to use.")
            sys.exit(1)

        self.save_model(path=os.path.join(self.config.base_model_path, Path(
            self.config.params_pre_trained_model + ".keras")), model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate, optimizer):
        if freeze_all:
            for _ in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for _ in model.layers[:-freeze_till]:
                model.trainable = False

        global_avg_in = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(global_avg_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        elif optimizer == 'adagrad':
            opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

        else:
            print("Provided optimizer is not available to use.")
            sys.exit(1)

        full_model.compile(
            optimizer=opt,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.TopKCategoricalAccuracy(
                    name="top1_accuracy", k=1),
                tf.keras.metrics.TopKCategoricalAccuracy(
                    name="top5_accuracy"),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(
                    name="auc", multi_label=True, num_labels=classes, from_logits=False, label_weights=None),

            ]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
            optimizer=self.config.optimizer
        )

        self.save_model(path=os.path.join(self.config.updated_base_model_path, Path(self.config.params_pre_trained_model + ".keras")),
                        model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
