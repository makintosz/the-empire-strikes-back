from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from the_empire_strikes_back.config.model import MODEL_ARCHITECTURE


class ConvolutedModelWrapper:
    """ Class with CNN model and it's interface. """
    def __init__(self):
        self.model = None

    def initialise(self):
        self.model = keras.models.Sequential()
        self.model.add(
            keras.layers.Conv2D(
                MODEL_ARCHITECTURE['l1_filters'],
                MODEL_ARCHITECTURE['l1_size'],
                activation=MODEL_ARCHITECTURE['l1_activation'],
                padding=MODEL_ARCHITECTURE['l1_padding'],
                input_shape=MODEL_ARCHITECTURE['input_shape'],
            )
        )
        self.model.add(
            keras.layers.MaxPooling2D(
                MODEL_ARCHITECTURE['l2_pooling'],
            )
        )
        self.model.add(
            keras.layers.Flatten()
        )
        self.model.add(
            keras.layers.Dense(
                MODEL_ARCHITECTURE['l3_dense'],
                activation=MODEL_ARCHITECTURE['l3_activation'],
            )
        )
        self.model.add(
            keras.layers.Dense(
                MODEL_ARCHITECTURE['l4_dense'],
                activation=MODEL_ARCHITECTURE['l4_activation'],
            )
        )
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='sgd',
            metrics=['accuracy']
        )

    def make_predictions(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights: List[np.ndarray]):
        """ Sets new weights from a chromosome. """
        self.model.set_weights(weights)
