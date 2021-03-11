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
                input_shape=MODEL_ARCHITECTURE['input_shape']
            )
        )

    def make_predictions(self, x: np.ndarray):
        pass

    def get_weights(self):
        return self.model.get_weights()
