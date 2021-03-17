from typing import Tuple

import numpy as np

from the_empire_strikes_back.data.data_transformation import (
    load_transform_save
)
from the_empire_strikes_back.config.general import TRANSFORM_DATA


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Returns data for training. """
    if TRANSFORM_DATA:
        load_transform_save('train')

    data = np.load('data/data_train.npy')
    data_prices = np.load('data/prices_train.npy')

    return data, data_prices


def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Returns data for testing. """
    if TRANSFORM_DATA:
        load_transform_save('test')

    data = np.load('data/data_test.npy')
    data_prices = np.load('data/prices_test.npy')

    return data, data_prices
