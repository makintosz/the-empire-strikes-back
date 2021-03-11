import numpy as np

from the_empire_strikes_back.data.data_transformation import load_transform_save
from the_empire_strikes_back.config.general import TRANSFORM_DATA
from the_empire_strikes_back.config.fitness import (
    TIMEFRAMES,
)


def load_data() -> np.ndarray:
    """ Returns dict with loaded dataframes. """
    if TRANSFORM_DATA:
        load_transform_save()

    data = np.load('data/data.npy')
    return data
