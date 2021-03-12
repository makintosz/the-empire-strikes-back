from typing import List

import numpy as np
import matplotlib.pyplot as plt

from the_empire_strikes_back.config.evolutionary import (
    PMC,
    PMG
)


class Chromosome:
    def __init__(self):
        self.weights = None

    def __str__(self):
        return f'Chromosome with: {len(self.weights)} layers'

    def generate(
        self,
        chromosome_format: List[np.ndarray]
    ) -> None:
        """ Generates random values of genes in chromosome for certain
        shape of weights given from the model. """
        self.weights = []
        for array in chromosome_format:
            self.weights.append(np.random.normal(
                loc=0,
                scale=0.01,
                size=array.shape)
            )

    def get_weights(self) -> List[np.ndarray]:
        """ Returns weights of the chromosome for fitness function. """
        return self.weights

    def mutate(self) -> None:
        """ Randomly mutates weights of the chromosome. """
        rand_float = np.random.uniform(0, 1)
        if rand_float < PMC:
            for array in self.weights:
                if array.ndim == 1:
                    for i in range(len(array)):
                        random_float = np.random.uniform(0, 1)
                        if random_float < PMG:
                            array[i] = np.random.uniform(-0.01, 0.01)

                if array.ndim == 4:
                    for i in range(array.shape[0]):
                        for j in range(array.shape[1]):
                            for k in range(array.shape[2]):
                                for l in range(array.shape[3]):
                                    random_float = np.random.uniform(0, 1)
                                    if random_float < PMG:
                                        array[i, j, k, l] = np.random.uniform(
                                            -0.01,
                                            0.01,
                                        )

                if array.ndim == 2:
                    for i in range(array.shape[0]):
                        for j in range(array.shape[1]):
                            random_float = np.random.uniform(0, 1)
                            if random_float < PMG:
                                array[i, j] = np.random.uniform(
                                    -0.01,
                                    0.01,
                                )





