from typing import List

import numpy as np
import matplotlib.pyplot as plt


class Chromosome:
    def __init__(self):
        self.weights = []

    def generate(
        self,
        chromosome_format: List[np.ndarray]
    ) -> None:
        """ Generates random values of genes in chromosome for certain
        shape of weights given from the model. """
        for array in chromosome_format:
            self.weights.append(np.random.normal(
                loc=0,
                scale=0.01,
                size=array.shape)
            )

    def get_weights(self):
        """ Returns weights of the chromosome for fitness function. """
        return self.weights

    def mutate(self):
        """ Randomly mutates weights of the chromosome. """
        pass


