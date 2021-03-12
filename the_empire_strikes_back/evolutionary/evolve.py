import time

import numpy as np
import matplotlib.pyplot as plt

from fitness_function.fitness_function import FitnessFunction
from the_empire_strikes_back.model.model import ConvolutedModelWrapper
from the_empire_strikes_back.evolutionary.evo_utils import (
    generate_population,
    select_chromosomes,
    mutate_population,
    crossover_population,
)
from the_empire_strikes_back.config.evolutionary import (
    N_GENERATIONS
)


def evolve() -> None:
    """ Main evolutionary process. """
    model = ConvolutedModelWrapper()
    model.initialise()
    fitness = FitnessFunction()
    population = generate_population()
    best, mean = [], []
    best_current = -1
    for gdx in range(N_GENERATIONS):
        print('\nPokolenie: {}'.format(gdx))
        population_fitness = []
        population_counter = []
        start = time.time()
        for chromosome in population:
            profit, counter = fitness.calculate(chromosome)
            while counter == 0:
                print('zero trans')
                chromosome.generate(model.get_weights())
                profit, counter = fitness.calculate(chromosome)

            population_fitness.append(profit)
            population_counter.append(counter)

        print("Fitness calculation: " + str(time.time() - start))
        mean.append(np.mean(population_fitness))
        best_population = max(population_fitness)
        if best_population > best_current:
            best_current = best_population

        best.append(best_current)

        start = time.time()
        population_new = select_chromosomes(population_fitness, population)
        print("Selection calculation: " + str(time.time() - start))

        start = time.time()
        population_new = mutate_population(population_new)
        print("Mutation calculation: " + str(time.time() - start))

        start = time.time()
        population_new = crossover_population(population_new)
        print("Crossover calculation: " + str(time.time() - start))
        population = population_new[:]

    plt.plot(mean)
    plt.plot(best)
    plt.show()

