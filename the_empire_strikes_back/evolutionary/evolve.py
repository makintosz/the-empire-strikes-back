import time
from copy import deepcopy

import numpy as np

from fitness_function.fitness_function import FitnessFunction
from the_empire_strikes_back.model.model import ConvolutedModelWrapper
from the_empire_strikes_back.evolutionary.evo_utils import (
    generate_population,
    select_chromosomes,
    mutate_population,
    crossover_population,
    calculate_fitness,
    show_plots
)
from the_empire_strikes_back.evolutionary.chromosome import Chromosome
from the_empire_strikes_back.config.evolutionary import (
    N_GENERATIONS
)


def evolve() -> Chromosome:
    """ Main evolutionary process. """
    model = ConvolutedModelWrapper()
    model.initialise()
    model.print_summary()
    fitness = FitnessFunction()
    population = generate_population()
    best, mean = [], []
    best_current = -1
    best_chromosome = None
    for gdx in range(N_GENERATIONS):
        print('\nPokolenie: {}/{}'.format(gdx+1, N_GENERATIONS))
        start = time.time()
        population_fitness, population = calculate_fitness(
            model,
            fitness,
            deepcopy(population)
        )
        print("Fitness calculation: " + str(time.time() - start))
        mean.append(np.mean(population_fitness))
        best_population = max(population_fitness)
        best_chromosome_population = deepcopy(
            population[
                population_fitness.index(max(population_fitness))
            ]
        )
        if best_population > best_current:
            best_current = best_population
            best_chromosome = best_chromosome_population

        best.append(best_current)
        population_new = select_chromosomes(population_fitness, population)
        start = time.time()
        population_new = mutate_population(population_new)
        print("Mutation calculation: " + str(time.time() - start))
        start = time.time()
        population_new = crossover_population(population_new)
        print("Crossover calculation: " + str(time.time() - start))
        population = population_new[:]

    show_plots(mean, best)
    return best_chromosome



