from typing import Tuple, List
from copy import deepcopy

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from fitness_function.fitness_function import FitnessFunction
from the_empire_strikes_back.evolutionary.chromosome import Chromosome
from the_empire_strikes_back.model.model import ConvolutedModelWrapper
from the_empire_strikes_back.config.evolutionary import (
    POP_SIZE,
    PCC,
    PCG
)


def generate_population() -> List[Chromosome]:
    """ Generates population of randomly initialised chromosomes. """
    model = ConvolutedModelWrapper()
    model.initialise()
    population = []
    for n in range(POP_SIZE):
        chromosome = Chromosome()
        chromosome.generate(model.get_weights())
        population.append(chromosome)

    return population


def calculate_fitness(
    model: ConvolutedModelWrapper,
    fitness: FitnessFunction,
    population: List[Chromosome]
) -> Tuple[list, List[Chromosome]]:
    """ Calculates fitness of entire population. Regenerates chromosome when
    zero transactions. """
    population_fitness = []
    for chromosome in population:
        profit, counter, _, __ = fitness.calculate(chromosome, 'train')
        while counter == 0:
            chromosome.generate(model.get_weights())
            profit, counter, _, __ = fitness.calculate(chromosome, 'train')
            print(profit)
            print(counter)

        population_fitness.append(profit)

    if check_if_same_elements(population_fitness):
        print('Taka sama populacja. Reset populacji.')
        print(population_fitness)
        population_fitness = []
        population = generate_population()
        for chromosome in population:
            profit, counter, _, __ = fitness.calculate(chromosome, 'train')
            while counter == 0:
                chromosome.generate(model.get_weights())
                profit, counter, _, __ = fitness.calculate(chromosome, 'train')

            population_fitness.append(profit)

    return population_fitness, population


def select_chromosomes(
    population_fitness: list,
    population: List[Chromosome]
) -> List[Chromosome]:
    """ Selects chromosomes to the next population based on roullet
     selection of its fitness results. """
    fitness_scaler = MinMaxScaler()
    fitness_population = np.array(population_fitness)
    fitness_population = fitness_scaler.fit_transform(
        fitness_population.reshape((-1, 1))
    ).flatten().tolist()
    sum_fitness_population = sum(fitness_population)
    probability_population = []
    for fitness in fitness_population:
        probability_population.append(fitness / sum_fitness_population)

    distribution_population = [0.0]
    distribution_temp = 0.0
    for prob in probability_population:
        distribution_temp += prob
        distribution_population.append(distribution_temp)

    distribution_population[-1] = 1.0
    population_new = []
    for j in range(POP_SIZE):
        random_prob = np.random.uniform(0, 1)
        for k in range(POP_SIZE):
            if distribution_population[k] <= random_prob <= \
                    distribution_population[k + 1]:
                population_new.append(deepcopy(population[k]))

    return population_new


def mutate_population(population: List[Chromosome]) -> List[Chromosome]:
    """ Mutates the population of chromosomes. """
    for chromosome in population:
        chromosome.mutate()

    return population


def crossover_population(
    population: List[Chromosome]
) -> List[Chromosome]:
    """ Conducts crossover on the given population. """
    to_cross = []
    for c in range(POP_SIZE):
        random_prob = np.random.uniform(0, 1)
        if random_prob < PCC:
            to_cross.append(c)

    if is_even(to_cross) == 0:
        del to_cross[-1]

    np.random.shuffle(to_cross)
    for pdx in range(0, len(to_cross), 2):
        (
            population[to_cross[pdx]],
            population[to_cross[pdx + 1]]
        ) = chromosomes_crossover(
            population[to_cross[pdx]],
            population[to_cross[pdx + 1]])

    return population


def chromosomes_crossover(
    chromosome1: Chromosome,
    chromosome2: Chromosome
) -> Tuple[Chromosome, Chromosome]:
    """ Crosses two chromosomes together. """
    for adx, array in enumerate(chromosome1.weights):
        if array.ndim == 1:
            for i in range(len(array)):
                if np.random.uniform(0, 1) < PCG:
                    weight_temp_c1 = chromosome1.weights[adx][i]
                    weight_temp_c2 = chromosome2.weights[adx][i]
                    chromosome1.weights[adx][i] = weight_temp_c2
                    chromosome2.weights[adx][i] = weight_temp_c1

        if array.ndim == 2:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    if np.random.uniform(0, 1) < PCG:
                        weight_temp_c1 = chromosome1.weights[adx][i, j]
                        weight_temp_c2 = chromosome2.weights[adx][i, j]
                        chromosome1.weights[adx][i, j] = weight_temp_c2
                        chromosome2.weights[adx][i, j] = weight_temp_c1

        if array.ndim == 4:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    for k in range(array.shape[2]):
                        for l in range(array.shape[3]):
                            if np.random.uniform(0, 1) < PCG:
                                weight_temp_c1 = chromosome1.weights[adx][
                                    i, j, k, l
                                ]
                                weight_temp_c2 = chromosome2.weights[adx][
                                    i, j, k, l
                                ]
                                chromosome1.weights[adx][
                                    i, j, k, l
                                ] = weight_temp_c2
                                chromosome2.weights[adx][
                                    i, j, k, l
                                ] = weight_temp_c1

    return chromosome1, chromosome2


def is_even(seq: list) -> int:
    """ Function checks if list has an even number of elements. """
    if len(seq) % 2 == 0:
        return 1
    else:
        return 0


def show_plots(mean: list, best: list) -> None:
    """ Shows plots of training process for evolutionary optimisation. """
    plt.Figure(figsize=(18, 9))
    plt.plot(mean)
    plt.plot(best)
    plt.grid()
    plt.show()


def check_if_same_elements(population: list) -> bool:
    """ Checks if all elements in a list have the same value. """
    if len(set(population)) == 1:
        return True
    else:
        return False
