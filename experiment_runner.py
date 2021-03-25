import matplotlib.pyplot as plt
import tensorflow as tf

from the_empire_strikes_back.evolutionary.evolve import evolve
from fitness_function.fitness_function import FitnessFunction
from the_empire_strikes_back.evolutionary.chromosome import Chromosome
from the_empire_strikes_back.model.model import ConvolutedModelWrapper


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

fitness = FitnessFunction()

"""
model = ConvolutedModelWrapper()
model.initialise()
chromosome = Chromosome()
chromosome.generate(model.get_weights())
profit, amount, ratio, equity = fitness.calculate(chromosome, 'train')
exit()
"""

best_chromosome = evolve()

profit_train, amount_train, ratio_train, equity = fitness.calculate(best_chromosome, 'train')
print(profit_train)
print(amount_train)
print(ratio_train)
plt.plot(equity)
plt.title('Equity line train')
plt.show()

print("\n")

profit_test, amount_test, ratio_test, equity = fitness.calculate(best_chromosome, 'test')
print(profit_test)
print(amount_test)
print(ratio_test)
plt.plot(equity)
plt.title('Equity line test')
plt.show()

fitness.save_signals_plots(best_chromosome, 'train')
fitness.save_signals_plots(best_chromosome, 'test')
