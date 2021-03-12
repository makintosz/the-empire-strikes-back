from the_empire_strikes_back.model.model import ConvolutedModelWrapper
from the_empire_strikes_back.data.data_loading import load_data
from the_empire_strikes_back.evolutionary.evolve import evolve
from the_empire_strikes_back.evolutionary.chromosome import Chromosome


data = load_data()
chromosome = Chromosome()
model = ConvolutedModelWrapper()
model.initialise()
chromosome.generate(model.get_weights())
