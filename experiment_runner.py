from the_empire_strikes_back.model.model import ConvolutedModelWrapper
from the_empire_strikes_back.data.data_loading import load_data


data = load_data()
model = ConvolutedModelWrapper()
model.initialise()
model.set_weights()
pred = model.make_predictions(data)

