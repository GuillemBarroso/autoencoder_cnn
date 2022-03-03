from training_data import Data
from fcnn_model import FCNN
from model import Model
from postprocessing import plottingPrediction
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    dataset = 'beam'
    plotPredictions = True

    data = Data(dataset, verbose=True, saveInfo=True)
    data.load()
    data.rgb2greyScale()
    data.blackAndWhite()
    data.rehsapeDataToArray()

    fcnn = FCNN(data,verbose=True, saveInfo=True)
    fcnn.build(codeSize=25, nNeurons=40, nHidLayers=2)

    model = Model(fcnn)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(epochs=50, nBatch=16, earlyStopPatience=50, earlyStopTol=10e-5)
    model.predict()

    nDisplay = 5
    if plotPredictions:
        plottingPrediction(data, model, nDisplay)

