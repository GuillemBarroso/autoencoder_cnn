from common.training_data import Data
from common.fcnn import FCNN
from common.model import Model
from common.postprocessing import plottingPrediction
import numpy as np


if __name__ == '__main__':
    dataset = 'beam_homog'
    plotPredictions = True

    # Define test data
    mu1 = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
    mu2 = [round(x,2) for x in np.arange(0, 202.5, 22.5)]
    
    data = Data(dataset, testData=[mu1, mu2], verbose=True, saveInfo=True)
    data.load()
    data.rehsapeDataToArray()

    fcnn = FCNN(data,verbose=True, saveInfo=True)
    fcnn.build(codeSize=25, nNeurons=200, nHidLayers=4, regularisation=1e-4)

    model = Model(fcnn)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(epochs=500, nBatch=16, earlyStopPatience=50, earlyStopTol=1e-4)
    model.predict()

    imgDisplay = 5
    # mu1_test = [1.0, 1.0, 1.15, 1.25, 1.25]
    # mu2_test = [0, 180, 45, 0, 180]
    # imgDisplay = [mu1_test, mu2_test]
    if plotPredictions:
        plottingPrediction(data, model, imgDisplay)

