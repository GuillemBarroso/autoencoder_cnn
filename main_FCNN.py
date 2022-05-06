from common.training_data import Data
from common.fcnn import FCNN
from common.model import Model
from common.postprocessing import plottingPrediction

import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    # dataset = 'afreightdata'
    dataset = 'beam_simp_txt_4'
    plotPredictions = True              

    data = Data(dataset, verbose=True, saveInfo=True)
    data.load()
    # plt.imshow(data.x_test[0])
    # data.rgb2greyScale()
    # plt.imshow(data.x_test[0])

    data.rehsapeDataToArray()

    fcnn = FCNN(data,verbose=True, saveInfo=True)
    fcnn.build(codeSize=25, nNeurons=200, nHidLayers=4, regularisation=1e-4)

    model = Model(fcnn)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(epochs=500, nBatch=16, earlyStopPatience=50, earlyStopTol=10e-8)
    model.predict()

    nDisplay = 5
    if plotPredictions:
        plottingPrediction(data, model, nDisplay)

