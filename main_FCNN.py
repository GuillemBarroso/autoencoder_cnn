from common.training_data import Data
from common.fcnn import FCNN
from common.model import Model
from common.postprocessing import plottingPrediction


if __name__ == '__main__':
    dataset = 'mnist'
    plotPredictions = True

    data = Data(dataset, verbose=True, saveInfo=True)
    data.load()
    data.rgb2greyScale()
    data.blackAndWhite()
    data.rehsapeDataToArray()

    fcnn = FCNN(data,verbose=True, saveInfo=True)
    fcnn.build(codeSize=9, nNeurons=20, nHidLayers=1, regularisation=1e-4)

    model = Model(fcnn)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(epochs=500, nBatch=16, earlyStopPatience=50, earlyStopTol=10e-5)
    model.predict()

    nDisplay = 5
    if plotPredictions:
        plottingPrediction(data, model, nDisplay)

