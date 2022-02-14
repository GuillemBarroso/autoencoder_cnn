from training_data import Data
from cnn_model import Model
from postprocessing import plottingPrediction
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dataset = 'afreightdata'
    printSummary = True
    plotPredictions = True

    data = Data(dataset,verbose=True)
    data.load()
    data.rgb2greyScale()

    model = Model(data,verbose=True)
    model.build(nConvBlocks=3, nFilters=[10, 30, 60], kernelSize= [3, 3, 3], stride=2)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(epochs=500, nBatch=32, earlyStopPatience=10, earlyStopTol=10e-5)
    model.predict()

    if printSummary:
        data.summary()
        model.summary()

    nDisplay = 5
    if plotPredictions:
        plottingPrediction(data, model, nDisplay)

