from training_data import Data
from cnn_model import Model
from postprocessing import plottingPrediction


if __name__ == '__main__':
    dataset = 'mnist'
    plotPredictions = True

    data = Data(dataset,verbose=True, saveInfo=True)
    data.load()
    data.rgb2greyScale()
    model = Model(data,verbose=True, saveInfo=True)
    model.build(nConvBlocks=2, nFilters=[30, 60], kernelSize= [3, 3], stride=2)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(epochs=1, nBatch=256, earlyStopPatience=10, earlyStopTol=10e-5)
    model.predict()

    nDisplay = 5
    if plotPredictions:
        plottingPrediction(data, model, nDisplay)

