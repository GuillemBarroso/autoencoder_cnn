from training_data import Data
from cnn import CNN
from model import Model
from postprocessing import plottingPrediction


if __name__ == '__main__':
    dataset = 'beam'
    plotPredictions = True

    data = Data(dataset,verbose=True, saveInfo=True)
    data.load()
    data.rgb2greyScale()
    data.blackAndWhite()

    cnn = CNN(data,verbose=True, saveInfo=True)
    cnn.build(nConvBlocks=2, nFilters=[10, 20], kernelSize= [3, 3], stride=2)

    model = Model(cnn)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(epochs=50, nBatch=16, earlyStopPatience=10, earlyStopTol=10e-5)
    model.predict()

    nDisplay = 5
    if plotPredictions:
        plottingPrediction(data, model, nDisplay)

