from training_data import Data
from cnn_model import Model
from postprocessing import plottingPrediction, plottingCodeFiltres

if __name__ == '__main__':
    dataset = 'afreightdata'
    printSummary = True
    plotPredictions = True
    plotCode = False

    data = Data(dataset,verbose=True)
    data.load()
    data.rgb2greyScale()

    model = Model(data,verbose=True)
    model.build(nConvLayers=2, nFilters=10, kernel=3, stride=2)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(epochs=2, nBatch=32, earlyStopPatience=10)
    model.predict()

    if printSummary:
        data.summary()
        model.summary()

    if plotPredictions:
        nDisplay = 5
        plottingPrediction(data, model, nDisplay)

    if plotCode:
        nDisplay = 5
        nDisplayCodeFilters = 10
        plottingCodeFiltres(model,nDisplay,nDisplayCodeFilters)

