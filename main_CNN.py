from common.training_data import Data
from common.cnn import CNN
from common.model import Model
from common.postprocessing import plottingPrediction


if __name__ == '__main__':
    dataset = 'beam_simp_txt_4'
    plotPredictions = True

    data = Data(dataset,verbose=True, saveInfo=True)
    data.load()
    # data.rgb2greyScale()
    # data.blackAndWhite()

    cnn = CNN(data,verbose=True, saveInfo=True)
    cnn.build(nConvBlocks=2, codeSize=25, codeRegCoef=1e-4, nFilters=[20, 20], kernelSize=[3, 3], stride=2,
        nHidLayers=2, nNeurons=400)

    model = Model(cnn)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(epochs=200, nBatch=16, earlyStopPatience=50, earlyStopTol=1e-8)
    model.predict()

    nDisplay = 5
    if plotPredictions:
        plottingPrediction(data, model, nDisplay)

