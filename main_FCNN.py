from common.training_data import Data
from common.fcnn import FCNN
from common.model import Model
from common.postprocessing import plottingPrediction


if __name__ == '__main__':
    dataset = 'beam_homog'
    plotPredictions = True

    # Define test data
    # mu1 = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
    # mu1 = [0.6, 1.35, 2.2]
    # mu1 = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
    #        1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9,
    #        2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7]
    # mu1 = [0.8, 0.85, 1.45, 1.5, 2.2, 2.25]
    mu1 = [1.4, 1.45, 1.5, 1.55]

    # mu2 = [round(x,2) for x in np.arange(0, 202.5, 22.5)]
    # mu2 = [0.0, 22.5, 45, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0]
    # mu2 = [67.5]
    mu2 = [67.5, 90, 112.5]
    # testData = [mu1, mu2]
    testData = 0.1

    # Load data
    data = Data(dataset, testData=testData, verbose=True, saveInfo=True)
    data.load()
    data.rehsapeDataToArray()

    # Create NN
    fcnn = FCNN(data,verbose=True, saveInfo=True)
    fcnn.build(codeSize=25, nNeurons=200, nHidLayers=4, regularisation=1e-4)

    # Build, compile and train model
    model = Model(fcnn)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(epochs=500, nBatch=12, earlyStopPatience=100, earlyStopTol=1e-4)
    model.predict()

    # Results visualisation
    mu1_test = [1.4, 1.4, 1.45, 1.5, 1.55, 1.55]
    mu2_test = [67.5, 90, 67.5, 112.5, 90, 112.5]
    # imgDisplay = [mu1_test, mu2_test]
    imgDisplay = 6
    if plotPredictions:
        plottingPrediction(data, model, imgDisplay)

