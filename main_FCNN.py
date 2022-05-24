from common.training_data import Data
from common.fcnn import FCNN
from common.model import Model
from common.postprocessing import plottingPrediction


if __name__ == '__main__':
    dataset = 'beam_homog'
    multCoef = 1
    plotPredictions = True

    
    mu1_1_125_mu2_all = ['Fh0_Fv1_R0.txt', 'Fh0.382683_Fv0.92388_R0.txt', 'Fh0.707107_Fv0.707107_R0.txt', 'Fh0.92388_Fv0.382683_R0.txt', 'Fh1_Fv0_R0.txt', 'Fh0.92388_Fv-0.382683_R0.txt', 'Fh0.707107_Fv-0.707107_R0.txt', 'Fh0.382683_Fv-0.92388_R0.txt', 'Fh0_Fv-1_R0.txt', 'Fh0_Fv1_R0.05.txt', 'Fh0.382683_Fv0.92388_R0.05.txt', 'Fh0.707107_Fv0.707107_R0.05.txt', 'Fh0.92388_Fv0.382683_R0.05.txt', 'Fh1_Fv0_R0.05.txt', 'Fh0.92388_Fv-0.382683_R0.05.txt', 'Fh0.707107_Fv-0.707107_R0.05.txt', 'Fh0.382683_Fv-0.92388_R0.05.txt', 'Fh0_Fv-1_R0.05.txt', 'Fh0_Fv1_R0.1.txt', 'Fh0.382683_Fv0.92388_R0.1.txt', 'Fh0.707107_Fv0.707107_R0.1.txt', 'Fh0.92388_Fv0.382683_R0.1.txt', 'Fh1_Fv0_R0.1.txt', 'Fh0.92388_Fv-0.382683_R0.1.txt', 'Fh0.707107_Fv-0.707107_R0.1.txt', 'Fh0.382683_Fv-0.92388_R0.1.txt', 'Fh0_Fv-1_R0.1.txt', 'Fh0_Fv1_R0.15.txt', 'Fh0.382683_Fv0.92388_R0.15.txt', 'Fh0.707107_Fv0.707107_R0.15.txt', 'Fh0.92388_Fv0.382683_R0.15.txt', 'Fh1_Fv0_R0.15.txt', 'Fh0.92388_Fv-0.382683_R0.15.txt', 'Fh0.707107_Fv-0.707107_R0.15.txt', 'Fh0.382683_Fv-0.92388_R0.15.txt', 'Fh0_Fv-1_R0.15.txt', 'Fh0_Fv1_R0.2.txt', 'Fh0.382683_Fv0.92388_R0.2.txt', 'Fh0.707107_Fv0.707107_R0.2.txt', 'Fh0.92388_Fv0.382683_R0.2.txt', 'Fh1_Fv0_R0.2.txt', 'Fh0.92388_Fv-0.382683_R0.2.txt', 'Fh0.707107_Fv-0.707107_R0.2.txt', 'Fh0.382683_Fv-0.92388_R0.2.txt', 'Fh0_Fv-1_R0.2.txt', 'Fh0_Fv1_R0.25.txt', 'Fh0.382683_Fv0.92388_R0.25.txt', 'Fh0.707107_Fv0.707107_R0.25.txt', 'Fh0.92388_Fv0.382683_R0.25.txt', 'Fh1_Fv0_R0.25.txt', 'Fh0.92388_Fv-0.382683_R0.25.txt', 'Fh0.707107_Fv-0.707107_R0.25.txt', 'Fh0.382683_Fv-0.92388_R0.25.txt', 'Fh0_Fv-1_R0.25.txt']
    mu1_1_125_mu2_5angles = ['Fh0.707107_Fv0.707107_R0.txt', 'Fh0.92388_Fv0.382683_R0.txt', 'Fh1_Fv0_R0.txt', 'Fh0.92388_Fv-0.382683_R0.txt', 'Fh0.707107_Fv-0.707107_R0.txt', 'Fh0.707107_Fv0.707107_R0.05.txt', 'Fh0.92388_Fv0.382683_R0.05.txt', 'Fh1_Fv0_R0.05.txt', 'Fh0.92388_Fv-0.382683_R0.05.txt', 'Fh0.707107_Fv-0.707107_R0.05.txt', 'Fh0.707107_Fv0.707107_R0.1.txt', 'Fh0.92388_Fv0.382683_R0.1.txt', 'Fh1_Fv0_R0.1.txt', 'Fh0.92388_Fv-0.382683_R0.1.txt', 'Fh0.707107_Fv-0.707107_R0.1.txt', 'Fh0.707107_Fv0.707107_R0.15.txt', 'Fh0.92388_Fv0.382683_R0.15.txt', 'Fh1_Fv0_R0.15.txt', 'Fh0.92388_Fv-0.382683_R0.15.txt', 'Fh0.707107_Fv-0.707107_R0.15.txt', 'Fh0.707107_Fv0.707107_R0.2.txt', 'Fh0.92388_Fv0.382683_R0.2.txt', 'Fh1_Fv0_R0.2.txt', 'Fh0.92388_Fv-0.382683_R0.2.txt', 'Fh0.707107_Fv-0.707107_R0.2.txt', 'Fh0.707107_Fv0.707107_R0.25.txt', 'Fh0.92388_Fv0.382683_R0.25.txt', 'Fh1_Fv0_R0.25.txt', 'Fh0.92388_Fv-0.382683_R0.25.txt', 'Fh0.707107_Fv-0.707107_R0.25.txt']
    mu1_25_27_mu2_all = ['Fh0_Fv1_T0.5.txt', 'Fh0.382683_Fv0.92388_T0.5.txt', 'Fh0.707107_Fv0.707107_T0.5.txt', 'Fh0.92388_Fv0.382683_T0.5.txt', 'Fh1_Fv0_T0.5.txt', 'Fh0.92388_Fv-0.382683_T0.5.txt', 'Fh0.707107_Fv-0.707107_T0.5.txt', 'Fh0.382683_Fv-0.92388_T0.5.txt', 'Fh0_Fv-1_T0.5.txt', 'Fh0_Fv1_T0.45.txt', 'Fh0.382683_Fv0.92388_T0.45.txt', 'Fh0.707107_Fv0.707107_T0.45.txt', 'Fh0.92388_Fv0.382683_T0.45.txt', 'Fh1_Fv0_T0.45.txt', 'Fh0.92388_Fv-0.382683_T0.45.txt', 'Fh0.707107_Fv-0.707107_T0.45.txt', 'Fh0.382683_Fv-0.92388_T0.45.txt', 'Fh0_Fv-1_T0.45.txt', 'Fh0_Fv1_T0.4.txt', 'Fh0.382683_Fv0.92388_T0.4.txt', 'Fh0.707107_Fv0.707107_T0.4.txt', 'Fh0.92388_Fv0.382683_T0.4.txt', 'Fh1_Fv0_T0.4.txt', 'Fh0.92388_Fv-0.382683_T0.4.txt', 'Fh0.707107_Fv-0.707107_T0.4.txt', 'Fh0.382683_Fv-0.92388_T0.4.txt', 'Fh0_Fv-1_T0.4.txt', 'Fh0_Fv1_T0.35.txt', 'Fh0.382683_Fv0.92388_T0.35.txt', 'Fh0.707107_Fv0.707107_T0.35.txt', 'Fh0.92388_Fv0.382683_T0.35.txt', 'Fh1_Fv0_T0.35.txt', 'Fh0.92388_Fv-0.382683_T0.35.txt', 'Fh0.707107_Fv-0.707107_T0.35.txt', 'Fh0.382683_Fv-0.92388_T0.35.txt', 'Fh0_Fv-1_T0.35.txt', 'Fh0_Fv1_T0.3.txt', 'Fh0.382683_Fv0.92388_T0.3.txt', 'Fh0.707107_Fv0.707107_T0.3.txt', 'Fh0.92388_Fv0.382683_T0.3.txt', 'Fh1_Fv0_T0.3.txt', 'Fh0.92388_Fv-0.382683_T0.3.txt', 'Fh0.707107_Fv-0.707107_T0.3.txt', 'Fh0.382683_Fv-0.92388_T0.3.txt', 'Fh0_Fv-1_T0.3.txt']
    # mu1_21_235_mu2_all = ['Fh0_Fv1_T0.9.txt', 'Fh0.382683_Fv0.92388_T0.9.txt', 'Fh0.707107_Fv0.707107_T0.9.txt', 'Fh0.92388_Fv0.382683_T0.9.txt', 'Fh1_Fv0_T0.9.txt', 'Fh0.92388_Fv-0.382683_T0.9.txt', 'Fh0.707107_Fv-0.707107_T0.9.txt', 'Fh0.382683_Fv-0.92388_T0.9.txt', 'Fh0_Fv-1_T0.9.txt', 'Fh0_Fv1_T0.85.txt', 'Fh0.382683_Fv0.92388_T0.85.txt', 'Fh0.707107_Fv0.707107_T0.85.txt', 'Fh0.92388_Fv0.382683_T0.85.txt', 'Fh1_Fv0_T0.85.txt', 'Fh0.92388_Fv-0.382683_T0.85.txt', 'Fh0.707107_Fv-0.707107_T0.85.txt', 'Fh0.382683_Fv-0.92388_T0.85.txt', 'Fh0_Fv-1_T0.85.txt', 'Fh0_Fv1_T0.8.txt', 'Fh0.382683_Fv0.92388_T0.8.txt', 'Fh0.707107_Fv0.707107_T0.8.txt', 'Fh0.92388_Fv0.382683_T0.8.txt', 'Fh1_Fv0_T0.8.txt', 'Fh0.92388_Fv-0.382683_T0.8.txt', 'Fh0.707107_Fv-0.707107_T0.8.txt', 'Fh0.382683_Fv-0.92388_T0.8.txt', 'Fh0_Fv-1_T0.8.txt', 'Fh0_Fv1_T0.75.txt', 'Fh0.382683_Fv0.92388_T0.75.txt', 'Fh0.707107_Fv0.707107_T0.75.txt', 'Fh0.92388_Fv0.382683_T0.75.txt', 'Fh1_Fv0_T0.75.txt', 'Fh0.92388_Fv-0.382683_T0.75.txt', 'Fh0.707107_Fv-0.707107_T0.75.txt', 'Fh0.382683_Fv-0.92388_T0.75.txt', 'Fh0_Fv-1_T0.75.txt', 'Fh0_Fv1_T0.7.txt', 'Fh0.382683_Fv0.92388_T0.7.txt', 'Fh0.707107_Fv0.707107_T0.7.txt', 'Fh0.92388_Fv0.382683_T0.7.txt', 'Fh1_Fv0_T0.7.txt', 'Fh0.92388_Fv-0.382683_T0.7.txt', 'Fh0.707107_Fv-0.707107_T0.7.txt', 'Fh0.382683_Fv-0.92388_T0.7.txt', 'Fh0_Fv-1_T0.7.txt', 'Fh0_Fv1_T0.65.txt', 'Fh0.382683_Fv0.92388_T0.65.txt', 'Fh0.707107_Fv0.707107_T0.65.txt', 'Fh0.92388_Fv0.382683_T0.65.txt', 'Fh1_Fv0_T0.65.txt', 'Fh0.92388_Fv-0.382683_T0.65.txt', 'Fh0.707107_Fv-0.707107_T0.65.txt', 'Fh0.382683_Fv-0.92388_T0.65.txt', 'Fh0_Fv-1_T0.65.txt']
    data = Data(dataset, multCoef, testData=0.1, verbose=True, saveInfo=True)
    data.load()
    # data.thresholdFilter(tol=0)
    data.rehsapeDataToArray()

    fcnn = FCNN(data,verbose=True, saveInfo=True)
    fcnn.build(codeSize=25, nNeurons=200, nHidLayers=4, regularisation=1e-4)

    model = Model(fcnn)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(epochs=500, nBatch=16, earlyStopPatience=50, earlyStopTol=1e-4)
    model.predict()

    nDisplay = 5
    if plotPredictions:
        plottingPrediction(data, model, nDisplay)

