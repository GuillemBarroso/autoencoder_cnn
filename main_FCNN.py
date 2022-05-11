from common.training_data import Data
from common.fcnn import FCNN
from common.model import Model
from common.postprocessing import plottingPrediction
import matplotlib.pyplot as plt
# import pyvista as pv
# import numpy as np  
# import vtk


if __name__ == '__main__':
    dataset = 'beam_homog_x4'
    plotPredictions = True

    # mesh = pv.read('Fh0_Fv1_R0.vtk')
    
    # reader = vtk.vtkDataSetReader()
    # reader.SetFileName("lh.sulc.fundi.from.pits.pial.vtk")
    # reader.ReadAllScalarsOn()  # Activate the reading of all scalars
    # reader.Update()

    # data=reader.GetOutput()

    testData = ['Fh0_Fv1_R1.0.txt', 'Fh0_Fv1_R1.0 copy.txt', 'Fh0_Fv1_R1.0 copy 2.txt', 'Fh0_Fv1_R1.0 copy 3.txt', 'Fh0.382683_Fv0.92388_R1.0.txt', 'Fh0.382683_Fv0.92388_R1.0 copy.txt', 'Fh0.382683_Fv0.92388_R1.0 copy 2.txt', 'Fh0.382683_Fv0.92388_R1.0 copy 3.txt', 'Fh0.707107_Fv0.707107_R1.0.txt', 'Fh0.707107_Fv0.707107_R1.0 copy.txt', 'Fh0.707107_Fv0.707107_R1.0 copy 2.txt', 'Fh0.707107_Fv0.707107_R1.0 copy 3.txt', 'Fh0.92388_Fv0.382683_R1.0.txt', 'Fh0.92388_Fv0.382683_R1.0 copy.txt', 'Fh0.92388_Fv0.382683_R1.0 copy 2.txt', 'Fh0.92388_Fv0.382683_R1.0 copy 3.txt', 'Fh1_Fv0_R1.0.txt', 'Fh1_Fv0_R1.0 copy.txt', 'Fh1_Fv0_R1.0 copy 2.txt', 'Fh1_Fv0_R1.0 copy 3.txt', 'Fh0.92388_Fv-0.382683_R1.0.txt', 'Fh0.92388_Fv-0.382683_R1.0 copy.txt', 'Fh0.92388_Fv-0.382683_R1.0 copy 2.txt', 'Fh0.92388_Fv-0.382683_R1.0 copy 3.txt', 'Fh0.707107_Fv-0.707107_R1.0.txt', 'Fh0.707107_Fv-0.707107_R1.0 copy.txt', 'Fh0.707107_Fv-0.707107_R1.0 copy 2.txt', 'Fh0.707107_Fv-0.707107_R1.0 copy 3.txt', 'Fh0.382683_Fv-0.92388_R1.0.txt', 'Fh0.382683_Fv-0.92388_R1.0 copy.txt', 'Fh0.382683_Fv-0.92388_R1.0 copy 2.txt', 'Fh0.382683_Fv-0.92388_R1.0 copy 3.txt', 'Fh0_Fv-1_R1.0.txt', 'Fh0_Fv-1_R1.0 copy.txt', 'Fh0_Fv-1_R1.0 copy 2.txt', 'Fh0_Fv-1_R1.0 copy 3.txt', 'Fh0_Fv1_R0.95.txt', 'Fh0_Fv1_R0.95 copy.txt', 'Fh0_Fv1_R0.95 copy 2.txt', 'Fh0_Fv1_R0.95 copy 3.txt', 'Fh0.382683_Fv0.92388_R0.95.txt', 'Fh0.382683_Fv0.92388_R0.95 copy.txt', 'Fh0.382683_Fv0.92388_R0.95 copy 2.txt', 'Fh0.382683_Fv0.92388_R0.95 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.95.txt', 'Fh0.707107_Fv0.707107_R0.95 copy.txt', 'Fh0.707107_Fv0.707107_R0.95 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.95 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.95.txt', 'Fh0.92388_Fv0.382683_R0.95 copy.txt', 'Fh0.92388_Fv0.382683_R0.95 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.95 copy 3.txt', 'Fh1_Fv0_R0.95.txt', 'Fh1_Fv0_R0.95 copy.txt', 'Fh1_Fv0_R0.95 copy 2.txt', 'Fh1_Fv0_R0.95 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.95.txt', 'Fh0.92388_Fv-0.382683_R0.95 copy.txt', 'Fh0.92388_Fv-0.382683_R0.95 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.95 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.95.txt', 'Fh0.707107_Fv-0.707107_R0.95 copy.txt', 'Fh0.707107_Fv-0.707107_R0.95 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.95 copy 3.txt', 'Fh0.382683_Fv-0.92388_R0.95.txt', 'Fh0.382683_Fv-0.92388_R0.95 copy.txt', 'Fh0.382683_Fv-0.92388_R0.95 copy 2.txt', 'Fh0.382683_Fv-0.92388_R0.95 copy 3.txt', 'Fh0_Fv-1_R0.95.txt', 'Fh0_Fv-1_R0.95 copy.txt', 'Fh0_Fv-1_R0.95 copy 2.txt', 'Fh0_Fv-1_R0.95 copy 3.txt', 'Fh0_Fv1_R0.9.txt', 'Fh0_Fv1_R0.9 copy.txt', 'Fh0_Fv1_R0.9 copy 2.txt', 'Fh0_Fv1_R0.9 copy 3.txt', 'Fh0.382683_Fv0.92388_R0.9.txt', 'Fh0.382683_Fv0.92388_R0.9 copy.txt', 'Fh0.382683_Fv0.92388_R0.9 copy 2.txt', 'Fh0.382683_Fv0.92388_R0.9 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.9.txt', 'Fh0.707107_Fv0.707107_R0.9 copy.txt', 'Fh0.707107_Fv0.707107_R0.9 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.9 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.9.txt', 'Fh0.92388_Fv0.382683_R0.9 copy.txt', 'Fh0.92388_Fv0.382683_R0.9 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.9 copy 3.txt', 'Fh1_Fv0_R0.9.txt', 'Fh1_Fv0_R0.9 copy.txt', 'Fh1_Fv0_R0.9 copy 2.txt', 'Fh1_Fv0_R0.9 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.9.txt', 'Fh0.92388_Fv-0.382683_R0.9 copy.txt', 'Fh0.92388_Fv-0.382683_R0.9 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.9 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.9.txt', 'Fh0.707107_Fv-0.707107_R0.9 copy.txt', 'Fh0.707107_Fv-0.707107_R0.9 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.9 copy 3.txt', 'Fh0.382683_Fv-0.92388_R0.9.txt', 'Fh0.382683_Fv-0.92388_R0.9 copy.txt', 'Fh0.382683_Fv-0.92388_R0.9 copy 2.txt', 'Fh0.382683_Fv-0.92388_R0.9 copy 3.txt', 'Fh0_Fv-1_R0.9.txt', 'Fh0_Fv-1_R0.9 copy.txt', 'Fh0_Fv-1_R0.9 copy 2.txt', 'Fh0_Fv-1_R0.9 copy 3.txt', 'Fh0_Fv1_R0.85.txt', 'Fh0_Fv1_R0.85 copy.txt', 'Fh0_Fv1_R0.85 copy 2.txt', 'Fh0_Fv1_R0.85 copy 3.txt', 'Fh0.382683_Fv0.92388_R0.85.txt', 'Fh0.382683_Fv0.92388_R0.85 copy.txt', 'Fh0.382683_Fv0.92388_R0.85 copy 2.txt', 'Fh0.382683_Fv0.92388_R0.85 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.85.txt', 'Fh0.707107_Fv0.707107_R0.85 copy.txt', 'Fh0.707107_Fv0.707107_R0.85 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.85 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.85.txt', 'Fh0.92388_Fv0.382683_R0.85 copy.txt', 'Fh0.92388_Fv0.382683_R0.85 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.85 copy 3.txt', 'Fh1_Fv0_R0.85.txt', 'Fh1_Fv0_R0.85 copy.txt', 'Fh1_Fv0_R0.85 copy 2.txt', 'Fh1_Fv0_R0.85 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.85.txt', 'Fh0.92388_Fv-0.382683_R0.85 copy.txt', 'Fh0.92388_Fv-0.382683_R0.85 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.85 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.85.txt', 'Fh0.707107_Fv-0.707107_R0.85 copy.txt', 'Fh0.707107_Fv-0.707107_R0.85 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.85 copy 3.txt', 'Fh0.382683_Fv-0.92388_R0.85.txt', 'Fh0.382683_Fv-0.92388_R0.85 copy.txt', 'Fh0.382683_Fv-0.92388_R0.85 copy 2.txt', 'Fh0.382683_Fv-0.92388_R0.85 copy 3.txt', 'Fh0_Fv-1_R0.85.txt', 'Fh0_Fv-1_R0.85 copy.txt', 'Fh0_Fv-1_R0.85 copy 2.txt', 'Fh0_Fv-1_R0.85 copy 3.txt', 'Fh0_Fv1_R0.8.txt', 'Fh0_Fv1_R0.8 copy.txt', 'Fh0_Fv1_R0.8 copy 2.txt', 'Fh0_Fv1_R0.8 copy 3.txt', 'Fh0.382683_Fv0.92388_R0.8.txt', 'Fh0.382683_Fv0.92388_R0.8 copy.txt', 'Fh0.382683_Fv0.92388_R0.8 copy 2.txt', 'Fh0.382683_Fv0.92388_R0.8 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.8.txt', 'Fh0.707107_Fv0.707107_R0.8 copy.txt', 'Fh0.707107_Fv0.707107_R0.8 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.8 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.8.txt', 'Fh0.92388_Fv0.382683_R0.8 copy.txt', 'Fh0.92388_Fv0.382683_R0.8 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.8 copy 3.txt', 'Fh1_Fv0_R0.8.txt', 'Fh1_Fv0_R0.8 copy.txt', 'Fh1_Fv0_R0.8 copy 2.txt', 'Fh1_Fv0_R0.8 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.8.txt', 'Fh0.92388_Fv-0.382683_R0.8 copy.txt', 'Fh0.92388_Fv-0.382683_R0.8 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.8 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.8.txt', 'Fh0.707107_Fv-0.707107_R0.8 copy.txt', 'Fh0.707107_Fv-0.707107_R0.8 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.8 copy 3.txt', 'Fh0.382683_Fv-0.92388_R0.8.txt', 'Fh0.382683_Fv-0.92388_R0.8 copy.txt', 'Fh0.382683_Fv-0.92388_R0.8 copy 2.txt', 'Fh0.382683_Fv-0.92388_R0.8 copy 3.txt', 'Fh0_Fv-1_R0.8.txt', 'Fh0_Fv-1_R0.8 copy.txt', 'Fh0_Fv-1_R0.8 copy 2.txt', 'Fh0_Fv-1_R0.8 copy 3.txt']
    data = Data(dataset, testData=testData, verbose=True, saveInfo=True)
    data.load()
    data.thresholdFilter(tol=1e-6)
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

