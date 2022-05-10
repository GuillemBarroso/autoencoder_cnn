from common.training_data import Data
from common.fcnn import FCNN
from common.model import Model
from common.postprocessing import plottingPrediction
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

    data = Data(dataset, verbose=True, saveInfo=True)
    data.load()
    data.thresholdFilter(tol=1e-6)
    data.rehsapeDataToArray()

    fcnn = FCNN(data,verbose=True, saveInfo=True)
    fcnn.build(codeSize=25, nNeurons=200, nHidLayers=4, regularisation=1e-4)

    model = Model(fcnn)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(epochs=500, nBatch=16, earlyStopPatience=50, earlyStopTol=1e-8)
    model.predict()

    nDisplay = 5
    if plotPredictions:
        plottingPrediction(data, model, nDisplay)

