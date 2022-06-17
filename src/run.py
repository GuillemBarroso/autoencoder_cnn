from src.training_data import Data
from src.model import ModelNN
from src.postprocessing import plottingPrediction
from src.nn import NN
import numpy as np


def inputCheck(hyperParam):
    paramsFCNN = ['DATASET', 'TEST_DATA', 'VERBOSE', 'SAVE_INFO', 'ARCH', 'CODE_SIZE', 'N_NEURONS', 'N_HID_LAY', 'REGULARISATION',
        'EPOCHS', 'N_BATCH', 'EARLY_STOP_PATIENCE', 'EARLY_STOP_TOL', 'INDIV_TEST_LOSS', 'PLOT_PRED', 'IMG_DISPLAY']
    paramsCNN = ['DATASET', 'TEST_DATA', 'VERBOSE', 'SAVE_INFO', 'ARCH', 'CODE_SIZE', 'N_NEURONS', 'N_HID_LAY', 'REGULARISATION', 'N_CONV_BLOCKS', 'N_FILTERS', 'KERNEL_SIZE', 'STRIDE',
        'EPOCHS', 'N_BATCH', 'EARLY_STOP_PATIENCE', 'EARLY_STOP_TOL', 'INDIV_TEST_LOSS', 'PLOT_PRED', 'IMG_DISPLAY']
    paramsParamAE = ['DATASET', 'TEST_DATA', 'VERBOSE', 'SAVE_INFO', 'ARCH', 'CODE_SIZE', 'N_NEURONS', 'N_HID_LAY', 'N_NEURONS_PARAM', 'N_HID_LAY_PARAM',
        'REGULARISATION', 'EPOCHS', 'N_BATCH', 'EARLY_STOP_PATIENCE', 'EARLY_STOP_TOL', 'INDIV_TEST_LOSS', 'PLOT_PRED', 'IMG_DISPLAY']

    if hyperParam['ARCH'] == 'fcnn':
        paramRef = paramsFCNN
    elif hyperParam['ARCH'] == 'cnn':
        paramRef = paramsCNN
    elif hyperParam['ARCH'] == 'param_ae':
        paramRef = paramsParamAE
    else:
        raise ValueError('NN architecture not implemented.')

    for param in paramRef:
        if param not in hyperParam.keys():
            raise ValueError('{} not specified'.format(param))

def run(**kw):

    inputCheck(kw)

    # Load data
    data = Data(kw)
    data.load()

    # Create NN
    nn = NN(data, kw)

    # Build, compile and train model
    model = ModelNN(nn)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(kw)
    model.predict(kw)
    
    if kw['PLOT_PRED']:
        plottingPrediction(data, model, kw)

    return model.test_loss

# TODO: REFACTOR THIS AND ALL INFO REGARDING DATASET IN DATASET FILE (CREATE DATASET DIRECTORY FOR .py FILES)
def manualTestDataSelect():

    # Select test data by choosing mu1-mu2 combinations

    # ---------------------------------------------------------------
    # 4 columens evenly-spaced with all angles
    # mu1 = [0.6, 1.25, 1.7, 2.4] 
    # mu2 = list(np.arange(0, 180, 3))
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # 4 pairs of columens (8 total) evenly spaced with all angles
    # mu1 = [0.55, 0.6, 1.2, 1.25, 1.7, 1.75, 2.4, 2.45] 
    # mu2 = list(np.arange(0, 180, 3))
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # 4 triplets of columens (12 total) evenly spaced with all angles
    # mu1 = [0.5, 0.55, 0.6, 1.15, 1.2, 1.25, 1.7, 1.75, 1.8, 2.4, 2.45, 2.5] 
    # mu2 = list(np.arange(0, 180, 3))
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # 3 rows evenly spaced with all angles
    # mu1 = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
    #        1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9,
    #        2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7]
    # mu2 = [45, 90, 135]
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # 3 pairs of rows (6 in total) evenly spaced with all angles
    # mu1 = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
    #        1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9,
    #        2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7]
    # mu2 = [42, 45, 90, 93, 132, 135]
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # 3 triplets of rows (12 in total) evenly spaced with all angles
    mu1 = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
           1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9,
           2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7]
    mu2 = [39, 42, 45, 87, 90, 93, 129, 132, 135]
    # ---------------------------------------------------------------


    # Select which of the test data above is plotted after training
    # mu1_plot = [0.6, 1.25, 1.25, 1.7, 2.4, 2.4] 
    # mu2_plot = [0, 90, 9, 177, 90, 162]
    mu1_plot = [0.6, 1.25, 1.25, 1.7, 2.4, 2.4] 
    mu2_plot = [45, 90, 45, 90, 45, 135]

    return [[mu1, mu2], [mu1_plot, mu2_plot]]