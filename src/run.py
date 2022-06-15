from src.training_data import Data
from src.model import Model
from src.postprocessing import plottingPrediction
from src.nn import NN

def inputCheck(hyperParam):
    paramsFCNN = ['DATASET', 'TEST_DATA', 'VERBOSE', 'SAVE_INFO', 'ARCH', 'CODE_SIZE', 'N_NEURONS', 'N_HID_LAY', 'REGULARISATION',
        'EPOCHS', 'N_BATCH', 'EARLY_STOP_PATIENCE', 'EARLY_STOP_TOL', 'INDIV_TEST_LOSS', 'PLOT_PRED', 'IMG_DISPLAY']
    paramsCNN = ['DATASET', 'TEST_DATA', 'VERBOSE', 'SAVE_INFO', 'ARCH', 'CODE_SIZE', 'N_NEURONS', 'N_HID_LAY', 'REGULARISATION', 'N_CONV_BLOCKS', 'N_FILTERS', 'KERNEL_SIZE', 'STRIDE',
    'EPOCHS', 'N_BATCH', 'EARLY_STOP_PATIENCE', 'EARLY_STOP_TOL', 'INDIV_TEST_LOSS', 'PLOT_PRED', 'IMG_DISPLAY']

    if hyperParam['ARCH'] == 'fcnn':
        paramRef = paramsFCNN
    elif hyperParam['ARCH'] == 'cnn':
        paramRef = paramsCNN

    for param in paramRef:
        if param not in hyperParam.keys():
            raise ValueError('{} not specified'.format(param))

def run(**kw):

    inputCheck(kw)

    # Load data
    data = Data(kw)
    data.load()

    # Create NN
    nn = NN().build(data, kw)

    # Build, compile and train model
    model = Model(nn)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(kw)
    model.predict(kw)
    
    if kw['PLOT_PRED']:
        plottingPrediction(data, model, kw)

    return model.test_loss