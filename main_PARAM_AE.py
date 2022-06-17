from src.run import run, manualTestDataSelect


if __name__ == '__main__':
    # Parameters
    VERBOSE = True
    SAVE_INFO = True
    PLOT_PRED = True
    INDIV_TEST_LOSS = True

    DATASET = 'beam_homog_big'
    # TEST_DATA = 0.1
    TEST_DATA = manualTestDataSelect()[0]

    ARCH = 'param_ae'
    CODE_SIZE = 25
    N_NEURONS = 200
    N_HID_LAY = 4
    N_NEURONS_PARAM = 25
    N_HID_LAY_PARAM = 1
    REGULARISATION = 1e-4

    EPOCHS = 300
    N_BATCH = 12
    EARLY_STOP_PATIENCE = 100
    EARLY_STOP_TOL = 1e-4
    
    # IMG_DISPLAY = 6
    IMG_DISPLAY = manualTestDataSelect()[1]

    hyperParams = {'DATASET': DATASET, 'TEST_DATA': TEST_DATA, 'VERBOSE': VERBOSE,
        'SAVE_INFO':SAVE_INFO , 'ARCH': ARCH, 'CODE_SIZE': CODE_SIZE, 'N_NEURONS': N_NEURONS,
        'N_HID_LAY': N_HID_LAY, 'N_NEURONS_PARAM': N_NEURONS_PARAM,
        'N_HID_LAY_PARAM': N_HID_LAY_PARAM, 'REGULARISATION': REGULARISATION, 'EPOCHS': EPOCHS,
        'N_BATCH': N_BATCH, 'EARLY_STOP_PATIENCE': EARLY_STOP_PATIENCE, 'EARLY_STOP_TOL': EARLY_STOP_TOL,
        'INDIV_TEST_LOSS': INDIV_TEST_LOSS, 'PLOT_PRED': PLOT_PRED, 'IMG_DISPLAY': IMG_DISPLAY
    }

    loss = run(**hyperParams)

