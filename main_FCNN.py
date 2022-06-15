from src.run import run


if __name__ == '__main__':
    
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
    # Results visualisation
    mu1_test = [1.4, 1.4, 1.45, 1.5, 1.55, 1.55]
    mu2_test = [67.5, 90, 67.5, 112.5, 90, 112.5]

    # Parameters
    VERBOSE = True
    SAVE_INFO = True
    PLOT_PRED = True
    INDIV_TEST_LOSS = True

    DATASET = 'beam_homog_big'
    TEST_DATA = 0.1
    # TEST_DATA = [mu1, mu2]

    ARCH = 'fcnn'
    CODE_SIZE = 25
    N_NEURONS = 200
    N_HID_LAY = 4
    REGULARISATION = 1e-4

    EPOCHS = 500
    N_BATCH = 12
    EARLY_STOP_PATIENCE = 100
    EARLY_STOP_TOL = 1e-4
    
    IMG_DISPLAY = 6
    # IMG_DISPLAY = [mu1_test, mu2_test]

    hyperParams = {'DATASET': DATASET, 'TEST_DATA': TEST_DATA, 'VERBOSE': VERBOSE,
        'SAVE_INFO':SAVE_INFO , 'ARCH': ARCH, 'CODE_SIZE': CODE_SIZE, 'N_NEURONS': N_NEURONS,
        'N_HID_LAY': N_HID_LAY, 'REGULARISATION': REGULARISATION, 'EPOCHS': EPOCHS,
        'N_BATCH': N_BATCH, 'EARLY_STOP_PATIENCE': EARLY_STOP_PATIENCE, 'EARLY_STOP_TOL': EARLY_STOP_TOL,
        'INDIV_TEST_LOSS': INDIV_TEST_LOSS, 'PLOT_PRED': PLOT_PRED, 'IMG_DISPLAY': IMG_DISPLAY
    }

    loss = run(**hyperParams)

