from tensorflow import keras
from tensorflow.keras import layers, Model
from src.postprocessing import summaryInfo
import numpy as np
import timeit
import tensorflow.keras.backend as K


class FCNN():
    def __init__(self, data, kw):
        self.autoencoder = None
        self.encoder = None
        self.buildTime = None
        self.nTrainParam = None
        self.nNonTrainParam = None
        self.codeSize = None

        assert isinstance(data, object), '"data" must be an object'
        assert isinstance(kw['VERBOSE'], bool), '"verbose" must be a string'
        assert isinstance(kw['SAVE_INFO'], bool), '"saveInfo" must be a boolean'

        self.data = data
        self.verbose = kw['VERBOSE']
        self.saveInfo = kw['SAVE_INFO']

    def build(self, kw):
        def summary():
            data = [['NN arch', 'Fully-connected'],
            ['nHidLayers', self.nHidLayers],
            ['nNeurons/hidLayer', self.nNeurons],
            ['code size', self.codeSize],
            ['regularisation coef', '{:.0e}'.format(self.regularisation)],
            ['num trainable param', self.nTrainParam],
            ['num non trainable param', self.nNonTrainParam],
            ['build time', '{:.2}s'.format(self.buildTime)]]
            name = 'results/buildModel_{}.png'.format(self.data.dataset)
            summaryInfo(data, self.verbose, self.saveInfo, name)

        assert isinstance(kw['CODE_SIZE'], int), '"codeSize" must be an integer'
        assert isinstance(kw['N_NEURONS'], int), '"nNeurons" must be an integer'
        assert isinstance(kw['N_HID_LAY'], int), '"codenHidLayersSize" must be an integer'
        assert isinstance(kw['REGULARISATION'], (int,float)), '"regularisation" must be either an int or a float'

        self.codeSize = kw['CODE_SIZE']
        self.nNeurons = kw['N_NEURONS']
        self.nHidLayers = kw['N_HID_LAY']
        self.regularisation = kw['REGULARISATION']

        start = timeit.default_timer()
        input_img = keras.Input(shape=(self.data.dimension,))
        hidReg = 0

        # Encoder
        encoded = layers.Dense(self.nNeurons, activation='relu', kernel_initializer='he_normal')(input_img)
        for _ in range(self.nHidLayers-1):
            encoded = layers.Dense(self.nNeurons, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1(hidReg))(encoded)
            
        # Code
        encoded = layers.Dense(self.codeSize, activation='relu', kernel_initializer='he_normal', 
        kernel_regularizer=keras.regularizers.l1(self.regularisation))(encoded)
        
        # Decoder
        decoded = layers.Dense(self.nNeurons, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1(hidReg))(encoded)
        for _ in range(self.nHidLayers-1):
            decoded = layers.Dense(self.nNeurons, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1(hidReg))(decoded)
        decoded = layers.Dense(self.data.dimension, activation='sigmoid')(decoded)

        # Create autoencoder and encoder objects
        self.autoencoder = Model(input_img, decoded)
        self.encoder = Model(input_img, encoded)

        self.nTrainParam = int(np.sum([K.count_params(w) for w in self.autoencoder.trainable_weights]))
        self.nNonTrainParam = int(np.sum([K.count_params(w) for w in self.autoencoder.non_trainable_weights]))

        stop = timeit.default_timer()
        self.buildTime = stop - start
        summary()