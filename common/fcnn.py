from tensorflow import keras
from tensorflow.keras import layers
from common.postprocessing import summaryInfo
import numpy as np
import timeit
import tensorflow.keras.backend as K


class FCNN():
    def __init__(self, data, verbose=False, saveInfo=False):
        self.autoencoder = None
        self.encoder = None
        self.buildTime = None
        self.nTrainParam = None
        self.nNonTrainParam = None
        self.codeSize = None

        assert isinstance(data, object), '"data" must be an object'
        assert isinstance(verbose, bool), '"verbose" must be a string'
        assert isinstance(saveInfo, bool), '"saveInfo" must be a boolean'

        self.data = data
        self.verbose = verbose
        self.saveInfo = saveInfo

    def build(self, codeSize=25, nNeurons=40, nHidLayers=2, regularisation=0):
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

        assert isinstance(codeSize, int), '"codeSize" must be an integer'
        assert isinstance(nNeurons, int), '"nNeurons" must be an integer'
        assert isinstance(nHidLayers, int), '"codenHidLayersSize" must be an integer'
        assert isinstance(regularisation, (int,float)), '"regularisation" must be either an int or a float'

        self.codeSize = codeSize
        self.nNeurons = nNeurons
        self.nHidLayers = nHidLayers
        self.regularisation = regularisation

        start = timeit.default_timer()
        input_img = keras.Input(shape=(self.data.dimension,))
        
        # Encoder
        encoded = layers.Dense(nNeurons, activation='relu', kernel_initializer='he_normal')(input_img)
        for _ in range(nHidLayers-1):
            encoded = layers.Dense(nNeurons, activation='relu', kernel_initializer='he_normal')(encoded)
        
        # Code
        encoded = layers.Dense(self.codeSize, activation='relu', kernel_initializer='he_normal', 
        kernel_regularizer=keras.regularizers.l1(self.regularisation))(encoded)
        
        # Decoder
        decoded = layers.Dense(nNeurons, activation='relu', kernel_initializer='he_normal')(encoded)
        for _ in range(nHidLayers-1):
            decoded = layers.Dense(nNeurons, activation='relu', kernel_initializer='he_normal')(decoded)
        decoded = layers.Dense(self.data.dimension, activation='relu')(decoded)

        # Create autoencoder and encoder objects
        self.autoencoder = keras.Model(input_img, decoded)
        self.encoder = keras.Model(input_img, encoded)

        self.nTrainParam = int(np.sum([K.count_params(w) for w in self.autoencoder.trainable_weights]))
        self.nNonTrainParam = int(np.sum([K.count_params(w) for w in self.autoencoder.non_trainable_weights]))

        stop = timeit.default_timer()
        self.buildTime = stop - start
        summary()