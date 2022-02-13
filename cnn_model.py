from tensorflow import keras
from tensorflow.keras import layers
import timeit
from prettytable import PrettyTable
from postprocessing import plotTraining
import tensorflow.keras.backend as K
import numpy as np


class Model():
    def __init__(self, data, verbose=False):
        self.data = data
        self.verbose = verbose
        self.buildTime = None
        self.compileTime = None
        self.trainTime = None
        self.autoencoder = None
        self.encoder = None
        self.predictions = None
        self.nConvBlock = None
        self.nFilters = None
        self.kernel = None
        self.stride = None
        self.optimizer = None
        self.loss = None
        self.epochs = None
        self.nBatch = None
        self.history = None
        self.min_loss = None
        self.min_valLoss = None
        self.test_loss = None
        self.red_res = None
        self.nTrainParam = None
        self.nNonTrainParam = None
        self.codeSize = None

    def build(self, nConvBlock=2, codeSize=36, nFilters=10, kernel=0.1, stride=2):
        self.nConvBlock = nConvBlock
        self.nFilters = nFilters
        self.kernel = kernel
        self.stride = stride
        self.codeSize = codeSize

        def inputCheck():
            assert isinstance(self.nConvBlock, int), '"nConvBlock" must be an integer'
            assert isinstance(self.nFilters, int), '"nFilters" must be an integer'
            assert isinstance(self.kernel, float), '"kernel" must be a float'
            assert 0 <= self.kernel <= 1, '"kernel" should be in [0,1]'
            assert isinstance(self.stride, (int,tuple)), '"stride" must be an integer or a tuple'
            self.kernel = (self.kernel, self.kernel)
            if isinstance(self.stride, int):
                self.stride = (self.stride, self.stride)
            self.red_res = (self.data.resolution[0]/self.stride[0]**self.nConvBlock,
                       self.data.resolution[1]/self.stride[1]**self.nConvBlock)
            if not self.red_res[0].is_integer() or not self.red_res[1].is_integer():
                raise ValueError('Conflict between image resolution, number of convolutional blocks and stride:\n'
                                 'Original image is {}x{} and encoder would reduce it to {}x{}'.format(
                    self.data.resolution[0], self.data.resolution[1], self.red_res[0], self.red_res[1]))
            self.red_res = (int(self.red_res[0]), int(self.red_res[1]))

        start = timeit.default_timer()
        inputCheck()
        kernelSize = (int(self.kernel[0]*self.data.resolution[0]), int(self.kernel[1]*self.data.resolution[1]))
        nFilters = [10, 10]
        kernelSize = [(3,3), (3,3)]

        # Encoder
        input = layers.Input(shape=self.data.resolution)
        encoded = self.Encoder(input, nConvBlock, nFilters, kernelSize, stride)
        encoded = encoded.build()

        # Fully-connected layers to compress information
        encoded = layers.Flatten()(encoded)
        nNeuronsDense = self.red_res[0]*self.red_res[1]*nFilters[-1]
        encoded = layers.Dense(nNeuronsDense*0.1, activation='relu')(encoded)
        encoded = layers.Dense(nNeuronsDense*0.01, activation='relu')(encoded)
        encoded = layers.Dense(self.codeSize, activation='relu')(encoded)

        decoded = layers.Dense(nNeuronsDense*0.1*0.1, activation='relu')(encoded)
        decoded = layers.Dense(nNeuronsDense*0.1, activation='relu')(decoded)
        decoded = layers.Dense(nNeuronsDense, activation='relu')(decoded)
        decoded = layers.Reshape((self.red_res[0], self.red_res[1], nFilters[-1]))(decoded)

        # Decoder
        decoded = self.Decoder(decoded, nConvBlock, list(reversed(nFilters)), list(reversed(kernelSize)), stride)
        decoded = decoded.build()
        decoded = layers.Conv2D(self.data.resolution[2], kernelSize[0], activation="sigmoid", padding="same")(decoded)

        # Build autoencoder and encoder (so "code" or "latent vector" is accessible)
        self.autoencoder = keras.Model(input, decoded)
        self.encoder = keras.Model(input, encoded)
        del encoded; del decoded

        self.nTrainParam = int(np.sum([K.count_params(w) for w in self.autoencoder.trainable_weights]))
        self.nNonTrainParam = int(np.sum([K.count_params(w) for w in self.autoencoder.non_trainable_weights]))

        stop = timeit.default_timer()
        self.buildTime = stop - start

    def compile(self,optimizer='adam', loss='mean_squared_error'):
        assert isinstance(optimizer, str), '"optimizer" must be a string'
        assert isinstance(loss, str), '"loss" must be a string'
        start = timeit.default_timer()
        self.optimizer = optimizer
        self.loss = loss
        self.autoencoder.compile(optimizer=self.optimizer, loss=self.loss)

        stop = timeit.default_timer()
        self.compileTime = stop - start
        if self.verbose:
            self.autoencoder.summary()

    def train(self, epochs=50, nBatch=32, earlyStopPatience=10, earlyStopTol=10e-4):
        ## TODO: add min_delta outside as a default param
        assert isinstance(epochs, int), '"epochs" must be an integer'
        assert isinstance(nBatch, int), '"nBatch" must be an integer'
        assert isinstance(earlyStopPatience, int), '"earlyStopPatience" must be an integer'
        start = timeit.default_timer()
        self.epochs = epochs
        self.nBatch = nBatch
        self.earlyStopPatience = earlyStopPatience
        earlyStop = keras.callbacks.EarlyStopping(patience=self.earlyStopPatience,monitor='val_loss',
                                                  restore_best_weights=True, min_delta=earlyStopTol)
        self.history = self.autoencoder.fit(self.data.x_train, self.data.x_train, epochs=self.epochs,
                                       batch_size=self.nBatch, shuffle=True,
                                       validation_data=(self.data.x_val, self.data.x_val), verbose=self.verbose,
                                       callbacks=[earlyStop])

        self.min_loss = min(self.history.history['loss'])
        self.min_valLoss = min(self.history.history['val_loss'])
        stop = timeit.default_timer()
        self.trainTime = stop - start
        if self.verbose:
            plotTraining(self.history, self.trainTime)

        ## TODO: save model and load saved models

    def predict(self):
        self.predictions = self.autoencoder.predict(self.data.x_test)
        self.test_loss = self.autoencoder.evaluate(self.data.x_test, self.data.x_test, verbose=self.verbose)
        self.code = self.encoder.predict(self.data.x_test)

    def summary(self):
        buildInfo = PrettyTable(['Parameter', 'Value'])
        buildInfo.title = 'Build model'
        buildInfo.add_row(['nConvBlock', self.nConvBlock])
        buildInfo.add_row(['nFilters', self.nFilters])
        buildInfo.add_row(['kernel size', self.kernel])
        buildInfo.add_row(['stride size', self.stride])
        buildInfo.add_row(['code size', (self.red_res[0], self.red_res[1], self.nFilters)])
        buildInfo.add_row(['num trainable param', self.nTrainParam])
        buildInfo.add_row(['num non trainable param', self.nNonTrainParam])
        buildInfo.add_row(['build time', '{:.2}s'.format(self.buildTime)])

        compileInfo = PrettyTable(['Parameter', 'Value'])
        compileInfo.title = 'Compile model'
        compileInfo.add_row(['optimizer', self.optimizer])
        compileInfo.add_row(['loss function', self.loss])
        compileInfo.add_row(['compile time', '{:.2}s'.format(self.compileTime)])

        trainInfo = PrettyTable(['Parameter', 'Value'])
        trainInfo.title = 'Training model'
        trainInfo.add_row(['epochs', self.epochs])
        trainInfo.add_row(['nBatch', self.nBatch])
        trainInfo.add_row(['early stop patience', '{} epochs'.format(self.earlyStopPatience)])
        trainInfo.add_row(['training time', '{:.2}s'.format(self.trainTime)])
        trainInfo.add_row(['min training loss', '{:.2}'.format(self.min_loss)])
        trainInfo.add_row(['min validation loss', '{:.2}'.format(self.min_valLoss)])
        trainInfo.add_row(['test loss evaluation', '{:.2}'.format(self.test_loss)])

        print(buildInfo)
        print(compileInfo)
        print(trainInfo)

    class Encoder():
        def __init__(self, input, nConvBlock, nFilters, kernelSize, stride):
            self.input = input
            self.nConvBlock = nConvBlock
            self.nFilters = nFilters
            self.kernelSize = kernelSize
            self.stride = stride

        def build(self):
            for iBlock in range(self.nConvBlock):
                encoded = layers.Conv2D(self.nFilters[iBlock], self.kernelSize[iBlock], activation="relu", padding="same",
                kernel_initializer='he_uniform', activity_regularizer=keras.regularizers.l1(10e-10))(self.input)
                encoded = layers.MaxPooling2D(self.stride, padding="same")(encoded)
                self.input = encoded
            return encoded

    class Decoder():
        def __init__(self, input, nConvBlock, nFilters, kernelSize, stride):
            self.input = input
            self.nConvBlock = nConvBlock
            self.nFilters = nFilters
            self.kernelSize = kernelSize
            self.stride = stride

        def build(self):
            for iBlock in range(self.nConvBlock):
                decoded = layers.Conv2DTranspose(self.nFilters[iBlock], self.kernelSize[iBlock], strides=self.stride, activation="relu", padding="same",
                kernel_initializer='he_uniform', activity_regularizer=keras.regularizers.l1(10e-10))(self.input)
                self.input = decoded
            return decoded
