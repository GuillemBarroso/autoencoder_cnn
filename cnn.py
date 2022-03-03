from tensorflow import keras
from tensorflow.keras import layers
import timeit
from prettytable import PrettyTable
from postprocessing import plotTraining, summaryInfo
import tensorflow.keras.backend as K
import numpy as np


class CNN():
    def __init__(self, data, verbose=False, saveInfo=False):
        self.buildTime = None
        self.compileTime = None
        self.trainTime = None
        self.autoencoder = None
        self.encoder = None
        self.predictions = None
        self.nConvBlocks = None
        self.nFilters = None
        self.stride = None
        self.kernelSize = None
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

        assert isinstance(data, object), '"data" must be an object'
        assert isinstance(verbose, bool), '"verbose" must be a string'
        assert isinstance(saveInfo, bool), '"saveInfo" must be a boolean'

        self.data = data
        self.verbose = verbose
        self.saveInfo = saveInfo

    def build(self, nConvBlocks=1, codeSize=36, nFilters=10, kernelSize=3, stride=2):
        def summary():
            data = [['nConvBlocks', self.nConvBlocks],
            ['nFilters', self.nFilters],
            ['kernelSize', self.kernelSize],
            ['stride size', self.stride],
            ['code size', self.codeSize],
            ['num trainable param', self.nTrainParam],
            ['num non trainable param', self.nNonTrainParam],
            ['build time', '{:.2}s'.format(self.buildTime)]]
            name = 'results/buildModel_{}.png'.format(self.data.dataset)
            summaryInfo(data, self.verbose, self.saveInfo, name)

        self.nConvBlocks = nConvBlocks
        self.nFilters = nFilters
        self.kernelSize = kernelSize
        self.stride = stride
        self.codeSize = codeSize

        def inputCheck():
            assert isinstance(self.nConvBlocks, int), '"nConvBlock" must be an integer'
            assert isinstance(self.nFilters, (list,int)), '"nFilters" must be a a list or an integer'
            assert isinstance(self.kernelSize, (list,int,tuple)), '"kernel" must be an integer or a tuple'
            assert isinstance(self.stride, (int,tuple)), '"stride" must be an integer or a tuple'
            if isinstance(self.nFilters, list):
                assert len(self.nFilters) == self.nConvBlocks, '"nFilters" list length must match "nConvBlocks"'
            else:
                self.nFilters = [self.nFilters]
            if isinstance(self.kernelSize, list):
                assert len(self.kernelSize) == self.nConvBlocks, '"kernelSize" list length must match "nConvBlocks"'
            else:
                self.kernelSize = [self.kernelSize]

            self.red_res = (self.data.resolution[0]/self.stride**self.nConvBlocks,
                       self.data.resolution[1]/self.stride**self.nConvBlocks)
            if not self.red_res[0].is_integer() or not self.red_res[1].is_integer():
                raise ValueError('Conflict between image resolution, number of convolutional blocks and stride:\n'
                                 'Original image is {}x{} and encoder would reduce it to {}x{}'.format(
                    self.data.resolution[0], self.data.resolution[1], self.red_res[0], self.red_res[1]))
            self.red_res = (int(self.red_res[0]), int(self.red_res[1]))

        start = timeit.default_timer()
        inputCheck()

        # Encoder
        input = layers.Input(shape=self.data.resolution)
        encoded = self.Encoder(self, input)
        encoded = encoded.build()

        # Fully-connected layers to compress information
        encoded = layers.Flatten()(encoded)
        nNeuronsDense = self.red_res[0]*self.red_res[1]*self.nFilters[-1]
        # encoded = layers.Dense(nNeuronsDense*0.1, activation='relu')(encoded)
        # encoded = layers.Dense(nNeuronsDense*0.01, activation='relu')(encoded)
        encoded = layers.Dense(self.codeSize, activation='relu')(encoded)

        # decoded = layers.Dense(nNeuronsDense*0.1*0.1, activation='relu')(encoded)
        # decoded = layers.Dense(nNeuronsDense*0.1, activation='relu')(decoded)
        decoded = layers.Dense(nNeuronsDense, activation='relu')(encoded)
        decoded = layers.Reshape((self.red_res[0], self.red_res[1], self.nFilters[-1]))(decoded)

        # Decoder
        decoded = self.Decoder(self, decoded)
        decoded = decoded.build()
        decoded = layers.Conv2D(self.data.resolution[2], self.kernelSize[0], activation="sigmoid", padding="same")(decoded)

        # Build autoencoder and encoder (so "code" or "latent vector" is accessible)
        self.autoencoder = keras.Model(input, decoded)
        self.encoder = keras.Model(input, encoded)
        del encoded; del decoded

        self.nTrainParam = int(np.sum([K.count_params(w) for w in self.autoencoder.trainable_weights]))
        self.nNonTrainParam = int(np.sum([K.count_params(w) for w in self.autoencoder.non_trainable_weights]))

        stop = timeit.default_timer()
        self.buildTime = stop - start
        summary()

    class Encoder():
        def __init__(self, model, input):
            self.input = input
            self.model = model
        def build(self):
            for iBlock in range(self.model.nConvBlocks):
                encoded = layers.Conv2D(self.model.nFilters[iBlock], self.model.kernelSize[iBlock], self.model.stride, activation="relu", padding="same")(self.input)
                # encoded = layers.MaxPooling2D(self.model.stride, padding="same")(encoded)
                self.input = encoded
            return encoded

    class Decoder():
        def __init__(self, model, input):
            self.model = model
            self.input = input

        def build(self):
            reversedFilters = list(reversed(self.model.nFilters))
            revKernelSize = list(reversed(self.model.kernelSize))
            for iBlock in range(self.model.nConvBlocks):
                decoded = layers.Conv2DTranspose(reversedFilters[iBlock], revKernelSize[iBlock], strides=self.model.stride, activation="relu", padding="same")(self.input)
                self.input = decoded
            return decoded
