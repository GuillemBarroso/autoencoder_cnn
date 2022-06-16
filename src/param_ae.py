from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, Model
from src.postprocessing import summaryInfo
import numpy as np
import timeit
import tensorflow.keras.backend as K


class PARAM_AE():
    def __init__(self, data, kw):
        self.autoencoder = None
        self.encoder = None
        self.buildTime = None
        self.nTrainParam = None
        self.nNonTrainParam = None
        self.codeSize = None
        self.paramAE = None
        self.x_nn = None
        self.code1 = None
        self.code2 = None
        self.x = None
        self.x_nnParam = None

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
        self.nNeuronsParam = kw['N_NEURONS_PARAM']
        self.nHidLayersParam = kw['N_HID_LAY_PARAM']
        self.regularisation = kw['REGULARISATION']

        start = timeit.default_timer()

        # Encodes input to low-dimensional code
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(200, activation=tf.nn.relu, name='encoder_hidden1', kernel_initializer='he_normal'),
                tf.keras.layers.Dense(200, activation=tf.nn.relu, name='encoder_hidden2', kernel_initializer='he_normal'),
                tf.keras.layers.Dense(200, activation=tf.nn.relu, name='encoder_hidden3', kernel_initializer='he_normal'),
                tf.keras.layers.Dense(200, activation=tf.nn.relu, name='encoder_hidden4', kernel_initializer='he_normal'),
                tf.keras.layers.Dense(200, activation=tf.nn.relu, name='encoder_hidden5', kernel_initializer='he_normal'),
                tf.keras.layers.Dense(200, activation=tf.nn.relu, name='encoder_hidden6', kernel_initializer='he_normal'),
                tf.keras.layers.Dense(25, activation='relu', name='encoder_code',
                kernel_initializer='he_normal')
            ],
            name='encoder'
        )

        # Decodes from low-dimensional code to output
        decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(200, activation=tf.nn.relu, name='decoder_hidden1', kernel_initializer='he_normal'),
                tf.keras.layers.Dense(200, activation=tf.nn.relu, name='decoder_hidden2', kernel_initializer='he_normal'),
                tf.keras.layers.Dense(200, activation=tf.nn.relu, name='decoder_hidden3', kernel_initializer='he_normal'),
                tf.keras.layers.Dense(200, activation=tf.nn.relu, name='decoder_hidden4', kernel_initializer='he_normal'),
                tf.keras.layers.Dense(200, activation=tf.nn.relu, name='decoder_hidden5', kernel_initializer='he_normal'),
                tf.keras.layers.Dense(200, activation=tf.nn.relu, name='decoder_hidden6', kernel_initializer='he_normal'),
                tf.keras.layers.Dense(12800, activation=tf.nn.sigmoid, name='decoder_output_flat', kernel_initializer='he_normal'),
            ],
            name='decoder'
        )

        # Parameter NN
        parameter = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(25, activation='relu', name='param_hidden1', kernel_initializer='he_normal'),
                tf.keras.layers.Dense(25, activation='relu', name='param_hidden2', kernel_initializer='he_normal'),
                tf.keras.layers.Dense(25, activation='relu', name='param_code',
                kernel_initializer='he_normal')
            ],
            name='parameter'
        )

        #######################################
        # # Encoder
        # # encoder = layers.Dense(self.nNeurons, activation='relu', kernel_initializer='he_normal')(input_img)
        # for _ in range(self.nHidLayers):
        #     self.encoder = layers.Dense(self.nNeurons, activation='relu', kernel_initializer='he_normal')
        # self.encoder = layers.Dense(self.codeSize, activation='relu', kernel_initializer='he_normal')
        
        # # Decoder
        # # decoded = layers.Dense(self.nNeurons, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1(hidReg))(encoded)
        # for _ in range(self.nHidLayers):
        #     decoder = layers.Dense(self.nNeurons, activation='relu', kernel_initializer='he_normal')
        # decoder = layers.Dense(self.data.dimension, activation='sigmoid', kernel_initializer='he_normal')

        # # Parameter
        # for _ in range(self.nHidLayersParam-1):
        #     parameter = layers.Dense(self.nNeuronsParam, activation='relu', kernel_initializer='he_normal')
        # parameter = layers.Dense(self.codeSize, activation='relu', kernel_initializer='he_normal')

        # # Create autoencoder and encoder objects
        # self.autoencoder = Model(input_img, decoded)
        # self.encoder = Model(input_img, encoded)
        #######################################

        # Connect sub-models
        self.x = tf.keras.Input(shape=(self.data.dimension,))
        mu = tf.keras.Input(shape=(2,))
        self.code_E = self.encoder(self.x)
        self.x_ED = decoder(self.code_E)
        self.code_P = parameter(mu)
        self.x_PD = decoder(self.code_P)

        self.autoencoder = Model(
            inputs={'x': self.x, 'mu': mu},
            outputs={'x_ED': self.x_ED,
                    'code_E': self.code_E,
                    'code_P': self.code_P,
                    'x_PD': self.x_PD
                    }
        )

        self.encoder = Model(
            inputs={'x': self.x},
            outputs={'code_E': self.code_E}
        )

        self.parameter = Model(
            inputs={'mu': mu},
            outputs={'code_P': self.code_P}
        )

        self.nTrainParam = int(np.sum([K.count_params(w) for w in self.autoencoder.trainable_weights]))
        self.nNonTrainParam = int(np.sum([K.count_params(w) for w in self.autoencoder.non_trainable_weights]))

        stop = timeit.default_timer()
        self.buildTime = stop - start
        summary()