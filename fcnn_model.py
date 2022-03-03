from tensorflow import keras
from tensorflow.keras import layers

class FCNN():
    def __init__(self, data, verbose=False, saveInfo=False):
        self.autoencoder = None
        self.encoder = None
        self.data = data
        self.verbose = verbose
        self.saveInfo = saveInfo

    def build(self, codeSize=25, nNeurons=40, nHidLayers=2):
        self.codeSize = codeSize
        self.nNeurons = nNeurons
        self.nHidLayers = nHidLayers

        input_img = keras.Input(shape=(self.data.dimension,))
        
        # Encoder
        encoded = layers.Dense(nNeurons, activation='relu', kernel_initializer='he_normal')(input_img)
        for _ in range(nHidLayers-1):
            encoded = layers.Dense(nNeurons, activation='relu', kernel_initializer='he_normal')(encoded)
        
        # Code
        encoded = layers.Dense(self.codeSize, activation='relu', kernel_initializer='he_normal', 
        kernel_regularizer=keras.regularizers.l1(1e-4))(encoded)
        
        # Decoder
        decoded = layers.Dense(nNeurons, activation='relu', kernel_initializer='he_normal')(encoded)
        for _ in range(nHidLayers-1):
            decoded = layers.Dense(nNeurons, activation='relu', kernel_initializer='he_normal')(decoded)
        decoded = layers.Dense(self.data.dimension)(decoded)

        # Create autoencoder and encoder objects
        self.autoencoder = keras.Model(input_img, decoded)
        self.encoder = keras.Model(input_img, encoded)
