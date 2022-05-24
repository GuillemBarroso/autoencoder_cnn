from tensorflow import keras
from common.postprocessing import summaryInfo
from common.training_data import Data
import timeit
import numpy as np
import matplotlib.pyplot as plt

class Model():
    def __init__(self,nn):
        self.optimizer = None
        self.loss = None
        self.compileTime = None
        self.epochs = None
        self.nBatch = None
        self.earlyStopPatience = None
        self.history = None
        self.min_loss = None
        self.min_valLoss = None
        self.trainTime = None
        self.nn = nn

    def compile(self,optimizer='adam', loss='mean_squared_error'):
        def summary():
            data = [['optimizer', self.optimizer],
            ['loss function', self.loss],
            ['compile time', '{:.2}s'.format(self.compileTime)]]
            name = 'results/compileModel_{}.png'.format(self.nn.data.dataset)
            summaryInfo(data, self.nn.verbose, self.nn.saveInfo, name)

        assert isinstance(optimizer, str), '"optimizer" must be a string'
        assert isinstance(loss, str), '"loss" must be a string'
        start = timeit.default_timer()
        self.optimizer = optimizer
        self.loss = loss
        self.nn.autoencoder.compile(optimizer=self.optimizer, loss=self.loss)

        stop = timeit.default_timer()
        self.compileTime = stop - start
        summary()
        if self.nn.verbose:
            self.nn.autoencoder.summary()

    def train(self, epochs=50, nBatch=32, earlyStopPatience=10, earlyStopTol=10e-4):
        assert isinstance(epochs, int), '"epochs" must be an integer'
        assert isinstance(nBatch, int), '"nBatch" must be an integer'
        assert isinstance(earlyStopPatience, int), '"earlyStopPatience" must be an integer'
        assert isinstance(earlyStopTol, float), '"earlyStopTol" must be a float'
        start = timeit.default_timer()
        self.epochs = epochs
        self.nBatch = nBatch
        self.earlyStopPatience = earlyStopPatience
        self.earlyStopTol = earlyStopTol
        earlyStop = keras.callbacks.EarlyStopping(patience=self.earlyStopPatience,monitor='val_loss',
                                                  restore_best_weights=True, min_delta=self.earlyStopTol)
        self.history = self.nn.autoencoder.fit(self.nn.data.x_train, self.nn.data.x_train, epochs=self.epochs,
                                       batch_size=self.nBatch, shuffle=True,
                                       validation_data=(self.nn.data.x_val, self.nn.data.x_val),
                                       verbose=self.nn.verbose,
                                       callbacks=[earlyStop])

        self.min_loss = min(self.history.history['loss'])
        self.min_valLoss = min(self.history.history['val_loss'])
        stop = timeit.default_timer()
        self.trainTime = stop - start
        if self.nn.verbose:
            plt.plot(self.history.epoch, self.history.history['loss'])
            plt.plot([x + 1 for x in self.history.epoch], self.history.history['val_loss'])
            plt.title('Model loss. Training time = {:.2}min'.format(self.trainTime/60))
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['training', 'validation'], loc='upper right')
            plt.savefig('results/{}_training.png'.format(self.nn.data.dataset))
            plt.show()

        ## TODO: add option to save model and load saved models

    def predict(self):
        ## TODO: refactor predict to a different class that makes predictions from a loaded model
        def getCodeInfo():
            avg = np.true_divide(self.code.sum(0), self.code.shape[0])
            self.codeRealSize = np.count_nonzero(avg)
            self.averageCodeMagnitude = np.true_divide(avg.sum(),(avg!=0).sum())

        def summary():
            data = [['epochs', self.epochs],
            ['nBatch', self.nBatch],
            ['early stop tol', '{:.0e}'.format(self.earlyStopTol)],
            ['early stop patience', '{} epochs'.format(self.earlyStopPatience)],
            ['training time', '{:.2}s'.format(self.trainTime)],
            ['min training loss', '{:.2}'.format(self.min_loss)],
            ['min validation loss', '{:.2}'.format(self.min_valLoss)],
            ['test loss evaluation', '{:.2}'.format(self.test_loss)],
            ['real code size', self.codeRealSize],
            ['avg pixel code mag', '{:.2}'.format(self.averageCodeMagnitude)]]
            name = 'results/trainModel_{}.png'.format(self.nn.data.dataset)
            summaryInfo(data, self.nn.verbose, self.nn.saveInfo, name)

        self.predictions = self.nn.autoencoder.predict(self.nn.data.x_test)
        # Scale 
        self.predictions = Data(self.nn.data.dataset, self.nn.data.dataMultCoef).normaliseArray(self.predictions)
        self.predictions = Data(self.nn.data.dataset, self.nn.data.dataMultCoef).thresholdArrayFilter(
            self.predictions, limits=[0, 1], tol=0)

        self.test_loss = self.nn.autoencoder.evaluate(
            self.nn.data.x_test, self.nn.data.x_test, verbose=self.nn.verbose)
        # TODO: compute error with postprocess filter
        self.code = self.nn.encoder.predict(self.nn.data.x_test)
        getCodeInfo()
        summary()