from tensorflow import keras
import tensorflow as tf
from src.postprocessing import summaryInfo, plotTraining, getLosses
import timeit
import numpy as np

class ModelNN():
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
        self.test_loss = None
        self.test_loss_per_image = None
        self.activeCode = None
        self.activeCode_E = None
        self.activeCode_P = None
        self.codeRealSize = None
        self.codeRealSize_E = None
        self.codeRealSize_P = None
        self.averageCodeMagnitude = None
        self.averageCodeMagnitude_E = None
        self.averageCodeMagnitude_P = None
        self.indivTestLoss = None
        self.nn = nn

    def compile(self,optimizer='adam', loss='mean_squared_error'):
        def summary():
            data = [['optimizer', self.optimizer],
            ['loss function', self.loss],
            ['compile time', '{:.2}s'.format(self.compileTime)]]
            name = 'results/compileModel_{}.png'.format(self.nn.model.data.dataset)
            summaryInfo(data, self.nn.model.verbose, self.nn.model.saveInfo, name)

        assert isinstance(optimizer, str), '"optimizer" must be a string'
        assert isinstance(loss, str), '"loss" must be a string'
        start = timeit.default_timer()
        self.optimizer = optimizer
        if self.nn.model.data.arch == 'param_ae':
            # Loss function terms
            coef = self.nn.model.data.dimension/self.nn.model.codeSize
            lossWeight = [1,1,1]
            # 1. Image loss: x = x_recon
            image_loss = tf.keras.losses.mean_squared_error(self.nn.model.x, self.nn.model.x_ED)
            self.nn.model.autoencoder.add_loss(lossWeight[0]*image_loss)

            # 2. Future state prediction loss: x1 = x1_pred
            code_loss = tf.keras.losses.mean_squared_error(self.nn.model.code_E, self.nn.model.code_P)
            self.nn.model.autoencoder.add_loss(lossWeight[1]*code_loss)

            imageParam_loss = tf.keras.losses.mean_squared_error(self.nn.model.x, self.nn.model.x_PD)
            self.nn.model.autoencoder.add_loss(lossWeight[2]*imageParam_loss)

            # Add metrics
            self.nn.model.autoencoder.add_metric(image_loss, name='L_image_ED')
            self.nn.model.autoencoder.add_metric(code_loss, name='L_code')
            self.nn.model.autoencoder.add_metric(imageParam_loss, name='L_image_PD')

            self.nn.model.autoencoder.summary()
            self.nn.model.autoencoder.compile(optimizer=self.optimizer) # loss_weights=[1,coef,1]

        else:
            self.loss = loss
            self.nn.model.autoencoder.compile(optimizer=self.optimizer, loss=self.loss)            

        stop = timeit.default_timer()
        self.compileTime = stop - start
        summary()
        if self.nn.model.verbose:
            self.nn.model.autoencoder.summary()

    def train(self, kw):
        assert isinstance(kw['EPOCHS'], int), '"epochs" must be an integer'
        assert isinstance(kw['N_BATCH'], int), '"nBatch" must be an integer'
        assert isinstance(kw['EARLY_STOP_PATIENCE'], int), '"earlyStopPatience" must be an integer'
        assert isinstance(kw['EARLY_STOP_TOL'], float), '"earlyStopTol" must be a float'
        start = timeit.default_timer()
        self.epochs = kw['EPOCHS']
        self.nBatch = kw['N_BATCH']
        self.earlyStopPatience = kw['EARLY_STOP_PATIENCE']
        self.earlyStopTol = kw['EARLY_STOP_TOL']
        earlyStop = keras.callbacks.EarlyStopping(patience=self.earlyStopPatience,monitor='val_loss',
                                                  restore_best_weights=True, min_delta=self.earlyStopTol)

        if self.nn.model.data.arch == 'param_ae':
            self.history = self.nn.model.autoencoder.fit(x={'x': self.nn.model.data.x_train, 'mu': self.nn.model.data.mu_train},
                                    epochs=self.epochs,
                                    batch_size=self.nBatch, shuffle=True,
                                    validation_data={'x': self.nn.model.data.x_val, 'mu': self.nn.model.data.mu_val},
                                    verbose=self.nn.model.verbose,
                                    callbacks=[earlyStop]
            )
        else:
            x_data = self.nn.model.data.x_train
            y_data = self.nn.model.data.x_train
            x_val = self.nn.model.data.x_val
            y_val = self.nn.model.data.x_val
            self.history = self.nn.model.autoencoder.fit(x=x_data, y=y_data, epochs=self.epochs,
                                       batch_size=self.nBatch, shuffle=True,
                                       validation_data=(x_val, y_val),
                                       verbose=self.nn.model.verbose,
                                       callbacks=[earlyStop]
            )

        stop = timeit.default_timer()
        self.trainTime = stop - start

        # Compute mean error for the entire test dataset
        if self.nn.model.data.arch == 'param_ae':
            self.test_loss = self.nn.model.autoencoder.evaluate(
                x = {'x': self.nn.model.data.x_test, 'mu': self.nn.model.data.mu_test}, verbose=self.nn.model.verbose)
        else:
            self.test_loss = self.nn.model.autoencoder.evaluate(
                self.nn.model.data.x_test, self.nn.model.data.x_test, verbose=self.nn.model.verbose)

        if self.nn.model.saveInfo:
            plotTraining(self.history, self.trainTime, self.nn.model.data.dataset)

        # Get losses from training
        self.losses = getLosses(self.history, self.test_loss, self.nn.model.data.dataset, self.nn.model.verbose, self.nn.model.saveInfo)

        ## TODO: add option to save model and load saved models

    def predict(self, kw):
        ## TODO: refactor predict to a different class that makes predictions from a loaded model
        def codeInfo(code):
            avg = np.true_divide(code.sum(0), code.shape[0])
            activeCode = np.nonzero(avg)[0]
            codeRealSize = len(activeCode)
            averageCodeMagnitude = np.true_divide(avg.sum(),(avg!=0).sum())
            
            return activeCode, codeRealSize, averageCodeMagnitude

        def getCodeInfo():
            self.activeCode, self.codeRealSize, self.averageCodeMagnitude = codeInfo(self.code)
            
        def getCodeInfoParam():
            self.activeCode_E, self.codeRealSize_E, self.averageCodeMagnitude_E = codeInfo(self.code_E)
            self.activeCode_P, self.codeRealSize_P, self.averageCodeMagnitude_P = codeInfo(self.code_P)

        def summary():
            data = [['epochs', self.epochs],
            ['nBatch', self.nBatch],
            ['early stop tol', '{:.0e}'.format(self.earlyStopTol)],
            ['early stop patience', '{} epochs'.format(self.earlyStopPatience)],
            ['training time', '{:.2}s'.format(self.trainTime)],
            ['real code size', self.codeRealSize],
            ['avg pixel code mag', '{:.2}'.format(self.averageCodeMagnitude)]]
            name = 'results/trainModel_{}.png'.format(self.nn.model.data.dataset)
            summaryInfo(data, self.nn.model.verbose, self.nn.model.saveInfo, name)

        def summary_param():
            data = [['epochs', self.epochs],
            ['nBatch', self.nBatch],
            ['early stop tol', '{:.0e}'.format(self.earlyStopTol)],
            ['early stop patience', '{} epochs'.format(self.earlyStopPatience)],
            ['training time', '{:.2}s'.format(self.trainTime)],
            ['real code_E size', self.codeRealSize_E],
            ['real code_P size', self.codeRealSize_P],
            ['avg pixel code_E mag', '{:.2}'.format(self.averageCodeMagnitude_E)],
            ['avg pixel code_P mag', '{:.2}'.format(self.averageCodeMagnitude_P)]]
            name = 'results/trainModel_{}.png'.format(self.nn.model.data.dataset)
            summaryInfo(data, self.nn.model.verbose, self.nn.model.saveInfo, name)

        assert isinstance(kw['INDIV_TEST_LOSS'], bool), '"indivTestLoss" must be a boolean'
        self.indivTestLoss = kw['INDIV_TEST_LOSS']

        if self.nn.model.data.arch == 'param_ae':
            self.predictions = self.nn.model.autoencoder.predict(x = {'x': self.nn.model.data.x_test, 'mu': self.nn.model.data.mu_test})
        else:
            self.predictions = self.nn.model.autoencoder.predict(self.nn.model.data.x_test)
                
        # Get individual errors for each of the test image selected manually
        if self.nn.model.data.arch == 'fcnn' or self.nn.model.data.arch == 'param_ae':
            aux = (1, self.nn.model.data.dimension)
        elif self.nn.model.data.arch == 'cnn':
            aux = (1, self.nn.model.data.resolution[0], self.nn.model.data.resolution[1])
        else:
            raise ValueError('Check this for other architectures!')

        if self.indivTestLoss:
            self.test_loss_per_image = []
            for i in range(self.nn.model.data.nTest):
                if self.nn.model.data.arch == 'param_ae':
                    self.test_loss_per_image.append(
                        self.nn.model.autoencoder.evaluate(
                            x = {'x': self.nn.model.data.x_test[i].reshape(aux), 'mu': self.nn.model.data.mu_test[i].reshape(1,2)},
                            verbose=self.nn.model.verbose)
                )
                else:
                    self.test_loss_per_image.append(
                        self.nn.model.autoencoder.evaluate(
                            self.nn.model.data.x_test[i].reshape(aux), 
                            self.nn.model.data.x_test[i].reshape(aux), verbose=self.nn.model.verbose)
                )

        # TODO: compute error with postprocess filter
        if self.nn.model.data.arch == 'param_ae':
            self.code_E = self.nn.model.encoder.predict(x = {'x': self.nn.model.data.x_test})['code_E']
            self.code_P = self.nn.model.parameter.predict(x = {'mu': self.nn.model.data.mu_test})['code_P']
        else:
            self.code = self.nn.model.encoder.predict(self.nn.model.data.x_test)

        if self.nn.model.data.arch == 'param_ae':
            getCodeInfoParam()
            summary_param()
        else:
            getCodeInfo()
            summary()