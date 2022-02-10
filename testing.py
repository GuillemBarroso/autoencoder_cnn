from training_data import Data
from cnn_model import Model
import numpy as np


def loadData_testing(testingData):
    print('TESTING DATA LOADING:')
    for i, dataset in enumerate(testingData['dataset']):
        data = Data(dataset)
        data.load()
        checkScale(data, [0, 255])
        if not testingData['colour'][i]:
            data.rgb2greyScale()
        checkDim(data, testingData['resolution'][i])
        checkChannels(data, testingData['resolution'][i])
        checkScale(data, [0,1])
        print('Data loaded correctly for "{}" dataset'.format(dataset))
    print('\n')


def train_testing(testingData):
    print('TESTING CNN TRAINING:')
    for i, dataset in enumerate(testingData['dataset']):
        data = Data(dataset)
        data.load()
        data = getOneDataPoint(data)
        if not testingData['colour'][i]:
            data.rgb2greyScale()

        try:
            model = Model(data)
        except:
            raise ValueError('Failed when creating model.')
        try:
            model.build(nConvLayers=1,nFilters=1, kernel=3, stride=2)
        except:
            raise ValueError('Failed when building model.')
        try:
            model.compile(optimizer='adam',loss='mean_squared_error')
        except:
            raise ValueError('Failed when compiling model.')
        try:
            model.train(epochs=1, nBatch=256, earlyStopPatience=10)
        except:
            raise ValueError('Failed when training model.')
        try:
            model.predict()
        except:
            raise ValueError('Failed when predicting.')

        print('Training successfully tested for "{}" dataset'.format(dataset))


def checkDim(data, refResolution):
    assert data.x_train.shape[1:3] == refResolution[0:2], 'Training dataset dimension is incorrect'
    assert data.x_val.shape[1:3] == refResolution[0:2], 'Validation dataset dimension is incorrect'
    assert data.x_test.shape[1:3] == refResolution[0:2], 'Testing dataset dimension is incorrect'


def checkChannels(data, refResolution):
    assert data.resolution[2] == refResolution[2], 'Data has an incorrect number of channels'


def checkScale(data, lim):
    assert data.x_train.min() >= lim[0] and data.x_train.max() <= lim[1],\
        'Training dataset not scaled between {} and {}'.format(lim[0],lim[1])
    assert data.x_val.min() >= lim[0] and data.x_val.max() <= lim[1], \
        'Validation dataset not scaled between {} and {}'.format(lim[0], lim[1])
    assert data.x_test.min() >= lim[0] and data.x_test.max() <= lim[1],\
        'Testing dataset not scaled between {} and {}'.format(lim[0],lim[1])

def getOneDataPoint(data):
    data.x_train = data.x_train[0].reshape(1, data.resolution[0], data.resolution[1], data.resolution[2])
    data.x_val = data.x_val[0].reshape(1, data.resolution[0], data.resolution[1], data.resolution[2])
    data.x_test = data.x_test[0].reshape(1, data.resolution[0], data.resolution[1], data.resolution[2])
    return data


if __name__ == '__main__':
    testingData = {
        'dataset' : ['mnist', 'afreightdata_test', 'afreightdata_test'],
        'colour' : [False, False, True],
        'resolution' : [(28, 28, 1), (120, 160, 1), (120, 160, 3)]
    }
    loadData_testing(testingData)
    train_testing(testingData)

