from training_data import Data
from cnn_model import Model


def loadData_testing(testingDatasets):
    print('TESTING DATA LOADING:')
    for dataset in testingDatasets:
        data = Data(dataset)
        data.load()
        checkDim(data)
        checkScale(data, [0,1])
        print('Data loaded correctly for "{}" dataset'.format(dataset))
    print('\n')


def train_testing(testingDatasets):
    print('TESTING CNN TRAINING:')
    for dataset in testingDatasets:
        data = Data(dataset)
        data.load()

        try:
            model = Model(data)
        except:
            raise ValueError('Failed when creating model.')
        try:
            model.build(nConvLayers=2,nFilters=10, kernel=3, stride=2)
        except:
            raise ValueError('Failed when building model.')
        try:
            model.compile(optimizer='adam',loss='mean_squared_error')
        except:
            raise ValueError('Failed when compiling model.')
        try:
            model.train(epochs=1, nBatch=256)
        except:
            raise ValueError('Failed when training model.')
        try:
            model.predict()
        except:
            raise ValueError('Failed when predicting.')

        print('Training successfully tested for "{}" dataset'.format(dataset))


def checkDim(data):
    assert data.x_train.shape[1:3] == data.resolution, 'Training dataset dimension is incorrect'
    assert data.x_test.shape[1:3] == data.resolution, 'Testing dataset dimension is incorrect'


def checkScale(data, lim):
    assert data.x_train.min() >= lim[0] and data.x_train.max() <= lim[1],\
        'Training dataset not scaled between {} and {}'.format(lim[0],lim[1])
    assert data.x_test.min() >= lim[0] and data.x_test.max() <= lim[1],\
        'Testing dataset not scaled between {} and {}'.format(lim[0],lim[1])


if __name__ == '__main__':
    testingDatasets = ['mnist', 'afreightdata_test']
    loadData_testing(testingDatasets)
    train_testing(testingDatasets)

