from keras.datasets import mnist
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import os, os.path
from prettytable import PrettyTable
import timeit


class Data():
    def __init__(self, dataset, testSize=0.1, verbose=False):
        self.x_test = None
        self.x_train = None
        self.dimension = None
        self.resolution = None
        self.scale = None
        self.nTrain = None
        self.nTest = None
        self.nVal = None
        self.dirPath = None

        assert isinstance(dataset, str), '"dataset" variable must be a string'
        assert isinstance(testSize, (int,float)), '"testSize" variable must be an integer or a float'
        assert 0 <= testSize <= 1, 'testSize should be in [0,1]'
        assert isinstance(verbose, bool), '"verbose" variable must be a boolean'

        self.testSize = testSize
        self.dataset = dataset
        self.verbose = verbose

    def load(self):
        def existsDirectory():
            return os.path.isdir(self.dirPath)

        def scale():
            self.x_train = self.x_train.astype('float32') / 255.
            self.x_val = self.x_val.astype('float32') / 255.
            self.x_test = self.x_test.astype('float32') / 255.
            self.scale = (min(self.x_train.min(), self.x_test.min(), self.x_val.min()),
                          max(self.x_train.max(), self.x_test.max(), self.x_val.max()))

        def getDataSize():
            self.nTrain = self.x_train.shape[0]
            self.nTest = self.x_test.shape[0]
            self.nVal = self.x_val.shape[0]

        start = timeit.default_timer()
        if self.dataset == 'mnist':
            (self.x_train, _), (self.x_test, _) = mnist.load_data()
            self.x_train, self.x_val = train_test_split(np.asarray(self.x_train), test_size=0.1, shuffle=False)
            self.getImageData(self.x_train[0])
        else:
            self.dirPath = './{}'.format(self.dataset)
            if existsDirectory():
                self.x_train, self.x_val, self.x_test = self.loadImagesFromDir()
            else:
                raise ValueError('Invalid dataset name. "dataset" should be the name of the directory')
        scale()
        getDataSize()
        stop = timeit.default_timer()
        self.loadTime = stop - start

    def loadImagesFromDir(self):
        def findFirstValidFile(imgList):
            findImageData = True
            i = 0
            while findImageData and i <= len(imgList):
                try:
                    array = self.openImageToArray(imgList[i])
                    self.getImageData(array)
                    findImageData = False
                except:
                    i += 1

        imgList = os.listdir(self.dirPath)
        findFirstValidFile(imgList)

        # Loop over images and store them in the proper format
        data = []
        for imgName in imgList:
            try:  # Try so it accepts having other files or folders that are not images inside the same directory
                array = self.openImageToArray(imgName)
                assert self.checkImageSize(array), 'Images with different sizes'
                assert self.checkChannels(), 'Some images are coloured and some are grey-scale'
                data.append(array)
            except Exception as e:
                if self.verbose:
                    print('Ignoring file when loading from {}. Error: {}'.format(self.dataset, e))

        valSize = self.testSize/(1-self.testSize)
        x_train, x_test = train_test_split(np.asarray(data), test_size=self.testSize, shuffle=False)
        x_train, x_val = train_test_split(np.asarray(x_train), test_size=valSize, shuffle=True)
        return x_train, x_val, x_test

    def openImageToArray(self, imgName):
        assert isinstance(imgName, str)
        img = Image.open('{}/{}'.format(self.dirPath, imgName))
        return np.asarray(img.getdata()).reshape(img.height, img.width, 3)

    def getImageData(self, image):
        self.assertNdarray(image)
        self.resolution = (image.shape[0], image.shape[1], self.getChannels(image))
        self.dimension = np.prod(self.resolution[0:2])

    def getChannels(self, image):
        self.assertNdarray(image)
        try:
            channels = image.shape[2]
        except:
            channels = 1
        return channels

    def checkImageSize(self, image):
        self.assertNdarray(image)
        return image.shape == self.resolution

    def checkChannels(self):
        return self.resolution[2]

    def assertNdarray(self, array):
        assert type(array).__module__ == np.__name__

    def rgb2greyScale(self):
        rgb_weights = [0.2989, 0.5870, 0.1140]
        if self.resolution[2] == 3:
            self.x_train = np.dot(self.x_train[:], rgb_weights)
            self.x_val = np.dot(self.x_val[:], rgb_weights)
            self.x_test = np.dot(self.x_test[:], rgb_weights)
            self.resolution = (self.resolution[0], self.resolution[1], 1)

    def summary(self):
        dataInfo = PrettyTable(['Parameter', 'Value'])
        dataInfo.title = 'Load data'
        dataInfo.add_row(['dataset name', self.dataset])
        dataInfo.add_row(['image resolution', self.resolution])
        dataInfo.add_row(['scale', self.scale])
        dataInfo.add_row(['train data size', self.nTrain])
        dataInfo.add_row(['validation data size', self.nVal])
        dataInfo.add_row(['test data size', self.nTest])
        dataInfo.add_row(['load time', '{:.2}s'.format(self.loadTime)])
        print(dataInfo)