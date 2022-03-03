from multiprocessing.sharedctypes import Value
from keras.datasets import mnist
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import os, os.path
import timeit
from postprocessing import summaryInfo

class Data():
    def __init__(self, dataset, testSize=0.1, verbose=False, saveInfo=False):
        self.x_test = None
        self.x_train = None
        self.dimension = None
        self.resolution = None
        self.scale = None
        self.nTrain = None
        self.nTest = None
        self.nVal = None
        self.dirPath = None
        self.summary = None

        assert isinstance(dataset, str), '"dataset" variable must be a string'
        assert isinstance(testSize, (int,float)), '"testSize" variable must be an integer or a float'
        assert 0 <= testSize <= 1, 'testSize should be in [0,1]'
        assert isinstance(verbose, bool), '"verbose" variable must be a boolean'
        assert isinstance(saveInfo, bool), '"saveSummary" variable must be a boolean'

        self.testSize = testSize
        self.dataset = dataset
        self.verbose = verbose
        self.saveInfo = saveInfo

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

        def summary():
            data = [['dataset name', self.dataset],
            ['image resolution', self.resolution],
            ['scale', self.scale],
            ['train data size', self.nTrain],
            ['validation data size', self.nVal],
            ['test data size', self.nTest],
            ['load time', '{:.2}s'.format(self.loadTime)]]
            name = 'results/loadData_{}.png'.format(self.dataset)
            summaryInfo(data, self.verbose, self.saveInfo, name)

        start = timeit.default_timer()
        if self.dataset == 'mnist':
            (self.x_train, _), (self.x_test, _) = mnist.load_data()
            self.x_train, self.x_val = train_test_split(np.asarray(self.x_train), test_size=0.1, shuffle=False)
            self.resolution = (self.x_train[0].shape[0], self.x_train[0].shape[1], 1)
            self.dimension = np.prod(self.resolution[0:2])
        else:
            self.dirPath = './{}'.format(self.dataset)
            if existsDirectory():
                self.x_train, self.x_val, self.x_test = self.loadImagesFromDir()
            else:
                raise ValueError('Invalid dataset name. "dataset" should be the name of the directory')

        stop = timeit.default_timer()
        self.loadTime = stop - start

        scale()
        getDataSize()
        summary()

    def loadImagesFromDir(self):
        def findFirstValidFile(imgList):
            findImageData = True
            i = 0
            while findImageData and i <= len(imgList):
                try:
                    array = self.openImageToArray(imgList[i], isFirst=True)
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

    def openImageToArray(self, imgName, isFirst=None):
        assert isinstance(imgName, str)
        img = Image.open('{}/{}'.format(self.dirPath, imgName))
        if isFirst:
            self.resolution = (img.height, img.width, self.getChannels(img))
            self.dimension = np.prod(self.resolution[0:2])
        return np.asarray(img.getdata()).reshape(self.resolution)

    def getChannels(self, image):
        if image.mode == 'L':
            channels = 1
        elif image.mode == 'RGB':
            channels = 3
        else:
            raise ValueError('Image mode not implemented (for now only "L" and "RGB")')
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
        elif self.resolution[2] == 1:
            print('Image already in grey scale')
        else:
            raise ValueError('Number of channels in the image not supported. It must be either 1 or 3.')

    def blackAndWhite(self,threshold=0.3):
        if self.resolution[2] == 1:
            self.x_train  = np.where(self.x_train > threshold, 1, 0)
            self.x_val  = np.where(self.x_val > threshold, 1, 0)
            self.x_test  = np.where(self.x_test > threshold, 1, 0)
        else:
            print('BlackAndWhite method only supported for greyScale images. Since dataset is coloured this option has been neglected.')

    def rehsapeDataToArray(self):
        self.x_train = self.x_train.reshape(self.nTrain,self.dimension)
        self.x_val = self.x_val.reshape(self.nVal,self.dimension)
        self.x_test = self.x_test.reshape(self.nTest,self.dimension)