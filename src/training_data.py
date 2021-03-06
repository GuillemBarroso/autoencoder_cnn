from multiprocessing.sharedctypes import Value
from keras.datasets import mnist
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import os, os.path
import timeit
from src.postprocessing import summaryInfo
from src.read_txt import Mesh, TxtData
from src.beam_homog import BeamHomog
from src.beam_homog_big import BeamHomogBig


class Data():
    def __init__(self, kw):
        self.x_test = None
        self.x_train = None
        self.dimension = None
        self.resolution = None
        self.format = None
        self.scale = None
        self.nTrain = None
        self.nTest = None
        self.nVal = None
        self.dirPath = None
        self.summary = None
        self.mesh = None
        self.imgList = None
        self.imgTestList = None
        self.datasetClass = None
        self.parametricProblem = None
        self.paramTrain = [[], []]
        self.paramVal = [[], []]
        self.paramTest = [[], []]
        self.mu_train = None
        self.mu_val = None
        self.mu_test = None

        assert isinstance(kw['DATASET'], str), '"dataset" variable must be a string'
        assert isinstance(kw['TEST_DATA'], (int,float, list)), '"testData" variable must be an integer, a float or a list'
        if not isinstance(kw['TEST_DATA'], list):
            assert 0 <= kw['TEST_DATA'] <= 1, 'If float, "testData" should be between [0,1]'
        assert isinstance(kw['VERBOSE'], bool), '"verbose" variable must be a boolean'
        assert isinstance(kw['SAVE_INFO'], bool), '"saveSummary" variable must be a boolean'
        assert isinstance(kw['ARCH'], str), '"arch" variable must be a string'

        self.testData = kw['TEST_DATA']
        self.dataset = kw['DATASET']
        self.verbose = kw['VERBOSE']
        self.saveInfo = kw['SAVE_INFO']
        self.arch = kw['ARCH']

    def load(self):
        def existsDirectory():
            return os.path.isdir(self.dirPath)

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
            self.scale = self.getArrayScale(self.x_train)
        else:
            self.dirPath = './datasets/{}'.format(self.dataset)
            if existsDirectory():
                self.loadImagesFromDir()
            else:
                raise ValueError('Invalid dataset name. "dataset" should be the name of the directory')

        stop = timeit.default_timer()
        self.loadTime = stop - start

        self.normaliseDataset()
        getDataSize()
        summary()
        if self.arch == 'fcnn' or 'param_ae':
            self.rehsapeDataToArray()

    def loadImagesFromDir(self):
        def findFirstValidFile(imgList):
            findImageData = True
            i = 0
            while findImageData and i <= len(imgList):
                try:
                    _ = self.openImageToArray(imgList[i], isFirst=True)
                    findImageData = False
                except Exception as e:
                    if self.verbose:
                        print('Error when reading images for the first time. Error: {}'.format(e))
                    i += 1
            if findImageData:
                raise ValueError('First valid image not found!')

        self.imgList = os.listdir(self.dirPath)
        findFirstValidFile(self.imgList)

        #??Load extra info from certain datasets (parametric datasets)
        if self.dataset == 'beam_homog':
            self.datasetClass = BeamHomog()
            self.parametricProblem = True
        elif self.dataset == 'beam_homog_big':
            self.datasetClass = BeamHomogBig()
            self.parametricProblem = True
        elif self.dataset == 'beam_homog_test':
            self.datasetClass = BeamHomog()
            self.parametricProblem = True

        # Loop over images and store them in the proper format
        data = []
        aux = []
        params = [[], []]

        for imgName in self.imgList:
            try:  # Try so it accepts having other files or folders that are not images inside the same directory
                array = self.openImageToArray(imgName)
                assert self.checkImageSize(array), 'Images with different sizes'
                assert self.checkChannels(), 'Images with different number of channels'
                assert self.checkFormat(imgName), 'Images with different formats'
                data.append(array)
                aux.append(imgName)

                if self.datasetClass:
                    mus = self.getMusFromImgName(imgName)
                    params[0].append(mus[0])
                    params[1].append(mus[1])

            except Exception as e:
                if self.verbose:
                    print('Ignoring file when loading from {}. Error: {}'.format(self.dataset, e))
            self.imgList = aux

        if isinstance(self.testData, list):
            self.testData, _, _ = self.datasetClass.getImageNamesFromMus(self.testData[0], self.testData[1])
            x_noTest = []
            x_test = []
            self.imgTestList = []
            paramNoTest = [[], []]

            for iImage, imgName in enumerate(self.imgList):
                if imgName in self.testData:
                    x_test.append(data[iImage])
                    self.imgTestList.append(imgName)
                    self.paramTest[0].append(params[0][iImage])
                    self.paramTest[1].append(params[1][iImage])
                else:
                    x_noTest.append(data[iImage])
                    paramNoTest[0].append(params[0][iImage])
                    paramNoTest[1].append(params[1][iImage])

            if len(x_test) != len(self.testData):
                raise ValueError('WARNING: number of test images requested is {}. Found {} with the same name in dataset.'.format(
                    len(self.testData), len(x_test)))

            if self.parametricProblem:
                self.x_train, self.x_val, self.paramTrain[0], self.paramVal[0], self.paramTrain[1], self.paramVal[1] = train_test_split(
                    np.asarray(x_noTest),np.asarray(paramNoTest[0]), np.asarray(paramNoTest[1]), test_size=0.1, shuffle=True)
            else:
                self.x_train, self.x_val, = train_test_split(np.asarray(x_noTest), test_size=0.1, shuffle=True)

            self.x_test = np.asarray(x_test)

        else:
            valSize = self.testData/(1-self.testData)
            idx = range(len(data))
            if self.parametricProblem:
                self.x_train, self.x_test, self.paramTrain[0], self.paramTest[0], self.paramTrain[1], self.paramTest[1], _, idx_test = train_test_split(
                    np.asarray(data), np.asarray(params[0]), np.asarray(params[1]), idx, test_size=self.testData, shuffle=True)
                self.x_train, self.x_val, self.paramTrain[0], self.paramVal[0], self.paramTrain[1], self.paramVal[1] = train_test_split(
                    self.x_train, self.paramTrain[0], self.paramTrain[1], test_size=valSize, shuffle=True)
            else:
                self.x_train, self.x_test, _, idx_test = train_test_split(np.asarray(data), idx, test_size=self.testData, shuffle=True)
                self.x_train, self.x_val = train_test_split(self.x_train, test_size=valSize, shuffle=True)
            
            self.imgTestList = [self.imgList[x] for x in idx_test]
        if self.parametricProblem:
            self.mu_train = np.vstack((self.paramTrain[0], self.paramTrain[1])).T
            self.mu_val = np.vstack((self.paramVal[0], self.paramVal[1])).T
            self.mu_test = np.vstack((self.paramTest[0], self.paramTest[1])).T
    
    def getMusFromImgName(self, imgName):
        Fh, Fv, loc, pos = self.datasetClass.getParamsFromImageName(imgName)
        return self.datasetClass.getMusFromParams(Fh, Fv, loc, pos)

    def openImageToArray(self, imgName, isFirst=None):
        assert isinstance(imgName, str)
        if self.getFormat(imgName) == 'png':
            array = self.readPNG(imgName, isFirst)
        elif self.getFormat(imgName) == 'txt':
            array = self.readTXT(imgName, isFirst)
        else:
            raise ValueError('Invalid format. Implemented image formats are PNG and TXT')
        return array
    
    def readPNG(self, imgName, isFirst):
        img = Image.open('{}/{}'.format(self.dirPath, imgName))
        array = np.asarray(img.getdata()).reshape(self.resolution)
        if isFirst:
            self.resolution = (img.height, img.width, self.getChannels(img))
            self.format = self.getFormat(imgName)
            self.updateDimension()
            self.scale = self.getArrayScale(array)
        return array

    def readTXT(self, imgName, isFirst):
        if isFirst:
            self.mesh = Mesh()
            self.mesh.getMesh(self.dataset)

        fileObj = TxtData(self.mesh)
        values = fileObj.extractValuesFromFile(imgName)
        array = fileObj.computePixelsFromP0values(values)

        if isFirst:        
            self.resolution = array.shape
            self.format = self.getFormat(imgName)
            self.updateDimension()
            self.scale = self.getArrayScale(array)
        return array

    def getChannels(self, image):
        if image.mode == 'L' or image.mode == 'P':
            channels = 1
        elif image.mode == 'RGB':
            channels = 3
        elif image.mode == 'RGBA':
            channels = 4
        else:
            raise ValueError('Image mode not implemented (for now only "L", "RGB" and "RBGA")')
        return channels

    def getFormat(self, name):
        assert isinstance(name, str)
        return name[name.rfind('.')+1:]

    def checkImageSize(self, image):
        self.assertNdarray(image)
        return image.shape == self.resolution

    def checkChannels(self):
        return self.resolution[2]

    def checkFormat(self, imgName):
        assert isinstance(imgName, str)
        return self.getFormat(imgName) == self.format

    def updateDimension(self):
        self.dimension = np.prod(self.resolution)

    def assertNdarray(self, array):
        assert type(array).__module__ == np.__name__

    def normaliseDataset(self):
        # maxVal = max(np.amax(self.x_train), np.amax(self.x_val), np.amax(self.x_test))
        # normalising the 3 datasets differently (normalising each at a time), is that OK? 
        self.x_train = self.normaliseArray(self.x_train)
        self.x_val = self.normaliseArray(self.x_val)
        self.x_test = self.normaliseArray(self.x_test)
        self.updateScale()

    def updateScale(self):
        self.scale = (min(self.x_train.min(), self.x_test.min(), self.x_val.min()),
                max(self.x_train.max(), self.x_test.max(), self.x_val.max()))

    def normaliseArray(self, arr):
        return (arr.astype('float32') - np.amin(arr)) / (np.amax(arr) - np.amin(arr) )

    def getArrayScale(self, arr):
        return (arr.min(), arr.max())

    def rgb2greyScale(self):
        rgb_weights = [0.2989, 0.5870, 0.1140]
        if self.resolution[2] == 3 or self.resolution[2] == 4:
            self.x_train = np.dot(self.x_train[:,:,:,:3], rgb_weights)
            self.x_val = np.dot(self.x_val[:,:,:,:3], rgb_weights)
            self.x_test = np.dot(self.x_test[:,:,:,:3], rgb_weights)
            self.resolution = (self.resolution[0], self.resolution[1], 1)
            self.updateDimension()
            
        elif self.resolution[2] == 1:
            print('Image already in grey scale')
        else:
            raise ValueError('Number of channels in the image not supported. It must be either 1 or 3.')

    def blackAndWhite(self,threshold=0.3):
        if self.resolution[2] == 1:
            self.x_train = np.where(self.x_train > threshold, 1, 0)
            self.x_val = np.where(self.x_val > threshold, 1, 0)
            self.x_test = np.where(self.x_test > threshold, 1, 0)
            self.updateScale()
        else:
            print('BlackAndWhite method only supported for greyScale images. Since dataset is coloured this option has been neglected.')

    def thresholdFilter(self, tol=1e-8):
        limits = [0, 1]
        self.x_train = self.thresholdArrayFilter(self.x_train, limits, tol)
        self.x_val = self.thresholdArrayFilter(self.x_val, limits, tol)
        self.x_test = self.thresholdArrayFilter(self.x_test, limits, tol)
        self.updateScale()

    def thresholdArrayFilter(self, arr, limits, tol):
        arr = np.where(arr < limits[0] + tol, limits[0], arr)
        arr = np.where(arr > limits[1] - tol, limits[1], arr)
        return arr

    def rehsapeDataToArray(self):
        self.x_train = self.x_train.reshape(self.nTrain,self.dimension)
        self.x_val = self.x_val.reshape(self.nVal,self.dimension)
        self.x_test = self.x_test.reshape(self.nTest,self.dimension)