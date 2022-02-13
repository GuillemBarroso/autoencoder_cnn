import matplotlib.pyplot as plt
import numpy as np


def plottingPrediction(data, model, numDisplay):
    if numDisplay > data.x_test.shape[0]:
        numDisplay = data.x_test.shape[0]
    for i in range(numDisplay):
        # Display original
        ax = plt.subplot(3, numDisplay, i + 1)
        plt.imshow(data.x_test[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display code
        codeSizeSqrt = int(np.sqrt(model.codeSize))
        ax = plt.subplot(3, numDisplay, i + 1 + numDisplay)
        plt.imshow(model.code[i].reshape(codeSizeSqrt,codeSizeSqrt))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(3, numDisplay, i + 1 + 2*numDisplay)
        plt.imshow(model.predictions[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plotTraining(history, trainTime):
    plt.plot(history.epoch, history.history['loss'])
    plt.plot([x + 1 for x in history.epoch], history.history['val_loss'])
    plt.title('Model loss. Training time = {:.2}min'.format(trainTime/60))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
