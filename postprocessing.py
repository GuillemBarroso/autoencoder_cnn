import matplotlib.pyplot as plt


def plottingPrediction(data, model, numDisplay):
    if numDisplay > data.x_test.shape[0]:
        numDisplay = data.x_test.shape[0]
    plt.figure(figsize=(20, 4))
    for i in range(numDisplay):
        # Display original
        ax = plt.subplot(2, numDisplay, i + 1)
        plt.imshow(data.x_test[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, numDisplay, i + 1 + numDisplay)
        plt.imshow(model.predictions[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plotTraining(history, trainTime):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss. Training time = {:.2}min'.format(trainTime/60))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()


def plottingCodeFiltres(model,nDisplay,nFilters):
    i = 0
    for iImage in range(nDisplay):
        for iFilter in range(nFilters):
            ax = plt.subplot(nDisplay, nFilters, i + 1)
            plt.imshow(model.code[iImage,:,:,iFilter])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            i += 1
    plt.show()
