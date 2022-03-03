import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dataframe_image as dfi


def plottingPrediction(data, model, numDisplay):
    if numDisplay > data.x_test.shape[0]:
        numDisplay = data.x_test.shape[0]
    
    if not data.x_test.shape == data.resolution:
        data.x_test = data.x_test.reshape(data.nTest, data.resolution[0],data.resolution[1],data.resolution[2])
        model.predictions = model.predictions.reshape(data.nTest, data.resolution[0],data.resolution[1],data.resolution[2])

    plt.figure(figsize=(20, 4))
    for i in range(numDisplay):
        # Display original
        ax = plt.subplot(3, numDisplay, i + 1)
        plt.imshow(data.x_test[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display code
        codeSizeSqrt = int(np.sqrt(model.nn.codeSize))
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
    if model.nn.verbose:
        plt.savefig('results/{}_prediction.png'.format(data.dataset))
    plt.show()


def summaryInfo(info, printInfo, saveInfo, name):
    df = pd.DataFrame(info, columns=['Parameter', 'Value'])
    if printInfo:
        print(df)
    if saveInfo:
        dfi.export(df, name)