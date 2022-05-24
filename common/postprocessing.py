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

    plt.figure(figsize=(20, 5))
    for i in range(numDisplay):
        # Display original
        ax = plt.subplot(4, numDisplay, i + 1)
        plt.imshow(data.x_test[i+i*i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display code
        codeSizeSqrt = int(np.sqrt(model.nn.codeSize))
        ax = plt.subplot(4, numDisplay, i + 1 + numDisplay)
        plt.imshow(model.code[i+i*i].reshape(codeSizeSqrt,codeSizeSqrt))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(4, numDisplay, i + 1 + 2*numDisplay)
        plt.imshow(model.predictions[i+i*i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display error for each test image
        imageError = '{:.2}'.format(model.test_loss_per_image[i+i*i])
        ax = plt.subplot(4, numDisplay, i + 1 + 3*numDisplay)
        ax.text(0.15,0.5,'mu1 = {}'.format(data.mu1_test[i+i*i]))
        ax.text(0.15,0.3,'mu2 = {}'.format(data.mu2_test[i+i*i]))
        ax.text(0.15,0.1,'image loss = {}'.format(imageError))
        ax.axis('off')
        if model.nn.verbose:
            plt.savefig('results/{}_prediction.png'.format(data.dataset))
    plt.show()

    if data.imgTestList:
        mu1_tot, mu2_tot = data.datasetClass.getMuDomain()
        data.datasetClass.plotMuDomain(mu1_tot, mu2_tot, data.mu1_test, data.mu2_test)
        displayedImages = [a+b*c for a,b,c in zip(range(numDisplay), range(numDisplay), range(numDisplay))]
        fig, ax = plt.subplots()
        for i in range(len(model.test_loss_per_image)):
            imageError = '{:.2}'.format(model.test_loss_per_image[i])
            if i in displayedImages:
                colRef = 'green'
            else:
                colRef = 'red'
            ax.scatter(data.mu1_test[i], data.mu2_test[i], color=colRef)
            ax.text(data.mu1_test[i], data.mu2_test[i], imageError)
        plt.xlabel("mu_1 (position)")
        plt.ylabel("mu_2 (angle in º)")
        plt.show()



def summaryInfo(info, printInfo, saveInfo, name):
    df = pd.DataFrame(info, columns=['Parameter', 'Value'])
    if printInfo:
        print(df)
    if saveInfo:
        dfi.export(df, name)