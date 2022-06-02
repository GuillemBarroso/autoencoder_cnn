import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dataframe_image as dfi


def plottingPrediction(data, model, imgDisplay):
    if isinstance(imgDisplay, int):
        # Set test image that will be displayed if the user does not specify them
        numDisplay = imgDisplay
        imgDispList = [a+b*c for a,b,c in zip(range(numDisplay), range(numDisplay), range(numDisplay))]
        muDispList = [[data.paramTest[0][x] for x in imgDispList], [data.paramTest[1][x] for x in imgDispList]]

    elif isinstance(imgDisplay, list):
        assert len(imgDisplay[0]) == len(imgDisplay[1]), 'mu1_test and mu2_test must have the same length!'
        numDisplay = len(imgDisplay[0])
        muDispList = imgDisplay
        imgDispList = []
        for i in range(numDisplay):
            Fh, Fv, loc, pos = data.datasetClass.getParamsFromMus(imgDisplay[0][i], imgDisplay[1][i])
            name = 'Fh{}_Fv{}_{}{}.txt'.format(Fh, Fv, loc, pos)
            
            for j, img in enumerate(data.imgTestList):
                if img == name:
                    imgDispList.append(j)
                    break
            else:
                raise ValueError('Requested test image for display not included in test dataset.')

    if numDisplay > data.x_test.shape[0]:
        numDisplay = data.x_test.shape[0]
    
    if not data.x_test.shape == data.resolution:
        data.x_test = data.x_test.reshape(data.nTest, data.resolution[0],data.resolution[1],data.resolution[2])
        model.predictions = model.predictions.reshape(data.nTest, data.resolution[0],data.resolution[1],data.resolution[2])

    plt.figure(figsize=(20, 5))
    for i in range(numDisplay):
        # Display original
        ax = plt.subplot(4, numDisplay, i + 1)
        plt.imshow(data.x_test[imgDispList[i]])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display code
        codeSizeSqrt = int(np.sqrt(model.nn.codeSize))
        ax = plt.subplot(4, numDisplay, i + 1 + numDisplay)
        plt.imshow(model.code[imgDispList[i]].reshape(codeSizeSqrt,codeSizeSqrt))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Draw active code pixels in code plot
        count = 0
        for x in range(codeSizeSqrt):
            for y in range(codeSizeSqrt):
                if count in model.activeCode:
                    plt.scatter(x,y, color='red',s=5)
                count += 1

        # Display reconstruction
        ax = plt.subplot(4, numDisplay, i + 1 + 2*numDisplay)
        plt.imshow(model.predictions[imgDispList[i]])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display error for each test image
        imageError = '{:.2}'.format(model.test_loss_per_image[imgDispList[i]])
        ax = plt.subplot(4, numDisplay, i + 1 + 3*numDisplay)
        ax.text(0.15,0.5,'mu1 = {}'.format(muDispList[0][i]))
        ax.text(0.15,0.3,'mu2 = {}'.format(muDispList[1][i]))
        ax.text(0.15,0.1,'image loss = {}'.format(imageError))
        ax.axis('off')
        if model.nn.verbose:
            plt.savefig('results/{}_prediction.png'.format(data.dataset))
    plt.show()

    if data.imgTestList:
        mu1_tot, mu2_tot = data.datasetClass.getMuDomain()
        # Plot training and test points
        data.datasetClass.plotMuDomain(mu1_tot, mu2_tot, data.paramTest[0], data.paramTest[1], model.nn.verbose)

        # Plot error for each test point
        fig, ax = plt.subplots()
        for i in range(len(model.test_loss_per_image)):
            imageError = '{:.2}'.format(model.test_loss_per_image[i])
            
            indMu1 = [k for k, x in enumerate(muDispList[0]) if x == data.paramTest[0][i]]
            if data.paramTest[1][i] in [muDispList[1][x] for x in indMu1]:
                colRef = 'green'
            else:
                colRef = 'red'

            ax.scatter(data.paramTest[0][i], data.paramTest[1][i], color=colRef)
            ax.text(data.paramTest[0][i], data.paramTest[1][i], imageError)
        ax.set_yticks(mu2_tot)
        plt.xlabel("mu_1 (position)")
        plt.ylabel("mu_2 (angle in º)")
        if model.nn.verbose:
            plt.savefig('results/{}_testError.png'.format(data.dataset))
        plt.show()


def summaryInfo(info, printInfo, saveInfo, name):
    df = pd.DataFrame(info, columns=['Parameter', 'Value'])
    if printInfo:
        print(df)
    if saveInfo:
        dfi.export(df, name)