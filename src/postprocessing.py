import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dataframe_image as dfi


def summaryInfo(info, printInfo, saveInfo, name):
    df = pd.DataFrame(info, columns=['Parameter', 'Value'])
    if printInfo:
        print(df)
    if saveInfo:
        dfi.export(df, name)

def plotImage(data, nRows, numDisplay, count):
    ax = plt.subplot(nRows, numDisplay, count)
    plt.imshow(data)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def addPlotNames(plotNames):
    for i, plotName in enumerate(reversed(plotNames)):
        plt.text(0.1, 0.12+0.23*i, plotName, fontsize=12, transform=plt.gcf().transFigure, rotation=90)

def addPlotNames_param(plotNames):
    for i, plotName in enumerate(reversed(plotNames)):
        plt.text(0.1, 0.12+0.14*i, plotName, fontsize=12, transform=plt.gcf().transFigure, rotation=90)

def plotActiveCode(codeSizeSqrt, activeCode):
    count = 0
    for y in range(codeSizeSqrt):
        for x in range(codeSizeSqrt):  
            if count in activeCode:
                plt.scatter(x,y, color='red',s=5)
            count += 1

def plottingPrediction(data, model, kw):
    imgDisplay = kw['IMG_DISPLAY']
    codeSizeSqrt = int(np.sqrt(model.nn.model.codeSize))

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
        if data.arch == 'param_ae':
            model.x_ED = model.predictions['x_ED'].reshape(data.nTest, data.resolution[0], data.resolution[1], data.resolution[2])
            model.code_E = model.predictions['code_E'].reshape(data.nTest, codeSizeSqrt, codeSizeSqrt)
            model.code_P = model.predictions['code_P'].reshape(data.nTest, codeSizeSqrt, codeSizeSqrt)
            model.x_PD = model.predictions['x_PD'].reshape(data.nTest, data.resolution[0],data.resolution[1],data.resolution[2])
            nRows = 6
            plt.figure(figsize=(20, 8))
            plotNames = ['X', 'X_EP', 'code_E', 'code_P', 'X_PD', 'fig data']
            for i in range(numDisplay):
                # Display NN outputs
                plotImage(data.x_test[imgDispList[i]], nRows, numDisplay, i+1)
                plotImage(model.x_ED[imgDispList[i]], nRows, numDisplay, i+1+numDisplay)
                plotImage(model.code_E[imgDispList[i]], nRows, numDisplay, i+1+2*numDisplay)
                plotActiveCode(codeSizeSqrt, model.activeCode_E)
                plotImage(model.code_P[imgDispList[i]], nRows, numDisplay, i+1+3*numDisplay)
                plotActiveCode(codeSizeSqrt, model.activeCode_P)
                plotImage(model.x_PD[imgDispList[i]], nRows, numDisplay, i+1+4*numDisplay)

                # Display error for each test image
                total_loss = '{:.2}'.format(model.test_loss_per_image[imgDispList[i]][0])
                x_ED_loss = '{:.2}'.format(model.test_loss_per_image[imgDispList[i]][1])
                code_loss = '{:.2}'.format(model.test_loss_per_image[imgDispList[i]][2])
                x_PD_loss = '{:.2}'.format(model.test_loss_per_image[imgDispList[i]][3])
                ax = plt.subplot(nRows, numDisplay, i + 1 + 5*numDisplay)
                ax.text(0.15,0.9,'mu1 = {}'.format(muDispList[0][i]))
                ax.text(0.15,0.7,'mu2 = {}'.format(muDispList[1][i]))
                ax.text(0.15,0.5,'L_total = {}'.format(total_loss))
                ax.text(0.15,0.3,'L_image ED = {}'.format(x_ED_loss))
                ax.text(0.15,0.1,'L_code = {}'.format(code_loss))
                ax.text(0.15,-0.1,'L_image_PD = {}'.format(x_PD_loss))
                ax.axis('off')
            addPlotNames_param(plotNames)

        else:
            plotNames = ['X', 'code', 'X_ED', 'fig data']
            model.predictions = model.predictions.reshape(data.nTest, data.resolution[0],data.resolution[1],data.resolution[2])
            model.code = model.code.reshape(data.nTest, codeSizeSqrt, codeSizeSqrt)
            nRows = 4
            plt.figure(figsize=(20, 5))
            for i in range(numDisplay):
                # Display NN outputs
                plotImage(data.x_test[imgDispList[i]], nRows, numDisplay, i+1)
                plotImage(model.code[imgDispList[i]], nRows, numDisplay, i+1+numDisplay)
                plotActiveCode(codeSizeSqrt, model.activeCode)
                plotImage(model.predictions[imgDispList[i]], nRows, numDisplay, i+1+2*numDisplay)

                # Display error for each test image
                imageError = '{:.2}'.format(model.test_loss_per_image[imgDispList[i]])
                ax = plt.subplot(4, numDisplay, i + 1 + 3*numDisplay)
                ax.text(0.15,0.5,'mu1 = {}'.format(muDispList[0][i]))
                ax.text(0.15,0.3,'mu2 = {}'.format(muDispList[1][i]))
                ax.text(0.15,0.1,'loss = {}'.format(imageError))
                ax.axis('off')
            addPlotNames(plotNames)

        if model.nn.model.verbose:
            plt.savefig('results/{}_prediction.png'.format(data.dataset))
        plt.show()

        if data.imgTestList:
            mu1_tot, mu2_tot = data.datasetClass.getMuDomain()
            # Plot training and test points
            data.datasetClass.plotMuDomain(mu1_tot, mu2_tot, data.paramTest[0], data.paramTest[1], model.nn.model.verbose)

            # Plot error for each test point
            fig, ax = plt.subplots()
            for i in range(len(model.test_loss_per_image)):
                if data.arch == 'param_ae':
                    imageError = '{:.2}'.format(model.test_loss_per_image[i][0])
                else:
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
            if model.nn.model.verbose:
                plt.savefig('results/{}_testError.png'.format(data.dataset))
            plt.show()

def plotTraining(train, trainTime, dataset):
    leg = []
    for key in train.history:
        plt.plot(train.epoch, train.history[key])
        leg.append(key)
    plt.title('Model loss. Training time = {:.2} min'.format(trainTime/60))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(leg, loc='upper right')
    ax = plt.gca()
    ax.set_ylim([0, 0.5])
    plt.savefig('results/{}_training.png'.format(dataset))
    plt.show()

def getLosses(train, test_loss, dataset, verbose, saveInfo):
    best_epoch = np.argmin(train.history['loss'])
    data = []
    for key in train.history:
        data.append([key, '{:.2}'.format(train.history[key][best_epoch])])
    if isinstance(test_loss, list):
        for i in range(int(len(train.history)/2)):
            data.append(['test_{}'.format(list(train.history.keys())[i]), '{:.2}'.format(test_loss[i])])
    else:
        data.append(['test_loss', '{:.2}'.format(test_loss)])
  
    name = 'results/losses_{}.png'.format(dataset)
    summaryInfo(data, verbose, saveInfo, name)
    return data