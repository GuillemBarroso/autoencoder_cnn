import matplotlib.pyplot as plt
import numpy as np
import math

def getParamFromStr(name):
    firstUnderscore = name.find('_')
    secondUnderscore = name.find('_',round(len(name)/2))
    Fh = name[2:firstUnderscore]
    Fv = name[firstUnderscore+3:secondUnderscore]
    loc = name[secondUnderscore+1:secondUnderscore+2]
    pos = name[secondUnderscore+2:-4]
    return Fh, Fv, loc, pos

def getMus(Fh, Fv, loc, pos):
    pos = round(float(pos),2)
    if loc == 'B':
        mu1 = pos
    elif loc == 'R':
        mu1 = 1 - pos + 1
    elif loc == 'T':
        mu1 = 2 + 1 - pos - 0.1
    
    Fh = float(Fh)
    Fv = float(Fv)

    h = np.sqrt(Fh**2 + Fv**2)
    rads = math.asin(Fh/h)
    mu2 = round(rads*180/np.pi, 1)

    return mu1, mu2

def getParamsFromMus(mu1,mu2):
    if mu2 == 0.0:
        Fh = 0
        Fv = 1
    elif mu2 == 22.5:
        Fh = 0.382683
        Fv = 0.92388
    elif mu2 == 45.0:
        Fh = 0.707107
        Fv = 0.707107
    elif mu2 == 67.5:
        Fh = 0.92388
        Fv = 0.382683
    elif mu2 == 90.0:
        Fh = 1
        Fv = 0
    elif mu2 == 112.5:
        Fh = 0.92388
        Fv = -0.382683
    elif mu2 == 135.0:
        Fh = 0.707107
        Fv = -0.707107
    elif mu2 == 157.5:
        Fh = 0.382683
        Fv = -0.92388
    elif mu2 == 180.0:
        Fh = 0
        Fv = -1

    if mu1 < 1.0:
        loc = 'B'
        pos = mu1
    elif 1.0 <= mu1 < 2.0:
        loc = 'R'
        pos = mu1 - 1
    elif 2.0 <= mu1:
        loc = 'T'
        pos = 1 - (mu1 - 2)

    pos = round(pos, 2)
    if pos == 0.0:
        pos = 0

    return Fh, Fv, loc, pos
    

if __name__ == '__main__':

    # Define test images
    # mu2_test = [45, 67.5, 90, 112.5, 135]
    # mu2_test = [22.5]
    mu2_test = [round(x,2) for x in np.arange(0, 202.5, 22.5)]
    mu1_test = [2.5, 2.55, 2.6, 2.65, 2.7]
    # mu1_test = [2.1, 2.15, 2.2, 2.25, 2.3, 2.35]
    # mu1_test = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
    # mu1_test = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]

    mu1_ext = []
    mu2_ext = []
    testData = []
    for mu1 in mu1_test:
        for mu2 in mu2_test:
            mu1_ext.append(mu1)
            mu2_ext.append(mu2)
            Fh, Fv, loc, pos = getParamsFromMus(mu1,mu2)
            name = 'Fh{}_Fv{}_{}{}.txt'.format(Fh, Fv, loc, pos)
            testData.append(name)

    # Define the parameter's domain
    mu2 = [round(x,2) for x in np.arange(0, 202.5, 22.5)]
    muBot = [round(x,2) for x in np.arange(0.3, 0.95, 0.05)]
    nBot = len(muBot)
    muRight = [round(x,2) for x in np.arange(0, 0.95, 0.05)]
    nRight = len(muRight)
    muTop = muBot
    nTop = nBot

    mu1 = list(muBot)
    for locRight in muRight:
        mu1.append(locRight+1)
    
    for locTop in reversed(muTop):
        mu1.append(3 - locTop)

    nMu1 = len(mu1)
    nMu2 = len(mu2)

    # Plot points for training and testing
    fig, ax = plt.subplots()
    for i in range(nMu1):
        for j in range(nMu2):
                if mu1[i] in mu1_ext:
                    indices = [k for k, x in enumerate(mu1_ext) if x == mu1[i]]
                    mu2aux = [mu2_ext[q] for q in indices]
                    if mu2[j] in mu2aux:
                        ax.scatter(mu1[i], mu2[j], color='red')
                    else:
                        ax.scatter(mu1[i], mu2[j], color='blue')
                else:
                    ax.scatter(mu1[i], mu2[j], color='blue')
    plt.xlabel("mu_1 (position)")
    plt.ylabel("mu_2 (angle in º)")
    plt.show()
    
    print(testData)
    print('dataset size = {}'.format(len(testData)))