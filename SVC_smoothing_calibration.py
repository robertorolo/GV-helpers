import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def class_max_percentage_change(y, s):
    ypercentages = {}
    unique_classes = np.unique(y)
    for i in unique_classes:
        ypercentages[i] = np.sum(y==i)/len(y) * 100

    spercentages = {}
    unique_classes = np.unique(s)
    for i in unique_classes:
        spercentages[i] = np.sum(s==i)/len(s) * 100

    perc_change = []
    for i in ypercentages:
        p = ypercentages[i]
        if i not in spercentages:
            perc_change.append(999)
        else:
            pf = spercentages[i]
            perc_change.append(np.abs(p-pf))
    
    return np.max(perc_change)

def svc_smoothing_calibration(X, Y, Z, y, outfl):
    Crange = [10**i for i in range(-3, 3)]
    GammaRange = [10**i for i in range(-7, -1)]

    matshape = (len(Crange), len(GammaRange))
    accmat = np.zeros(matshape)
    changemat = np.zeros((len(Crange), len(GammaRange)))

    X = np.array([X, Y, Z]).T
    
    for idxi, i in enumerate(Crange):
        for idxj, j in enumerate(GammaRange):
            smooth_model = SVC(kernel='rbf', C=i, gamma=j)

            smooth_model.fit(X, y)

            s = smooth_model.predict(X)

            acc = accuracy_score(y, s)
            accmat[idxi, idxj] = round(acc, 2)

            perchange = class_max_percentage_change(y, s)
            changemat[idxi, idxj] = round(perchange, 2)

    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].imshow(accmat)
    axs[1].imshow(changemat)

    axs[0].set_xticks(np.arange(len(Crange)))
    axs[0].set_xticklabels(Crange, rotation=45)
    axs[0].set_yticks(np.arange(len(GammaRange)))
    axs[0].set_yticklabels(GammaRange)
    axs[0].set_xlabel('C')
    axs[0].set_ylabel('Gamma')


    axs[1].set_xticks(np.arange(len(Crange)))
    axs[1].set_xticklabels(Crange, rotation=45)
    axs[1].set_yticks(np.arange(len(GammaRange)))
    axs[1].set_yticklabels(GammaRange)
    axs[1].set_xlabel('C')
    axs[1].set_ylabel('Gamma')

    for idxi, i in enumerate(Crange):
        for idxj, j in enumerate(GammaRange):
            axs[0].text(idxi, idxj, accmat[idxi, idxj], ha="center", va="center", color="w")
            axs[1].text(idxi, idxj, changemat[idxi, idxj], ha="center", va="center", color="w")

    axs[0].set_title('Accuracy')
    axs[1].set_title('Maximum percent change')

    plt.tight_layout()
    plt.savefig(outfl, bbox_inches='tight', transparent=False)