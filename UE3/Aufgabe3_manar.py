import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('WebAgg')
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from numpy import linalg as LA

def covarianceMatrixLoop(X, mu):
    rows, cols = X.shape
    cov = np.zeros((cols, cols))
    for x in X:
        cov += np.outer(x-mu, x-mu)
    return cov / rows

class LinearRegression:
    def __init__(self):
        # Zugriff auf attributen durch Index
        self.classes = [[]for i in range(0,10)]
        self.eigenWerts = [np.array([])]* 10
        self.eigenVectors = [np.array([])]* 10
        self.vReduced = [np.array([])]* 10
        self.plt = plt
        self.colors = cm.rainbow(np.linspace(0, 1, 56))
        fig, self.axs = self.plt.subplots(10, 10, figsize=(18, 9),sharex=True,sharey=True )

    def train(self, xTrain, yTrain, dim):
        # Klassen ordnen
        for index in range(0, len(xTrain)):
            self.classes[int(yTrain[index])].append(xTrain[index])

        self.classes = np.array(self.classes)

        # Berechne m fuer die Klassen
        mues = [np.array([])]* 10
        for index in range(len(self.classes)):
            if(len(self.classes[index]) == 0): continue
            for vector in self.classes[index]:
                if(len(mues[index]) == 0): mues[index] = np.array(vector.copy())     ##Reference by value error!
                else: mues[index] += vector
            mues[index] = mues[index] / len(self.classes[index])
            cov = covarianceMatrixLoop(np.array(self.classes[index]), mues[index])
            w, v = LA.eig(cov)
            sortedIndices = np.argsort(w)[::-1]
            v = v[sortedIndices]
            w = w[sortedIndices]
            v = np.array(v)[:, :dim]
            w = np.array(w)[:dim]
            self.eigenWerts[index], self.eigenVectors[index] = w,v
            self.vReduced[index] = np.subtract( self.classes[index], mues[index] )
            newdata = np.dot(v.T, self.vReduced[index].T).T
            print(newdata)

    def plot2D(self, firstIndex, secondIndex):

        cov1 = np.dot(self.eigenVectors[firstIndex].T, self.vReduced[firstIndex].T).T
        cov2 = np.dot(self.eigenVectors[secondIndex].T, self.vReduced[secondIndex].T).T
        self.axs[firstIndex, secondIndex].scatter(cov1[:, 0], cov1[:, 1], s=1)
        self.axs[firstIndex, secondIndex].scatter(cov2[:, 0], cov2[:, 1], s=1)
        self.axs[firstIndex, secondIndex].set_title("("+ str(firstIndex) + ","+ str(secondIndex)+")")


#  self.plt.plot(cov2[0][0], cov2[1][0], color=color,  marker='o')
#        for i in range(0, 3): self.color[i] = round( 0.2+ self.color[i], 3)
#        if self.color[i] >= 1 : self.color = [0.2,0.2,0.2]
#        self.plt.axis([ min(cov1[0][0], cov2[0][0]) -5, max(cov1[0][0], cov2[0][0])+5, min(cov1[1][0], cov2[1][0])-5, max(cov1[1][0], cov2[1][0])+5])

    def showPlot(self):
        self.plt.tight_layout()
        self.plt.show()

def loadFromFile(path):
    df = pd.read_csv(path, header=None, sep=" ")
    X = df.iloc[:, 1:257].values # there is an empty string at position 257, because every line ends with a space (== separator)
    y = df.iloc[:, 0].values
    return X, y

xTrain, yTrain = loadFromFile("zip.train/zip.train")
x=0
LR = LinearRegression()
LR.train(xTrain, yTrain, 2)
for i in range (0, 10):
    for u in range (i, 10):
        LR.plot2D(i,u)
        x+=1
        print(x)

LR.showPlot()