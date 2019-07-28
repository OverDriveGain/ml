import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('WebAgg')
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from numpy import linalg as LA

class LinearRegression:
    def __init__(self):
        # Zugriff auf attributen durch Index
        self.classes = [[]for i in range(0,10)]
        self.eigenWerts = [np.array([])]* 10
        self.eigenVectors = [np.array([])]* 10
        self.plt = plt
        self.colors = cm.rainbow(np.linspace(0, 1, 56))

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

        for index in range(0, len(self.classes)):
            if(len(self.classes[index]) == 0): continue
            sigma = []
            for num, vector in enumerate(self.classes[index]):
                sub = vector - mues[index]
                if(sigma ==[]):
                    sigma = np.subtract(vector, mues[index]) * np.subtract(vector, mues[index])[np.newaxis].T
                else:
                    sigma += np.subtract(vector, mues[index]) * np.subtract(vector, mues[index])[np.newaxis].T
            sigma = sigma / len(self.classes[index])
            w, v = LA.eig(sigma)
            sortedIndices = np.argsort(w)[::-1]
            v = v[sortedIndices]
            w = w[sortedIndices]
            v = np.array(v)[:dim, :dim]
            w = np.array(w)[:dim]
            self.eigenWerts[index], self.eigenVectors[index] = w,v

    def plot2D(self, firstIndex, secondIndex):
        cov1 = np.dot(np.diag(self.eigenWerts[firstIndex]),np.dot(self.eigenVectors[firstIndex],self.eigenWerts[firstIndex][np.newaxis].T))
        cov2 = np.dot(np.diag(self.eigenWerts[secondIndex]),np.dot(self.eigenVectors[secondIndex],self.eigenWerts[secondIndex][np.newaxis].T))
#        h = figure(‘Color’, [0.2 0.2 0.2])
        color = self.colors[-1]
        self.colors = self.colors[:-1]
        self.plt.plot(cov1[0][0], cov1[1][0], color=color,  marker='o')
        self.plt.plot(cov2[0][0], cov2[1][0], color=color,  marker='o')
#        for i in range(0, 3): self.color[i] = round( 0.2+ self.color[i], 3)
#        if self.color[i] >= 1 : self.color = [0.2,0.2,0.2]
#        self.plt.axis([ min(cov1[0][0], cov2[0][0]) -5, max(cov1[0][0], cov2[0][0])+5, min(cov1[1][0], cov2[1][0])-5, max(cov1[1][0], cov2[1][0])+5])

    def showPlot(self):
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