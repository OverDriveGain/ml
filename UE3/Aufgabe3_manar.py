import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('WebAgg')
import pandas as pd
import numpy as np
from numpy import linalg as LA

class LinearRegression:
    def __init__(self):
        # Zugriff auf attributen durch Index
        self.classes = [[]for i in range(0,10)]
        self.eigenWerts = [np.array([])]* 10
        self.eigenVectors = [np.array([])]* 10
        self.mues = [np.array([])]* 10
    def train(self, xTrain, yTrain, dim):
        # Klassen ordnen
        for index in range(0, len(xTrain)):
            self.classes[int(yTrain[index])].append(xTrain[index])

        self.classes = np.array(self.classes)
        # Berechne m fuer die Klassen
        for index in range(len(self.classes)):
            if(len(self.classes[index]) == 0): continue
            for vector in self.classes[index]:
                if(len(self.mues[index]) == 0): self.mues[index] = np.array(vector.copy())     ##Reference by value error!
                else: self.mues[index] += vector
            self.mues[index] = self.mues[index] / len(self.classes[index])

        sigmas = []
        for index in range(0, len(self.classes)):
            if(len(self.classes[index]) == 0): continue
            sigma = []
            for num, vector in enumerate(self.classes[index]):
                sub = vector - self.mues[index]
                if(sigma ==[]):
                    sigma = np.subtract(vector, self.mues[index]) * np.subtract(vector, self.mues[index])[np.newaxis].T
                else:
                    sigma += np.subtract(vector, self.mues[index]) * np.subtract(vector, self.mues[index])[np.newaxis].T
            sigma = sigma / len(self.classes[index])
            w, v = LA.eig(sigma)
            sortedIndices = np.argsort(w)
            v = v[sortedIndices]
            w = w[sortedIndices]
            v = np.array(v)[:dim, :dim]
            w = np.array(w)[:dim]
            self.eigenWerts[index], self.eigenVectors[index] = w,v

    def plot2D(self, firstIndex, secondIndex):
        cov1 = np.dot(np.diag(self.eigenWerts[firstIndex]),np.dot(self.eigenVectors[firstIndex],self.eigenWerts[firstIndex][np.newaxis].T))
        cov2 = np.dot(np.diag(self.eigenWerts[secondIndex]),np.dot(self.eigenVectors[secondIndex],self.eigenWerts[secondIndex][np.newaxis].T))
        print(cov1)
        print(cov2)
        plt.plot(cov1[0][0], cov1[1][0], 'ro')
        plt.plot(cov2[0][0], cov2[1][0], 'ro')
        plt.axis([ min(cov1[0][0], cov2[0][0]) -5, max(cov1[0][0], cov2[0][0])+5, min(cov1[1][0], cov2[1][0])-5, max(cov1[1][0], cov2[1][0])+5])
        plt.show()

def loadFromFile(path):
    df = pd.read_csv(path, header=None, sep=" ")
    X = df.iloc[:, 1:257].values # there is an empty string at position 257, because every line ends with a space (== separator)
    y = df.iloc[:, 0].values
    return X, y

xTrain, yTrain = loadFromFile("zip.train/zip.train")

LR = LinearRegression()
LR.train(xTrain, yTrain, 2)
LR.plot2D(0,2)