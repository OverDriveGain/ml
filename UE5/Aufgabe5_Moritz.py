import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.image import imread


# Aufgabe 5
# abgabe von Moritz Walter und Manar Zaboub
def s(x):
    return 1 / (1 + np.exp(-x))

def sabl(x):
    return 0


class NeuralNet:

    allw = []

    def __init__(self, layer):
        #layer [40,50,...] länge = anzahl der hidden layer. nummer = anzahl der neuronen
        self.layer = layer


    def calcForward(self, input):
        print("calc forward")
        olist = []
        curro = input
        for w in self.allw:
            curro = stepForward(curro,w)
            olist.append(curro)

    def stepForward(self,o,w):
        odach = np.expand_dims(getOdach(o), axis=0)
        wdach = getWDach(w.T).T
        mult = np.dot(odach, wdach)
        return s(mult)[0]


    def backpropagate(self):
        print("backpropa")

    def inputData(self, data):
        self.data = data
        y,x = data.shape
        self._createInitialWs(x)

    def _createInitialWs(self, numberOfInputs):
        for i in range(len(self.layer)):
            if i == 0:
                self.allw.append(np.ones((numberOfInputs, self.layer[i])))
            else:
                self.allw.append(np.ones((self.layer[i - 1], self.layer[i])))

    def _getWDach(self, w):
        x = w.shape
        return np.insert(w, x[1], 1, axis=1)

    def _getOdach(self, o):
        #return as zeilenvektor
        x = o.shape
        if x[0] == 1:
            return np.insert(o, x[1], 1)
        else:
            oo = o.T
            return np.insert(oo, x, 1)


def createInitialWs(numberOfInputs, layer):

            return np.zeros((numberOfInputs, layer))


def getOdach(o):
        #return as zeilenvektor
        x = o.shape
        if x[0] == 1:
            return np.insert(o, x[1], 1)
        else:
            oo = o.T
            return np.insert(oo, x, 1)

def getWDach(w):
    x = w.shape
    einsen = np.ones(x[1])
    #print(w.shape)
    #print(einsen.shape)
    return  np.insert(w, x[1], 1, axis=1)# np.vstack((w,einsen))

def load_from_file(path):
    # importfunktion für daten. Übernommen aus Tutoriumsvorlage
    df = pd.read_csv(path, header=None, sep=" ")
    x = df.iloc[:, 1:257].values
    # there is an empty string at position 257, because every line ends with a space (== separator)
    y = df.iloc[:, 0].values
    return x, y


def separate_data(x, y, digit):
    x_digit = x[y == digit]
    return x_digit


def project_data(vectors, data, mu):
    return np.dot(data - mu, vectors)


def stepForward(o,w):
    odach = np.expand_dims(getOdach(o), axis=0)

    wdach = getWDach(w.T).T
    #print(wdach)
   # print(odach)

    mult = np.dot(odach,wdach)
    #print(mult)
    return s(mult)

def tests():
    v = np.array([(0, 0, 1, 0, 1, 4, 5, 0, 1, 4, 5),
                (0, 0, 0, 0, 1, 4, 5, 0, 1, 4, 5),
                (1, 0, 0, 0, 1, 4, 5, 0, 1, 4, 5)])

    v2 = np.array([(0, 0, 1, 4, 5, 0, 1, 4, 5),
                  (0, 0, 0, 5 ,6, 0, 1, 4, 5),
                  (1, 0, 0, 6, 7, 0, 1, 4, 5)])

    model =NeuralNet([7,3])

    #nächste schritt soll 4 neuronen haben haben
   # print(v[0].shape[0])
    #w1 = createInitialWs(v[0].shape[0],4)
   # w2 = createInitialWs(v2[0].shape[0],4)

   # o1 = v[0]
   # o2 = v2[0]

    #print(stepForward(o1,w1))
   # print(stepForward(o2, w2))


    #print(v)
    #print(w1)
    #print(v2)
    #print(w2)

   # for i in v:
   #     print(getOdach(i))

    #for i in v2:
    #    print(getOdach(i))
    #print(v)

    #print(getWDach(v))
    #print("v2")
    #print(v2)
    #print(getWDach(v2))

def aufgabe1():
    x_train, y_train = load_from_file("../UE3/zip.train/zip.train")
    x_test, y_test = load_from_file("../UE3/zip.test/zip.test")


def main():
    tests()
    #aufgabe1()

if __name__ == "__main__":
    main()
