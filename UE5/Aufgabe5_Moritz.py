import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.image import imread


# Aufgabe 5
# abgabe von Moritz Walter und Manar Zaboub
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)* (1-sigmoid(x))



class NNode:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.w = np.ones((self.input_size, self.output_size))*0.001
        self.delta = np.ones((self.input_size, self.output_size))#*0.001

        self.output = np.zeros((self.output_size)).T
        self.input = np.zeros((self.input_size))
        self.d = np.zeros((self.output.shape[0], self.output.shape[0]))

        self.odach = self._getOdach(self.input)
        self.wdach = self._getWDach(self.w.T)


    def _getWDach(self, w):
        x = w.shape
        return np.insert(w, x[1], 1, axis=1)


    def getW(self):
        return np.delete(self.wdach, self.wdach.shape[1]-1, axis=1)

    def _getOdach(self, o):
        #return as zeilenvektor
        x = o.shape
        if x[0] == 1:
            return np.insert(o, x[1], 1)
        else:
            oo = o.T
            return np.insert(oo, x, 1)

    def calculate(self, input):
        self.input = input
        self.odach = self._getOdach(input)
        y = np.dot(self.odach,self.wdach.T)
        self.output = self.sigmoid(y)

    def back(self):

        print("back")

    def calcD(self):#, o2, o1):
        o2 = self.output #self.nodes[i].output
        o1 = self.input#self.nodes[i - 1].output

        # create d matrix
        self.d = np.zeros((o2.shape[0], o2.shape[0]))
        for j in range(len(o2)):
            oi2 = o2[j]
            self.d[j, j] = oi2 * (1 - oi2)


    def updateW(self, wdelta):
        self.w += wdelta

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return sigmoid(x) * (1 - sigmoid(x))


class NeuralNet:

    nodes = []
    y_predict = []

    def __init__(self, data, labels, layers):
        #layer [40,50,...] länge = anzahl der hidden layer. nummer = anzahl der neuronen pro layer. last layer = output layer, input layer kommt von daten
        self.data = data
        self.layer = layers
        self.y = labels
        input_length = data[0].shape[0]


        # create nodes
        for i in range(len(layers)):
            if i == 0:
                self.nodes.append(NNode(input_length,layers[i]))
            else:
                self.nodes.append(NNode(layers[i-1], layers[i]))

        #for i in self.nodes:
           # print("Shape node: (" + str(i.input_size)+"," + str(i.output_size) + ")")

    def predict(self, x):
        self.calcForward(x)
        return self.nodes[len(self.nodes) - 1].output

    def fit(self):
        for i in range(len(self.data)):
            self.calcForward(self.data[i])
            self.backpropagation(self.y[i],self.data[i])

    def calcForward(self, input):
        for i in range(len(self.nodes)):
            #print("calc node " + str(i))
            if i == 0:
                self.nodes[i].calculate(input)
            else:
               # print("output shape" + str(self.nodes[i-1].output.shape))
                self.nodes[i].calculate(self.nodes[i-1].output)


    def backpropagation(self, y, input):
        #print(y)
        deltas = []
        l = len(self.nodes)
        for i in range(l-1,-1,-1):
            currNode = self.nodes[i]
            print(currNode.wdach)
            print()

            currNode.calcD
            t = np.zeros(currNode.output.shape[0])
            t[y] = 1
            e = np.zeros(currNode.output.shape[0])
            for k in range(len(currNode.output)):
                e[k] = currNode.output[k]*currNode.output[k]*t[k]

           # print(e)

            if (l-1) == i:
               #deltas.append(np.dot(currNode.d,e))
               currNode.delta = np.dot(currNode.d, e)
            elif i >= 0:# cas == l
                nextNode = self.nodes[i + 1]
                currNode.delta = np.dot(np.dot(currNode.d,nextNode.getW),nextNode.delta)


            # calculate delta to add
            delta = 0


            # else:
            #     onplus1 = self.nodes[i]
            #     dw = np.zeros((onplus1.output.shape[0], onplus1.output.shape[0]))
            #     print(dw.shape)
                #dw = np.dot(on.output.T, ( (y - onplus1.output) * sigmoid_deriv(onplus1.output)))


    # def stepBack(self,on, onplus1, y):
    #     # o output
    #     # y desired result
    #
    #     d_weights2 = np.dot(on.T, (2 * (self.y - onplus1) * sigmoid_deriv(onplus1)))
    #     d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - onplus1) * sigmoid_deriv(onplus1), self.weights2.T) * sigmoid_deriv(on)))




def tests():
    v = np.array([(0, 0, 1, 0, 1, 4, 5, 0, 1, 4, 5),
                (0, 0, 0, 0, 1, 4, 5, 0, 1, 4, 5),
                (1, 0, 0, 0, 1, 4, 5, 0, 1, 4, 5)])

    v2 = np.array([(0, 0, 1, 4, 5, 0, 1, 4, 5),
                  (0, 0, 0, 5 ,6, 0, 1, 4, 5),
                  (1, 0, 0, 6, 7, 0, 1, 4, 5)])

    model = NeuralNet(v,np.array((1,1,0)),[8,7,5,2])

    model.fit()

    #model.calcForward(v[0])
    #model.backpropagation()

    #model = NNode(4,9)
    #model.calculate(np.ones(4))

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
    return sigmoid(mult)



def aufgabe1():
    x_train, y_train = load_from_file("../UE3/zip.train/zip.train")
    x_test, y_test = load_from_file("../UE3/zip.test/zip.test")


def main():
    tests()
    #aufgabe1()

if __name__ == "__main__":
    main()
