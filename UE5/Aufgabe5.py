import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Aufgabe 5
# abgabe von Moritz Walter und Manar Zaboub

class NNode:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        w = np.ones((self.input_size, self.output_size))*0.001
        self.delta = np.zeros((self.input_size, self.output_size))

        self.output = np.zeros((self.output_size)).T
        self.input = np.zeros((self.input_size))
        self.d = np.zeros((self.output.shape[0], self.output.shape[0]))

        self.odach = self._getOdach(self.input)
        self.wdach = self._getWDach(w.T)

        self.deltaWdach = np.zeros((self.input_size+1, self.output_size))

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

    def calcD(self):
        o2 = self.output
        o1 = self.input

        # create d matrix
        self.d = np.zeros((o2.shape[0], o2.shape[0]))
        for j in range(len(o2)):
            oi2 = o2[j]
            self.d[j, j] = self.sigmoid_deriv(oi2) * (1 - self.sigmoid_deriv(oi2))

    def updateW(self):
        self.wdach = np.subtract(self.wdach, self.deltaWdach)
        self.deltaWdach = np.zeros((self.input_size+1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


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

    def predict(self, x):
        self.calcForward(x)
        pred = np.around((self.nodes[len(self.nodes) - 1].output))

        for i in range(len(pred)):
            if pred[i] == 1:
                return i
        return -1

    def fit(self):
        for i in range(len(self.data)):
            self.calcForward(self.data[i])
            self.backpropagation(self.y[i])

    def calcForward(self, input):
        for i in range(len(self.nodes)):
            if i == 0:
                self.nodes[i].calculate(input)
            else:
                self.nodes[i].calculate(self.nodes[i-1].output)


    def backpropagation(self, y):
        l = len(self.nodes)
        for i in range(l-1,-1,-1):
            currNode = self.nodes[i]
            currNode.calcD()
            t = np.zeros(currNode.output.shape[0])
            t[int(y)] = 1
            e = np.zeros(currNode.output.shape[0])
            for k in range(len(currNode.output)):
                e[k] = currNode.output[k]*currNode.output[k]-t[k]

            if (l-1) == i:
               currNode.delta = np.dot(currNode.d, e)
            elif i >= 0:# cas == l
                currNode.delta = np.dot(currNode.d, e)

            # calculate delta to add
            dwdach = np.dot(np.array(([currNode.delta])).T , np.array(([currNode.odach])))
            currNode.deltaWdach = dwdach
        # nachdem alle deltas berechnet, deltas anwenden
        for i in self.nodes:
            i.updateW()


def teste(model, x_test, y_test):
    gefunden = np.array([(0,0,1,2,3,4,5,6,7,8,9),
                         (0,0,0,0,0,0,0,0,0,0,0),
                         (1,0,0,0,0,0,0,0,0,0,0),
                         (2,0,0,0,0,0,0,0,0,0,0),
                         (3,0,0,0,0,0,0,0,0,0,0),
                         (4,0,0,0,0,0,0,0,0,0,0),
                         (5,0,0,0,0,0,0,0,0,0,0),
                         (6,0,0,0,0,0,0,0,0,0,0),
                         (7,0,0,0,0,0,0,0,0,0,0),
                         (8,0,0,0,0,0,0,0,0,0,0),
                         (9,0,0,0,0,0,0,0,0,0,0)])

    richtig = 0
    falsch = 0
    for i in range(len(x_test)):
        predicted = model.predict(x_test[i])
        currLabel = int(y_test[i])
        gefunden[1 + predicted, 1 + currLabel] = gefunden[1 + predicted, 1 + currLabel] + 1
        if predicted == currLabel:
            richtig += 1
        else:
            falsch += 1
    return gefunden , richtig, falsch


def printKonfu(model, x_test, y_test):
    a, richtig, falsch = teste(model, x_test, y_test)
    print(a)
    print("Fehlerquote: " + str(falsch / (richtig + falsch) * 100) + " %")
    print()


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

def aufgabe1():
    x_train, y_train = load_from_file("../UE3/zip.train/zip.train")
    x_test, y_test = load_from_file("../UE3/zip.test/zip.test")
    neuronen = [80,10]
    epochen = 50
    model = NeuralNet(x_train, y_train, neuronen)

    iter = []
    genau = []

    for i in range(epochen):
        model.fit()
        a, richtig, falsch = teste(model, x_test, y_test)
        iter.append(i)
        genau.append(1 - (falsch / (richtig + falsch)))
        if i % 1 == 0:
            print("Konfusionsmatrix bei Epoche =" + str(i))
            printKonfu(model, x_test, y_test)

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    axs.plot(iter, genau, c="blue")
    axs.set_title("Mit Neuronen " + str(neuronen))
    plt.show()

def main():
    aufgabe1()

if __name__ == "__main__":
    main()
