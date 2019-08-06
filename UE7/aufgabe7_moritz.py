import pandas as pd
import numpy as np


# id 3 algo https://data-science-blog.com/blog/2017/12/08/id3-algorithmus-ein-rechenbeispiel/


def load_from_file(path):
    #importfunktion für daten. Übernommen aus Tutoriumsvorlage
    df = pd.read_csv(path, header=None, sep=",")
    X = df.iloc[:, 0:4].values
   # X = df.iloc[:, 0:7].values # there is an empty string at position 257, because every line ends with a space (== separator)
    y = df.iloc[:, 4:5].values

    #for i in X:



    return X, y.T[0]


class ID3Tree:

    def __init__(self, data, labels, descript = None, dept=0):
        self.isLeaf = False
        self.predictValue = False
        self.data = data
        self.tiefe = dept
        self.labels = labels
        self.descriptor = descript
        self.n = len(data)
        self.pos = self.countPositivs()
        self.gesamtEntro = self.entropy(self.pos, self.n - self.pos)
        self.blaetter = []



        if self.gesamtEntro > 0:
            self.splittedAttribute, self.entropies = self.chooseAttribute(self.data,self.labels)
            #print("Knoten mit Attribute " + str(self.splittedAttribute))

            for i in self.entropies:
                newdata, newlabel = self.dataSplit(i[0])
                knoten = ID3Tree(newdata, newlabel, i[0], self.tiefe + 1)
                self.blaetter.append(knoten)
        else:
            self.isLeaf = True
            self.predictValue = labels[0]

    def getDescriptor(self):
        return self.descriptor

    def predict(self, x):
        if self.isLeaf:
            return self.predictValue
        else:
            for blatt in self.blaetter:
                if blatt.descriptor == x[self.splittedAttribute]:
                    return blatt.predict(x)
        return None
        # TODO: else


    def entropy(self, pos, neg):
        # p liste mit der anzahl der einzelnen vorkommnisse der verschiedenen werte einer Variablen
        p = pos / (pos + neg)
        if p >= 1.0 or p <= 0.0:
            return 0
        else:
            return - p * np.log2(p) - (1 - p) * np.log2(1 - p)

    def infogewinn(self, tupels):
        sum = self.gesamtEntro
        for i in tupels:
            sum = sum - (((i[1] + i[2]) / self.n) * i[3])
        return sum

    def count(self, X, y, i):
        # X data, i attribute number
        sums = []
        found = []
        for j in range(len(X)):
            x = X[j]  # vektor
            l = y[j]  # label
            if x[i] in found:
                for z in range(len(sums)):
                    if sums[z][0] == x[i]:
                        temp = sums[z]
                        if bool(l):
                            temp = (temp[0], temp[1] + 1, temp[2])
                        else:
                            temp = (temp[0], temp[1], temp[2] + 1)
                        sums[z] = temp
            else:
                found.append(x[i])
                if bool(l):
                    sums.append((x[i], 1, 0))
                else:
                    sums.append((x[i], 0, 1))
        return sums

    def countPositivs(self):
        sum = 0
        for i in self.labels:
            if bool(i):
                sum += 1

        return sum

    def chooseAttribute(self, Data, Labels):
        N = len(Data)

        Entropies = []
        for var in range(len(Data[0])):
            c = self.count(Data, Labels, var)
            entro = []
            for i in c:
                e = self.entropy(i[1], i[2])
                entro.append((i[0], i[1], i[2], e))
            Entropies.append(entro)

        gains = []

        for e in Entropies:
            gains.append(self.infogewinn(e))

        maxi = np.max(gains)
        for i in range(len(gains)):
            if gains[i] == maxi:
                #print(Entropies)
                #print(gains)
                return i, Entropies[i]

    def dataSplit(self, Select):
        newData = []
        newLabel = []
        for i in range(self.n):
            d = self.data[i]
            if d[self.splittedAttribute] == Select:
                newData.append(d)
                newLabel.append(self.labels[i])
        return np.array(newData), newLabel


def main():
    X_train, y = load_from_file("test.data")

    knoten = ID3Tree(X_train, y)

   # for i in range(len(X_train)):
   #     print(knoten.predict(X_train[i]) == y[i])


    #print(entro)
    #print(infogewinn(0.94,len(X_train),entro))
   # print(X_train[0])
  #  X_test, y_test = load_from_file("zip.test/zip.test")

if __name__ == "__main__":
    main()