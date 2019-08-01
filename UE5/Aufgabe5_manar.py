import pandas as pd
import numpy as np

def loadDigitsFromFile(path):
    df = pd.read_csv(path, header=None, sep=" ")
    X = df.iloc[:, 1:257].values # there is an empty string at position 257, because every line ends with a space (== separator)
    y = df.iloc[:, 0].values
   # for i in range(9): return X[y == i]
    return np.array([ X[y == i] for i in range(10)])


class NeuralNetwork:

    def __printws(self):
        for i in self.w:
            print (i.shape)

    def __init__(self, attrcnt, classescnt, verbose = True, *wselementscnt):
        self.w = []
        self.w.append( np.full( (attrcnt+1, wselementscnt[0]), 0.1 ) )
        for wElementsCnt in wselementscnt[1:]:
            self.w.append(np.full((self.w[-1].shape[1] + 1, wElementsCnt), 0.1))
        self.w.append(np.full((wselementscnt[-1] + 1, classescnt), 0.1))

        if verbose: self.__printws()

    def __s(self, x):
        return 1 / (1 + np.exp(-x))
    def __sVector(self, v):
        return np.array([1 / (1 + np.exp(-x)) for x in v])
    def __t(self, n):
        return np.array([1 if i == n else 0 for i in range(0,10)])
    def __stepforwardAndBP(self, X):

        for featureClass in X:
            for feature in featureClass:
                tempOArray = []
                tempOHatArray = []
                o = feature
                for weight in self.w[:-1]:
                    o = np.append(o, 1)
                    tempOHatArray.append(o.T)
                    o = np.dot(o, weight)
                    o = self.__sVector(o)
                    tempOArray.append(o)

                o = np.append(o,1)
                tempOHatArray.append(o.T)
    #            print(len(tempOHatArray))
                o = np.dot( o[np.newaxis], self.w[-1])
                o = self.__sVector(o)

                ##static starting here
                e = o - self.__t(0)
                print(e)
            #    print(np.diag([ oD * ( 1 - oD ) for oD in tempOArray[-1]]))
                THETA3 = np.dot(np.diag([ oD * ( 1 - oD ) for oD in o[0]]), e.T)
                THETA2 = np.dot(np.dot(np.diag([ oD * ( 1 - oD ) for oD in tempOArray[-1]]), self.w[-1][:-1]), np.array(THETA3))
                THETA1 = np.dot(np.dot(np.diag([ oD * ( 1 - oD ) for oD in tempOArray[-2]]), self.w[-2][:-1]), np.array(THETA2))
            #    print(THETA3.shape)
           #     print(THETA1.shape)
                self.w[0] = np.dot(-0.1 , np.dot(THETA1, np.array(tempOHatArray[0][np.newaxis])).T)
                self.w[1] = np.dot(-0.1 , np.dot(THETA2, np.array(tempOHatArray[1][np.newaxis])).T)
                self.w[2] = np.dot(-0.1 , np.dot(THETA3, np.array(tempOHatArray[2][np.newaxis])).T)
           #     print("W SHAPES")
          #      print(self.w[0].shape)
         #       print(self.w[1].shape)

            #print(self.w[1][22:60])
#        tempThetArray = [np.dot(np.diag([ oD * ( 1 - oD ) for oD in o]), e)]

#            print(e)
      #      print(np.dot(np.diag(np.array([ oD * ( 1 - oD ) for oD in o])[0]), e.T))
      #      for index in range(len(self.w)-1, 0):
     #           print(tempThetArray)



    def train(self, trainfeatures):
        for i in range(0, 10):
            self.__stepforwardAndBP(trainfeatures)

        ##For each vector:
            ## find o3 (3 because 2 hidden layers) :

        ## Find confusion values

        ## If not satisfactory: back propogate
            ## calculate vector e.
            ## calculate matrix D.
        ## else stop:

        ##calculate b3 b2 b1
        ## find w1 w2 w3
        ## call train with w1 w2 w3

        ##

        ### This is for one vector.
#        w=[]
#        for X in xTrain:
            ## add one to vector
#            oDach = X
#            np.append(oDach, 1)
            ## multiply with w1 matrix #How to find w1?
#            y = oDach.T * w
#            o1 = np.array([s(i) for i in y])
#            np.append(o1, 1)
            ## Multiply y2 with w2 matrix # how to find w2?
#            y2 = o1.T * w2
#            o2 = np.array([s(i) for i in y2])
#            np.append(o2, 1)
#            y3 = o2.T * w3
#            o3 = np.array([s(i) for i in y3])

        ##calculate error
            #for item in o array:
                # en = o -

                
features = loadDigitsFromFile("zip.train/zip.train")
ne = NeuralNetwork(256, 10, True, 40, 50)
#print(features[0][:2])
ne.train(features)