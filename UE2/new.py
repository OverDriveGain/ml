import pandas as pd
import numpy as np
import datetime
import math
from operator import add

class gaussianVerteilung:
    def __init__(self, name, age):
        self.classes = [[]]
    def train(self, xTrain, yTrain):

        # Klassen ordnen
        for index in range(0, len(xTrain)):
            self.classes[int(yTrain[index])].append(xTrain[index,:])

        # Berechne m fuer die Klassen
        for num, vectors in enumerate(classes):
            for vector in vectors:
                if(len(mues[num]) == 0): mues[num] = vector.copy()     ##Reference by value error!
                else: mues[num] += vector
            mues[num] = mues[num] / len(vectors)

        # Berechne kovarianz Matrix fuer die Klassen
        #calculate covariance matrix
        sigmas = []
        for index in range(0, len(classes)):
            sigma = []
            for num, vector in enumerate(classes[index]):
                sub = vector - mues[index]
                #           if(index==0): print(sub)
                #            if(index ==0): print(mues[index])
                #           if(index ==0):
                #              print (sub)
                #             print(np.dot(sub[np.newaxis].T , sub[np.newaxis]))
                if(sigma ==[]):
                    sigma = np.subtract(vector, mues[index]) * np.subtract(vector, mues[index])[np.newaxis].T
                else:
                    sigma += np.subtract(vector, mues[index]) * np.subtract(vector, mues[index])[np.newaxis].T
                if(index ==0): print ((sigma))

            #        if(index ==0): print (sigma)
            sigma = sigma / len(classes[index])
            if(index ==0): print ((sigma))
    #      if(index !=0): print ((sigma))
    #     sigmas.append(sigma)>



testArray = np.array([[5 ,10 ,8], [4,6,3] , [2, 3,3] , [6,12, 3 ] , [8,14,13]])

dev = 1
test = False

valuesCount = -1;
if(dev):
    valuesCount = 20

def loadFromFile(path):
    df = pd.read_csv(path, header=None, sep=" ")
    X = df.iloc[:valuesCount, 1:257].values # there is an empty string at position 257, because every line ends with a space (== separator)
    y = df.iloc[:valuesCount, 0].values
    return X, y

def start():
    xTrain, yTrain = loadFromFile("zip.train/zip.train")

    # separate classes.
    classes = np.array([])
    for i in range(10):
        np.concatenate(classes,np.array([]))
    for index in range(0, len(xTrain)):
        np.concatenate(classes[int(yTrain[index])],xTrain[index,:])

    if(test):
       classes[0] = testArray
    print("CLAASSES ARRAY")
    # calculate mue
    mues = [np.array([])]* 10
    for num, vectors in enumerate(classes):
        for vector in classes[num]:
            if(len(mues[num]) == 0): mues[num] = vector.copy()     ##Reference by value error!
            else: mues[num] += vector
        mues[num] = mues[num] / len(vectors)
        print(mues[num])
    #calculate covariance matrix
    sigmas = []
    for index in range(0, len(classes)):
        sigma = []
        for num, vector in enumerate(classes[index]):
            sub = vector - mues[index]
 #           if(index==0): print(sub)
#            if(index ==0): print(mues[index])
 #           if(index ==0):
  #              print (sub)
   #             print(np.dot(sub[np.newaxis].T , sub[np.newaxis]))
            if(sigma ==[]):
                sigma = np.subtract(vector, mues[index]) * np.subtract(vector, mues[index])[np.newaxis].T
            else:
                sigma += np.subtract(vector, mues[index]) * np.subtract(vector, mues[index])[np.newaxis].T

    #        if(index ==0): print (sigma)
        sigma = sigma / len(classes[index])
        sigmas.append(sigma)
  #      if(index !=0): print ((sigma[32:55]))
  #      if(index !=0): print ((sigma))
   #     sigmas.append(sigma)


    xTest, yTest = loadFromFile("zip.test/zip.test")

    #Prediction phase
    for index in range (0, len(xTest)):
        testProbabilities = []
        for index2 in range(0, len(sigmas)):
            sub = xTest[index] - mues[index2]
            sub = np.array(sub)
            print("Determinant")
            print(sub)
            print("forst")
            print(xTest[index])
            print("sec")
            print( mues[index2])

# print(sigmas[index2])
            probability =  np.exp(-0.5 * sub[np.newaxis].dot(sub[np.newaxis].T)) / np.sqrt( abs(2*math.pi*np.linalg.det(sigmas[index2])))   #Transpose at end: just simple way to reverse the attributes to dimensions of each vector
            #print(probability)
#            probability =  np.linalg.det(sigmas[index2])
            if(probability !=0 or True):
                print("Klasse is" + str(index2) + " that "+ str(probability))
            testProbabilities.append(probability)
    print("Finished")




def distance(v1, v2):
    if(len(v1) > len(v2)): v1 = v1[:len(v2)]
    if(len(v1) < len(v2)): v2 = v2[:len(v1)]
    return np.linalg.norm(v1 - v2)

a = datetime.datetime.now()
start()

print("Time consumed in seconds: " + str((datetime.datetime.now() - a ).seconds))

