import pandas as pd
import numpy as np
import datetime
from operator import add

testArray = np.array([[5 ,10 ,8], [4,6,3] , [2, 3,3] , [6,12, 3 ] , [8,14,13]])

def readFile(dir):
    df = pd.read_csv(dir, header=None, sep=" ")
    X = df.iloc[:,:257].values
    return X[:10]

def start():
    trainVectors = readFile("zip.train/zip.train")

    # separate classes.
    classes = [[]]*10
    for trainVector in trainVectors:
        classes[int(trainVector[0])].append(trainVector[1:])

    classes[0] = testArray
    # calculate sum
    mues = [0]* 10
    for index in range (0, len(classes)):
        mues[index] = np.array([sum(x) for x in zip(*classes[index])])/ len(classes[index])
    print("Sum of class one vectors is " + str(mues[0]))

    #calculate kovarianz
    for index in range( 0,len(classes)):
        substraction = []
        for index2 in range (0, len(classes[index])):
            print("Multiplying" + str(np.subtract(classes[index][index2], mues[index])[np.newaxis].T) + " with " + str(np.subtract(classes[index][index2], mues[index])))
            print("LENGH OF FIRST " + str(len(np.subtract(classes[index][index2], mues[index]))) + " len of second" + str(len(np.subtract(classes[index][index2], mues[index])[np.newaxis].T)))
            substraction = np.subtract(classes[index][index2], mues[index]) * np.subtract(classes[index][index2], mues[index])[np.newaxis].T
            print("Substraction" + str(substraction))

    substraction = substraction / len(classes[index])





def distance(v1, v2):
    if(len(v1) > len(v2)): v1 = v1[:len(v2)]
    if(len(v1) < len(v2)): v2 = v2[:len(v1)]
    return np.linalg.norm(v1 - v2)

a = datetime.datetime.now()
start()

print("Time consumed in seconds: " + str((datetime.datetime.now() - a ).seconds))

