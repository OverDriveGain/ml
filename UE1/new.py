import pandas as pd
import numpy as np
import datetime

def readFile(dir):
    df = pd.read_csv(dir, header=None, sep=" ")
    X = df.iloc[:,:].values
    return X[:]

def start(testValue):
    if(testValue ==2 or testValue ==1):
        print("Note that running the program with values 1 or 2 is equal. See most_common function")

    trainVectors = readFile("zip.train/zip.train")
    testVectors = readFile("zip.test/zip.test")
    print("länge train:" + str(len(trainVectors)))
    print("länge test:" + str(len(testVectors)))
    ret = [[]]*10
    for i in range (len(ret)): ret[i] = [0]*10
    for testVector in testVectors:
        res = findNnearest(testValue, testVector[1:], trainVectors[:, 1:])
        res = most_common(res, trainVectors[:,0:1])
        ret[int(testVector[0])][res] +=1
    print("Results of test with value "+ str(testValue))
    print("Comparision of results of program (horizontal line) and values in test(vertical line) ")
    print("    ", end="")
    for index in range(0, len(ret)):
        if(index == len(ret)-1): print(str(index))
        else: print(str(index), end=", ")
    for index in range(0, len(ret)):
        print(str(index) + "  " + str(ret[index]))

def findNnearest(near, inputVector, trainVectors):
    currentNearest = [[distance(trainVectors[0], inputVector),0]]
    for index in range(0, len(trainVectors)):
        d = distance(trainVectors[index], inputVector)
        if( d < currentNearest[0][0]):
            currentNearest.append([d,index])
            currentNearest.sort(key = lambda x : x[0], reverse=True)
            if(len(currentNearest)>near): currentNearest = currentNearest[1:near+1]
    currentNearest.reverse()
    return [i[1] for i in currentNearest]

def distance(v1, v2):
    if(len(v1) > len(v2)): v1 = v1[:len(v2)]
    if(len(v1) < len(v2)): v2 = v2[:len(v1)]
    return np.linalg.norm(v1 - v2)

def most_common(arr, labels):
    # Most commin of two values is always the first value. Most common of one value is the same value.
    if(len(arr) == 1 or len(arr) ==2):
        return int(labels[arr[0]])
    maximumFreq = labels[int(max(set(arr), key = lambda x: labels[x]))]
    return int(maximumFreq)

a = datetime.datetime.now()

for i in range(1,4):
    start(i)

print("Time consumed in seconds: " + str((datetime.datetime.now() - a ).seconds))
