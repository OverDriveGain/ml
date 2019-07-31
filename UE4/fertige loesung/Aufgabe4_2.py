import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('WebAgg')
import pandas as pd
from sklearn.model_selection import train_test_split as trainTestSplit
from scipy.interpolate import make_interp_spline, BSpline   #To plot a curve
# Aufgabe 4
# abgabe von Moritz Walter und Manar Zaboub


def normalise(x):
    v = x
    z = []
    for i in range(len(v)):
        z.append((v[i] / np.linalg.norm(v[i])))
    return np.array(z)

def loadSpamData():
    data = pd.read_csv("spambase.data", header=None).values
    X = data[:, :-1]
    y = data[:, -1]
    xTrain, xTest, yTrain, yTest = trainTestSplit(X, y, test_size=0.2, random_state=30, stratify=y)
    return xTrain, xTest, yTrain, yTest

class LogisticRegression:
    def s(self, x):
        return 1 / (1 + np.exp(-x))

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.beta = np.array( [ 0.0 for i in X[0]] )

    def stepTrain(self):    #One iteration
        x = np.dot(self.X, self.beta)
        h = self.s(x)
        gradient = np.dot( self.X.T, (h - self.y)) / self.y.size
        self.beta -= 0.01 * gradient        #Learn rate

    def predict(self, X):
        return self.s(np.dot(X, self.beta)).round()

xTrain, xTest, yTrain, yTest = loadSpamData()
plotX = []
plotY = []
model = LogisticRegression(xTrain, yTrain)
result = None

for i in range (0, 1000000):
    model.stepTrain()
    correct = 0
    result = model.predict(xTest)
    for i2 in range (0, len(result)):
        if(result[i2] == yTest[i2]):
            correct +=1
    plotY.append(correct / len(yTest))
    plotX.append(i)


def confMatrix(results, testValues):
    print("Prediction values :y, Actual values: x")
    print("\t", end="")
    tp = fp = tn = fn = 0
    for index in range (0, len(results)):
        if(results[index] ==1):
            if(testValues[index] ==1):
                tp+=1
            else:
                fp +=1
        if(results[index] == 0):
            if(testValues[index] ==0):
                tn+=1
            else:
                fn +=1
    print("0 \t 1")
    print("0 \t "+ str(tn) + "\t" + str(fn)  )
    print("1 \t "+ str(fp) + "\t" + str(tp)  )


confMatrix(result, yTest)

xnew = np.linspace(min(plotX),max(plotX),300)
spl = make_interp_spline(plotX, plotY, k=3) #BSpline object
power_smooth = spl(xnew)
plt.plot(xnew,power_smooth)
plt.show()

