import pandas as pd
import numpy as np

#Aufgabe 1
# abgabe von Moritz Walter und Manar Zaboub

def load_from_file(path):
    #importfunktion für daten. Übernommen aus Tutoriumsvorlage
    df = pd.read_csv(path, header=None, sep=" ")
    X = df.iloc[:, 1:257].values # there is an empty string at position 257, because every line ends with a space (== separator)
    y = df.iloc[:, 0].values
    return X, y

def abstandt(v1, v2):
    #euklidischer abstandt zweier vektoren
    return np.linalg.norm(v1 - v2)

def separateData(X, y, digit):
    X_digit = X[y == digit]
    return X_digit

def abstandtarray(testBild, trainingData):
    # berechne abstand eines testbilds zur ganzen trainingsmenge
    return np.array([abstandt(xi, testBild) for xi in trainingData])

def calc_u(xtrain):
    n = len(xtrain)
    sum = xtrain[0] - xtrain[0]
    for i in xtrain:
        sum = sum + i

    return (1/n) * sum


def calc_sig(xtrain, u):
    n = len(xtrain)
    sum = 0 #np.zeros((n,n)) # erstellt lerres array der richtigen dimension
    for xi in xtrain:
        sum = sum + (xi-u).T * (xi-u)

    return (1/n) * sum

def KovarianzUndUVonZeilenvektoren(vektor):
    b = np.array([[[vektor[0]]]])

    for i in range(1, len(vektor)):
        b = np.append(b, np.array([[[vektor[i]]]]), axis=1)
    b = b[0]

    u = calc_u(b)
    matrix = calc_sig(b,u)
    return u, matrix


def main():
    # Importiere Daten
    #X_train,
    # y_train = load_from_file("zip.train/zip.train")
    #X_test, y_test = load_from_file("zip.test/zip.test")
    # separiere Trainingsdaten

    #X_train_0 = separateData(X_train,y_train,0)
    #X_train_1 = separateData(X_train, y_train, 1)
    #X_train_2 = separateData(X_train, y_train, 2)
    #X_train_3 = separateData(X_train, y_train, 3)
    #X_train_4 = separateData(X_train, y_train, 4)
    #X_train_5 = separateData(X_train, y_train, 5)
    #X_train_6 = separateData(X_train, y_train, 6)
    #X_train_7 = separateData(X_train, y_train, 7)
    #X_train_8 = separateData(X_train, y_train, 8)
    #X_train_9 = separateData(X_train, y_train, 9)

    #testMitte()
    testKovarianz()

def testMitte():
    v = np.array([[(1, 1)], [(3, 3)], [(3, 1)], [(1, 3)]])
    ergebnis = calc_u(v)
    print(ergebnis)

def testKovarianz():
    '''
    Beispiel:
    Matrix
    (5, 10, 8),
    (4, 6, 3),
    (2, 3, 3),
    (6, 12, 3),
    (8, 14, 13)

    Mittelpunkt: (5,9,6)

    Kovarianz:

    (4, 7.8, 6),
    (7.8, 16, 11),
    (6, 11, 16)
    '''

    v = np.array([(5, 10, 8), (4, 6, 3),(2, 3, 3), (6, 12, 3), (8, 14, 13)])

    u , m = KovarianzUndUVonZeilenvektoren(v)
    print("Mittelpunkt")
    print(u)
    print("Kovarianzmatrix")
    print(m)



























































































def findNnearestAndCreateTable(nearest, X_train, X_test, y_train, y_test):
    # erstelle leere konsusionsmatrix
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

    for i in range(len(X_test)):
        #erstelle temp variablen
        maxlabels = np.zeros(10) # leeres array zum ermitteln der label
        testbild = X_test[i] # aktuelles testbild
        testlabel = int(y_test[i]) # sollLabel für testbild

        #berechne Abstandsarray mit Soll-labels
        abstandt = np.array([abstandtarray(testbild, X_train), y_train])

        #sortiere Abstandsarray
        abstandtSorted = abstandt[:, abstandt[0].argsort()]

        # ermittle label für die n nearest
        for n in range(nearest):
            label = int(abstandtSorted[1, n])
            maxlabels[label] = maxlabels[label] + 1

        # ermittle maximum anzahl der ermittelten n nearest labels
        max = 0
        currLabel = -1
        for l in range(10):
            if maxlabels[l] > max:
                max = maxlabels[l]
                currLabel = l

        # Eintragen in Konfusionsmatrix
        gefunden[1 + testlabel, 1 + currLabel] = gefunden[1 + testlabel, 1 + currLabel] + 1

        # Fortschrittsausgabe
        if i % 200 == 0:
            print("Status für " + str(nearest) + "-NN " + str(i / len(X_test) * 100) + "%")
    return gefunden

if __name__ == "__main__":
    main()