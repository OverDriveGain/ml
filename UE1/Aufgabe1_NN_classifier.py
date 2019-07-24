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

def abstandtarray(testBild, trainingData):
    # berechne abstand eines testbilds zur ganzen trainingsmenge
    return np.array([abstandt(xi, testBild) for xi in trainingData])

def main():
    # Importiere Daten
    X_train, y_train = load_from_file("zip.train/zip.train")
    X_test, y_test = load_from_file("zip.test/zip.test")
    print("länge train:" + str(len(X_train)))
    print("länge test:" + str(len(X_test)))

    # Mit einem Nachtbar
    gefunden1 = findNnearestAndCreateTable(1,X_train,X_test,y_train,y_test)
    print("1-NN")
    print(gefunden1)
    #Mit zwei Nachtbarn
    gefunden2 = findNnearestAndCreateTable(2, X_train, X_test, y_train, y_test)
    print("2-NN")
    print(gefunden2)
    #Mit drei Nachtbarn
    gefunden3 = findNnearestAndCreateTable(3, X_train, X_test, y_train, y_test)
    print("3-NN")
    print(gefunden3)


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