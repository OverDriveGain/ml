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
        sum = sum + np.dot((xi-u).T, (xi-u))
        #sum = np.add(sum ,np.dot(np.subtract(xi,u).T, np.subtract(xi,u)))

    return np.dot((1/n) , sum)

def KovarianzUndUVonZeilenvektoren(vektor):
    b = np.array([[[vektor[0]]]])

    for i in range(1, len(vektor)):
        b = np.append(b, np.array([[[vektor[i]]]]), axis=1)
    b = b[0]

    u = calc_u(b)
    matrix = calc_sig(b,u)
    return u, matrix

def normalverteilung(u, m, x):
    k = len(x)
    print("k ist " + str(k))
    vorne = (2*np.pi)**(-k/2)
    mitte = np.linalg.det(m)**(-0.5)

    expt1 = -0.5*(x-u).T
    expt2 = np.linalg.pinv(m)
    expt3 = (x-u)
    #hinten = np.e**(np.dot(np.dot(expt1,expt2),expt3))
    hinten = np.e ** (expt1 * expt2 * expt3)
    return vorne*mitte*hinten



def main():
    '''
    # Importiere Daten
    X_train, y_train = load_from_file("zip.train/zip.train")
    #X_test, y_test = load_from_file("zip.test/zip.test")
    # separiere Trainingsdaten

    #ermittle mittelpunkte und kovarianzmatrizen
    u = []
    m = []
    for i in range(10):
        train_t = separateData(X_train,y_train,i)
        u0, m0 = KovarianzUndUVonZeilenvektoren(train_t)
        u.append(u0)
        m.append(m0)

    #v = np.array([(5, 10, 8), (4, 6, 3), (2, 3, 3), (6, 12, 3), (8, 14, 13)])
    #u, m = KovarianzUndUVonZeilenvektoren(v)
    #print(normalverteilung(u, m, np.array((1, 7, 3))))
    '''
    #testKovarianzWiki()
    #testKovarianz()
    testfuerpunkt()
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

def testKovarianzWiki():
    # https://de.wikipedia.org/wiki/Mehrdimensionale_Normalverteilung

    v = np.array([(3.3, 24, 27), (4.9, 41, 55),(5.9, 46, 52), (5.2, 49, 54), (3.6, 29, 34),(4.2,33,51),(5.0,42,43),(5.1,35,54),(6.8,60,70),(5.0,41,50)])
    v2 = np.array([(3.3, 24, 80)])
    v3 = np.array([(3.3, 90, 10)])


    u , m = KovarianzUndUVonZeilenvektoren(v)
    print("Mittelpunkt")
    print(u)
    print("Kovarianzmatrix")
    print(m)

    n1 = np.log(normalverteilung(u,m,v2))
    n2 = np.log(normalverteilung(u,m,v3))

    print("n1: ")
    print(n1)
    print("n2: ")
    print(n2)
    print("n1 > n2 ?: ")
    print(n1 > n2)

def testfuerpunkt():
    v = np.array([(1, 1), (3, 3), (3, 1), (1, 3)])
    v2 = np.array([(10, 10), (30, 30), (30, 10), (10, 30)])

    v3 = np.array([(2, 2)])

    u , m = KovarianzUndUVonZeilenvektoren(v)
    print("Mittelpunkt")
    print(u)
    print("Kovarianzmatrix")
    print(m)

    u2, m2 = KovarianzUndUVonZeilenvektoren(v2)
    print("Mittelpunkt 2")
    print(u2)
    print("Kovarianzmatrix 2")
    print(m2)

    n1 = np.log(normalverteilung(u,m,v3))
    n2 = np.log(normalverteilung(u2,m2,v3))

    print("n1: ")
    print(n1)
    print("n2: ")
    print(n2)
    print("n1 > n2 ?: ")
    print(n1 > n2)
























































































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