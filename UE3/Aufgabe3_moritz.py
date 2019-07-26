import pandas as pd
import numpy as np

#Aufgabe 2
# abgabe von Moritz Walter und Manar Zaboub

def load_from_file(path):
    #importfunktion für daten. Übernommen aus Tutoriumsvorlage
    df = pd.read_csv(path, header=None, sep=" ")
    X = df.iloc[:, 1:257].values # there is an empty string at position 257, because every line ends with a space (== separator)
    y = df.iloc[:, 0].values
    return X, y

def separateData(X, y, digit):
    X_digit = X[y == digit]
    return X_digit

def calc_u(xtrain):
    n = len(xtrain)
    sum = 0
    for i in xtrain:
        sum = np.add(sum , i)

    return np.dot((1/n) , sum)

def calc_sig(xtrain, u):
    n = len(xtrain)
    sum = 0
    for xi in xtrain:
        sum = np.add(sum ,np.dot(np.subtract(xi,u).T, np.subtract(xi,u)))

    return np.dot((1/n) , sum)

def KovarianzUndUVonZeilenvektoren(vektor):
    b = np.array([[[vektor[0]]]])

    for i in range(1, len(vektor)):
        b = np.append(b, np.array([[[vektor[i]]]]), axis=1)
    b = b[0]

    u = calc_u(b)
    matrix = calc_sig(b,u)

    #u = calc_u(vektor)
    #matrix = calc_sig(vektor, u)

    return u, matrix

def normalverteilung(u, m, x):
    k = len(x)
    vorne = (2*np.pi)**(-k/2)
    mitte = np.linalg.det(m)**(-0.5)

    expt1 = -0.5*(np.subtract(x,u))
    expt2 = np.linalg.inv(m)
    expt3 = (np.subtract(x,u)).T
    hinten = np.e**(np.dot(np.dot(expt1,expt2),expt3))
    return vorne*mitte*hinten

def regularize(m):
    a = 0.2
    return np.add(np.dot(a, np.identity(len(m))), np.dot((1-a), m) )

def normalize_rows(x: np.ndarray):
    """
    von https://necromuralist.github.io/neural_networks/posts/normalizing-with-numpy/
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)

def reduceDimensions(m, dim):
    v = np.linalg.eigh(m)
    eigenwerte = v[0]#np.sort(v[0])[::-1]
    eigenvektoren = v[1]

    idx = eigenwerte.argsort()[::-1]
    eigenValuesSort = eigenwerte[idx]
    eigenVectorsSort = eigenvektoren[:, idx]


    #print(eigenValuesSort)
    umatrix =  normalize_rows(eigenVectorsSort)

    eigenwertmatrix = np.diag(eigenValuesSort)
    print(eigenwertmatrix)
    for i in range(dim, len(eigenwertmatrix[0])):
        eigenwertmatrix[i][i] = 0

    print(eigenwertmatrix)
    #print("m ")
    #print(m)
    #print("berechnung")
    print(np.dot(np.dot(umatrix, eigenwertmatrix), umatrix.T))
    #print(eigenVectorsSort)
    return umatrix, eigenwertmatrix



def test():
    v = np.array([(5,10,8),(4,6,3),(2,3,3),(6,12,3),(8,14,13)])

    u,m = KovarianzUndUVonZeilenvektoren(v)

    umatrix ,neum = reduceDimensions(m, 2)



def main():
    test()
    # Importiere Daten
   # X_train, y_train = load_from_file("zip.train/zip.train")
   # X_test, y_test = load_from_file("zip.test/zip.test")
   # print("Daten Importiert...")
    # separiere Trainingsdaten und
    #ermittle mittelpunkte und kovarianzmatrizen
   # u = []
   # m = []
   # for i in range(1):
   #     train_t = separateData(X_train,y_train,i)
   #     u0, m0 = KovarianzUndUVonZeilenvektoren(train_t)
   #     u.append(u0)
   #     m.append(regularize(m0)) # regularize all, da in jeder matrix beim postcode notwendig



    '''
    print("Mittelpunkte und Kovarianzmatrizen bestimmt...")
    # erstelle leere konfusionsmatrix
    gefunden = np.array([(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                         (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                         (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                         (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                         (3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                         (4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                         (5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                         (6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                         (7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                         (8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                         (9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)])

    for i in range(len(X_test)):
        # erstelle temp variablen
        x = X_test[i]  # aktuelles testbild
        testlabel = int(y_test[i])  # sollLabel für testbild

        # finde Label
        gefundenesLabel = -1
        maxWkt = 0
        for j in range(10):
            wkt = normalverteilung(u[j],m[j],x)[0][0]
            if wkt > maxWkt:
                maxWkt = wkt
                gefundenesLabel = j


        # Eintragen in Konfusionsmatrix
        gefunden[1 + testlabel, 1 + gefundenesLabel] = gefunden[1 + testlabel, 1 + gefundenesLabel] + 1

        # Fortschrittsausgabe
        if i % 200 == 0:
            print("Aktueller Status: " + str(i / len(X_test) * 100) + "% fertig")

    #ausgabe Konfusionsmatrix
    print(gefunden)
    '''



if __name__ == "__main__":
    main()
