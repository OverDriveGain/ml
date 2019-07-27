import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from matplotlib.image import imread

def faces_Example():
    X = []
    h = w = 0
    max_num_images = np.inf
    path = "lfwcrop_grey/faces"

    for i, filepath in enumerate(glob.glob(os.path.join(path, "*.pgm"))):
        img = imread(filepath)
        X.append(img.flatten())

        if i >= max_num_images:
           break

    h, w = img.shape
    X = np.array(X)
    print("image heigth: {}  image width: {}".format(h, w))
    print(X.shape)

    num_samples = 90
    indices = np.random.choice(range(len(X)), num_samples)
    sample_faces = X[indices]

    fig = plt.figure(figsize=(20, 6))

    for i in range(num_samples):
        ax = plt.subplot(6, 15, i + 1)
        img = sample_faces[i].reshape((64, 64))
        plt.imshow(img, cmap='gray')
        plt.axis('off')





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

def covariance_matrix_loop(X, mu):
    rows, cols = X.shape
    cov = np.zeros((cols, cols))
    for x in X:
      cov += np.outer(x-mu, x-mu)
    return cov / rows

def regularize_covariance_matrix(cov, alpha_min):
    alpha = alpha_min
    cov_reg = np.eye(len(cov)) * alpha + (1 - alpha) * cov
    while np.linalg.det(cov_reg) == 0.0:
      alpha += 0.01
      cov_reg = np.eye(len(cov)) * alpha + (1 - alpha) * cov
    return cov_reg

def normalverteilung(u, m, x):
    x = x
    k = len(x)
    vorne = (2*np.pi)**(-k/2)
    mitte = np.linalg.det(m)**(-0.5)

    expt1 = -0.5*(np.subtract(x,u))
    expt2 = np.linalg.inv(m)
    expt3 = (np.subtract(x,u)).T
    hinten = np.e**(np.dot(np.dot(expt1,expt2),expt3))
    return vorne*mitte*hinten



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
    umatrix =  normalize_rows(eigenVectorsSort)[:,:dim]

    eigenwertmatrix = np.diag(eigenValuesSort)
    print(eigenwertmatrix)
    for i in range(dim, len(eigenwertmatrix[0])):
        eigenwertmatrix[i][i] = 0

    print(eigenwertmatrix)
    #print("m ")
    #print(m)
    #print("berechnung")
    print(np.dot(np.dot(umatrix.T, eigenwertmatrix),np.array([(5,10,8)])))
    #print(np.dot(np.dot(umatrix.T, eigenwertmatrix), umatrix))
    #print(eigenVectorsSort)
    return umatrix, eigenwertmatrix



def test():
    v = np.array([(5,10,8),(4,6,3),(2,3,3),(6,12,3),(8,14,13)])

    u = calc_u(v)
    m0 = covariance_matrix_loop(v, u)
    m = regularize_covariance_matrix(m0, 0.01)
    umatrix ,neum = reduceDimensions(m, 2)



def main():
    test()
    # Importiere Daten
    #faces_Example()


def aufgabe2maincode():
    X_train, y_train = load_from_file("zip.train/zip.train")
    X_test, y_test = load_from_file("zip.test/zip.test")
    print("Daten Importiert...")
    # separiere Trainingsdaten und
    # ermittle mittelpunkte und kovarianzmatrizen
    u = []
    m = []
    for i in range(10):
        train_t = separateData(X_train, y_train, i)
        u0 = calc_u(train_t)
        m0 = covariance_matrix_loop(train_t, u0)
        u.append(u0)
        m.append(regularize_covariance_matrix(m0, 0.01))  # regularize all, da in jeder matrix beim postcode notwendig

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
            wkt = normalverteilung(u[j], m[j], x)  # log_normal_distribution_pdf(x,m[j],u[j])
            if wkt > maxWkt:
                maxWkt = wkt
                gefundenesLabel = j

        # Eintragen in Konfusionsmatrix
        gefunden[1 + testlabel, 1 + gefundenesLabel] = gefunden[1 + testlabel, 1 + gefundenesLabel] + 1

        # Fortschrittsausgabe
        if i % 200 == 0:
            print("Aktueller Status: " + str(i / len(X_test) * 100) + "% fertig")

    # ausgabe Konfusionsmatrix
    print(gefunden)




if __name__ == "__main__":
    main()
