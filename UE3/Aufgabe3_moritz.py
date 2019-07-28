import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from matplotlib.image import imread

# Aufgabe 3
# abgabe von Moritz Walter und Manar Zaboub


def load_from_file(path):
    # importfunktion für daten. Übernommen aus Tutoriumsvorlage
    df = pd.read_csv(path, header=None, sep=" ")
    X = df.iloc[:, 1:257].values
    # there is an empty string at position 257, because every line ends with a space (== separator)
    y = df.iloc[:, 0].values
    return X, y


def separateData(X, y, digit):
    X_digit = X[y == digit]
    return X_digit


def Aufgabe1():
    X_train, y_train = load_from_file("zip.train/zip.train")
    X_test, y_test = load_from_file("zip.test/zip.test")


    X_redu_v, X_redu_u = dimensionsreduktion(X_train, 2)
    X_test_v, X_test_u = dimensionsreduktion(X_test, 2)


    X_redu = project(X_redu_v,X_redu_u,X_train)
    X_test_redu = project(X_test_v,X_test_u,X_test)

    points = []

    for i in range(10):
        points.append(separateData(X_redu,y_train,i))

    fig, axs = plt.subplots(9, 9, figsize=(18, 9),sharex=True,sharey=True )
    #fig.suptitle("linke Zahl Blau, rechte Zahl orange", y=1.05)
    #fig.tight_layout(rect=[0, 0.10, 1, 0.80])
    for i in range(9):
        for j in range(10):
            if j > i:
                iP = points[i]
                jP = points[j]
                axs[i, j-1].scatter(iP[:, 0], iP[:, 1], s=1)
                axs[i, j-1].scatter(jP[:, 0], jP[:, 1], s=1)
                axs[i, j-1].set_title("("+ str(i) + ","+ str(j)+")")
                Aufgabe1Part2(i,j,points,X_test_redu,y_test,)
            else:
                if j > 0:
                    axs[i, j-1].axis('off')
                #if i==7 and j == 4:
                    #axs[i, j-1].set_title("linke Zahl Blau, rechte Zahl orange")

    plt.tight_layout()
    plt.show()


def Aufgabe1Part2(Klasse1,Klasse2,Punkte,Testdata,TestLabels):
    print("Linear Regression for " + "("+ str(Klasse1) + ","+ str(Klasse2)+")")

    bvektor = calc_linear_regression_2D(Punkte[Klasse1],Punkte[Klasse2]) #Nahe 1 für Klasse 1, Nache -1 für Klasse 2
    K1_test = separateData(Testdata,TestLabels,Klasse1)
    K2_test = separateData(Testdata,TestLabels,Klasse2)

    print("Konfusionsmatrix ")
    gefunden = np.array([(0, Klasse1, Klasse2),
                         (Klasse1, 0, 0),
                         (Klasse2, 0, 0)])

    for i in K1_test:
        #print(i)
        p = klassify_newPoint_regression(np.array([i]),bvektor)
        gefundenesLabel = 0
        if p > 0:
            gefundenesLabel = 1 # gehört zu klasse 1
        else:
            gefundenesLabel = 2 # gehört zu klasse 2
        gefunden[1,gefundenesLabel] = gefunden[1, gefundenesLabel] + 1

    for i in K2_test:
        p = klassify_newPoint_regression(np.array([i]),bvektor)
        gefundenesLabel = 0
        if p > 0:
            gefundenesLabel = 1 # gehört zu klasse 1
        else:
            gefundenesLabel = 2 # gehört zu klasse 2
        gefunden[2,gefundenesLabel] = gefunden[2, gefundenesLabel] + 1
    print(gefunden)
    fehler = gefunden[1,2] + gefunden[2,1]
    richtig =  gefunden[1,1] + gefunden[2,2]
    print("Fehler: " + str(fehler))
    print("Richtig: " + str(richtig))
    print("Fehlerquote: " + str(fehler / richtig * 100) + " %")


def calc_linear_regression_2D(dataK1, dataK2):
    # Klasse 1 wird mit 1 gelabeld, Klasse 2 mit -1
    a = np.full((1, len(dataK1)), 1)
    b = np.full((1, len(dataK2)), -1)
    y = np.hstack((a, b))
    K1 = np.insert(dataK1,0,1,axis = 1)
    K2 = np.insert(dataK2,0,1,axis = 1)
    X = np.vstack((K1, K2))
    b = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)),X.T),y.T)
    return b.T


def klassify_newPoint_regression(datapoint, b):
    # Nahe 1 für Klasse 1, Nache -1 für Klasse 2
    return sum(np.multiply(b,np.insert(datapoint, 0, 1, axis=1))[0])


def dimensionsreduktion(bilder, dimensionen):
    # berechne durchschnittsgesicht
    mu = bilder.mean(axis=0)

    # ziehe durchschnitt ab
    bilder_durch = bilder - mu

    [n, d] = bilder.shape

    if n > d:
        c = np.dot(bilder_durch.T,bilder_durch)
        v = np.linalg.eigh(c)
        eigenwerte = v[0]  # np.sort(v[0])[::-1]
        eigenvektoren = v[1]
    else:
        c = np.dot(bilder_durch, bilder_durch.T)
        v = np.linalg.eigh(c)
        eigenwerte = v[0]  # np.sort(v[0])[::-1]
        eigenvektoren = np.dot(bilder_durch.T,v[1])

        # normiere eigenvektoren
        for i in range(n):
            eigenvektoren[:, i] = eigenvektoren[:, i] / np.linalg.norm(eigenvektoren[:, i])

    # sortieren
    idx = eigenwerte.argsort()[::-1]
    # eigenValuesSort = eigenwerte[idx]
    eigen_vectors_sort = eigenvektoren[:, idx]

    # übrige dimensionen entfernen
    tossed = eigen_vectors_sort[:, :dimensionen]

    return tossed , mu


def aufgabe2():
    x = []
    h = w = 0
    max_num_images = 200  # np.inf
    path = "lfwcrop_grey/faces"

    for i, filepath in enumerate(glob.glob(os.path.join(path, "*.pgm"))):
        img = imread(filepath)
        x.append(img.flatten())
        if i >= max_num_images:
            break

    h, w = img.shape
    x = np.array(x)
    # print("image heigth: {}  image width: {}".format(h, w))
    # print(X.shape)

    num_samples = 90
    indices = np.random.choice(range(len(x)), num_samples)

    sample_faces, q = dimensionsreduktion(x[indices], 4096)
    sample_faces = sample_faces.T

    fig = plt.figure(figsize=(20, 6))
    for i in range(num_samples):
        ax = plt.subplot(6, 15, i + 1)
        img = sample_faces[i].reshape((64, 64))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()


def project(vectors, data, mu=None):
    if mu is None:
        return np.dot(data, vectors)
    return np.dot(data - mu, vectors)


def main():
    Aufgabe1()
    aufgabe2()


if __name__ == "__main__":
    main()