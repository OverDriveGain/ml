import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.image import imread


# Aufgabe 3
# abgabe von Moritz Walter und Manar Zaboub


def load_from_file(path):
    # importfunktion für daten. Übernommen aus Tutoriumsvorlage
    df = pd.read_csv(path, header=None, sep=" ")
    x = df.iloc[:, 1:257].values
    # there is an empty string at position 257, because every line ends with a space (== separator)
    y = df.iloc[:, 0].values
    return x, y


def separate_data(x, y, digit):
    x_digit = x[y == digit]
    return x_digit


def project_data(vectors, data, mu):
    return np.dot(data - mu, vectors)


def aufgabe1():
    x_train, y_train = load_from_file("zip.train/zip.train")
    x_test, y_test = load_from_file("zip.test/zip.test")

    x_redu_v, x_redu_u = dimensionsreduktion(x_train, 2)
    x_test_v, x_test_u = dimensionsreduktion(x_test, 2)

    x_redu = project_data(x_redu_v, x_redu_u, x_train)
    x_test_redu = project_data(x_test_v, x_test_u, x_test)

    points = []

    for i in range(10):
        points.append(separate_data(x_redu, y_train, i))

    fig, axs = plt.subplots(9, 9, figsize=(18, 9), sharex=True, sharey=True)
    # fig.suptitle("linke Zahl Blau, rechte Zahl orange", y=1.05)
    # fig.tight_layout(rect=[0, 0.10, 1, 0.80])
    for i in range(9):
        for j in range(10):
            if j > i:
                i_p = points[i]
                j_p = points[j]
                axs[i, j - 1].scatter(i_p[:, 0], i_p[:, 1], s=1)
                axs[i, j - 1].scatter(j_p[:, 0], j_p[:, 1], s=1)
                axs[i, j - 1].set_title("(" + str(i) + "," + str(j) + ")")
                aufgabe1_part2(i, j, points, x_test_redu, y_test, )
            else:
                if j > 0:
                    axs[i, j - 1].axis('off')

    axs[3, 1].set_title("(blau,orange)")
    plt.tight_layout()


def aufgabe1_part2(klasse1, klasse2, punkte, testdata, test_labels):
    print("Linear Regression für " + "(" + str(klasse1) + "," + str(klasse2) + ")")

    bvektor = calc_linear_regression_2_d(punkte[klasse1], punkte[klasse2])  # Nahe 1 für Klasse 1, Nache -1 für Klasse 2
    k1_test = separate_data(testdata, test_labels, klasse1)
    k2_test = separate_data(testdata, test_labels, klasse2)

    print("Konfusionsmatrix ")
    gefunden = np.array([(0, klasse1, klasse2),
                         (klasse1, 0, 0),
                         (klasse2, 0, 0)])

    for i in k1_test:
        # print(i)
        p = klassify_new_point_regression(np.array([i]), bvektor)
        gefundenes_label = 0
        if p > 0:
            gefundenes_label = 1  # gehört zu klasse 1
        else:
            gefundenes_label = 2  # gehört zu klasse 2
        gefunden[1, gefundenes_label] = gefunden[1, gefundenes_label] + 1

    for i in k2_test:
        p = klassify_new_point_regression(np.array([i]), bvektor)
        gefundenes_label = 0
        if p > 0:
            gefundenes_label = 1  # gehört zu klasse 1
        else:
            gefundenes_label = 2  # gehört zu klasse 2
        gefunden[2, gefundenes_label] = gefunden[2, gefundenes_label] + 1
    print(gefunden)
    fehler = gefunden[1, 2] + gefunden[2, 1]
    richtig = gefunden[1, 1] + gefunden[2, 2]
    print("Fehler: " + str(fehler))
    print("Richtig: " + str(richtig))
    print("Fehlerquote: " + str((fehler / (richtig + fehler)) * 100) + " %")
    print("")


def calc_linear_regression_2_d(data_k1, data_k2):
    # Klasse 1 wird mit 1 gelabeld, Klasse 2 mit -1
    a = np.full((1, len(data_k1)), 1)
    b = np.full((1, len(data_k2)), -1)
    y = np.hstack((a, b))
    k1 = np.insert(data_k1, 0, 1, axis=1)
    k2 = np.insert(data_k2, 0, 1, axis=1)
    x = np.vstack((k1, k2))
    b = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y.T)
    return b.T


def klassify_new_point_regression(datapoint, b):
    # Nahe 1 für Klasse 1, Nache -1 für Klasse 2
    return sum(np.multiply(b, np.insert(datapoint, 0, 1, axis=1))[0])


def dimensionsreduktion(bilder, dimensionen):
    # berechne durchschnittsgesicht
    mu = bilder.mean(axis=0)

    # ziehe durchschnitt ab
    bilder_durch = bilder - mu

    [n, d] = bilder.shape

    if n > d:
        c = np.dot(bilder_durch.T, bilder_durch)
        v = np.linalg.eigh(c)
        eigenwerte = v[0]  # np.sort(v[0])[::-1]
        eigenvektoren = v[1]
    else:
        c = np.dot(bilder_durch, bilder_durch.T)
        v = np.linalg.eigh(c)
        eigenwerte = v[0]  # np.sort(v[0])[::-1]
        eigenvektoren = np.dot(bilder_durch.T, v[1])

        # normiere eigenvektoren
        for i in range(n):
            eigenvektoren[:, i] = eigenvektoren[:, i] / np.linalg.norm(eigenvektoren[:, i])

    # sortieren
    idx = eigenwerte.argsort()[::-1]
    # eigenValuesSort = eigenwerte[idx]
    eigen_vectors_sort = eigenvektoren[:, idx]

    # übrige dimensionen entfernen
    tossed = eigen_vectors_sort[0:dimensionen, :]
    return tossed.T, mu


def aufgabe2():
    x = []
    # h = w = 0
    max_num_images = 200  # np.inf
    path = "lfwcrop_grey/faces"

    for i, filepath in enumerate(glob.glob(os.path.join(path, "*.pgm"))):
        img = imread(filepath)
        x.append(img.flatten())
        if i >= max_num_images:
            break

    # h, w = img.shape
    x = np.array(x)
    # print("image heigth: {}  image width: {}".format(h, w))
    # print(X.shape)

    num_samples = 90
    indices = np.random.choice(range(len(x)), num_samples)

    # eigenfaces in orginalgröße

    sample_faces, q = dimensionsreduktion(x[indices], 4096)
    # sample_faces = sample_faces.T

    fig = plt.figure(figsize=(20, 6))
    fig.suptitle("Eigenfaces Dimension 4096, 96x96 Pixel")
    for i in range(num_samples):
        # ax = plt.subplot(6, 15, i + 1)
        plt.subplot(6, 15, i + 1)
        img = sample_faces[i].reshape((64, 64))
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    # eigenfaces mit dim = 400

    sample_faces_small, q2 = dimensionsreduktion(x[indices], 400)
    # sample_faces_small = sample_faces_small.T

    fig = plt.figure(figsize=(20, 6))
    fig.suptitle("Eigenfaces Dimension 400,  20x20 Pixel")
    for i in range(num_samples):
        # ax = plt.subplot(6, 15, i + 1)
        plt.subplot(6, 15, i + 1)
        img = sample_faces_small[i].reshape((20, 20))
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.show()


def main():
    aufgabe1()
    aufgabe2()


if __name__ == "__main__":
    main()
