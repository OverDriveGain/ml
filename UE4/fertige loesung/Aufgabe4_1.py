import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Aufgabe 4
# abgabe von Moritz Walter und Manar Zaboub

def load_spam_data():

    data = pd.read_csv("spambase.data", header=None).values
    X = data[:, :-1]
    y = data[:, -1]

    print("Der Datensatz besteht aus %d E-Mails, wovon %d Spam sind und %d nicht" % (len(y), sum(y == 1), sum(y == 0)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)
    return X_train, X_test, y_train, y_test

def separate_data(x, y, digit):
    x_digit = x[y == digit]
    return x_digit

def aufgabe1(X_train, X_test, y_train, y_test):
    print("Aufgabe 1")

    spam_train = separate_data(X_train,y_train,1)
    no_spam_train = separate_data(X_train, y_train, 0)

    u = calcFisherDisk(spam_train,no_spam_train) ## erste klasse > 0

    # projeziere Daten

    testdata = np.dot(X_test, u)

    # erstelle konfusionsmatrix
    gefunden = np.array([(0, 0, 1),
                         (0, 0, 0),
                         (1, 0, 0)])
    for i in range(len(testdata)):
        p = testdata[i]
        aktuellesLabel = int(y_test[i])
        gefundenes_label = -1
        if p > 0.92:
            gefundenes_label = 2  # gehört zu klasse spam
        else:
            gefundenes_label = 1  # gehört zu klasse kein spam
        gefunden[aktuellesLabel + 1, gefundenes_label] = gefunden[aktuellesLabel + 1, gefundenes_label] + 1

    print(gefunden)
    fehler = gefunden[1, 2] + gefunden[2, 1]
    richtig = gefunden[1, 1] + gefunden[2, 2]
    print("Fehler: " + str(fehler))
    print("Richtig: " + str(richtig))
    print("Fehlerquote: " + str((fehler / (richtig + fehler)) * 100) + " %")

    # plot points on line
    spam = np.dot(spam_train, u)
    nspam = np.dot(no_spam_train, u)

    fig, axs = plt.subplots(1, 1, figsize=(20, 1) )

    axs.scatter(nspam, np.zeros(len(nspam)), c ="blue", s=1)
    axs.scatter(0.92,0.5, c="black", s=1)
    axs.scatter(spam, np.ones(len(spam)), c="red", s=1) # plotte höher um überlappung besser zu sehen

    plt.show()

def normalise(x):
    v = x
    z = []
    for i in range(len(v)):

        z.append((v[i] / np.linalg.norm(v[i])))
    return np.array(z)

def regularize_covariance_matrix(cov, alpha_min):
    alpha = alpha_min
    cov_reg = np.eye(len(cov)) * alpha + (1 - alpha) * cov
    while np.linalg.det(cov_reg) == 0.0:
      alpha += 0.01
      cov_reg = np.eye(len(cov)) * alpha + (1 - alpha) * cov
    return cov_reg

def calcFisherDisk(klasse1, klasse2):
    # vektoren in format: [(1,...),(2,....), .... ,(n,......)]
    # berechne ua und ub
    u1 = np.mean(klasse1, axis=0)
    u2 = np.mean(klasse2, axis=0)

    # berechne Kovariansmatrizen

    cov1 = np.cov(klasse1.T)  # warum 4
    cov2 = np.cov(klasse2.T)  # warum 5

    # berechne summe
    cov_sum = cov1 + cov2

    #damit immer invertierbar
    if np.linalg.det(cov_sum) == 0.0:
        cov_sum = regularize_covariance_matrix(cov_sum,0.001)

    # berechne diskriminante u

    u = np.dot(np.linalg.inv(cov_sum),np.subtract(u1,u2))

    return u


def main():
    X_train, X_test, y_train, y_test = load_spam_data()
    aufgabe1(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
