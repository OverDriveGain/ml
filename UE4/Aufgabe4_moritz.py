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

    #for i in X_train:


    spam_train = separate_data(X_train,y_train,1)
    no_spam_train = separate_data(X_train, y_train, 0)

    spam_mean = np.mean(spam_train)
    no_spam_mean = np.mean(no_spam_train)
    train_mean = np.mean(X_train)

    u = calcFisherDisk(spam_train,no_spam_train) ## erste klasse > 0

    # projeziere Daten

    data = np.dot(X_test, u)

    spam = np.dot(spam_train, u)
    nspam = np.dot(no_spam_train, u)


    # erstelle konfusionsmatrix
    gefunden = np.array([(0, 0, 1),
                         (0, 0, 0),
                         (1, 0, 0)])
    for i in range(len(data)):
        # print(i)
        p = data[i]
        aktuellesLabel = int(y_test[i])
        gefundenes_label = -1
        if p > 0.9:
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
    # fig, axs = plt.subplots(1, 1, figsize=(5, 5) )
    #
    # axs.scatter(nspam, np.zeros(len(nspam)), c ="blue", s=1)
    # axs.scatter(spam, np.ones(len(spam)), c="red", s=1)
    #
    # plt.show()

    # e = np.dot( u, 0.5*(np.add(spam_mean,no_spam_mean)))
    # print(e)

def normalise(x):
    v = x
    z = []
    for i in range(len(v)):

        z.append((v[i] / np.linalg.norm(v[i])))
    return np.array(z)

def aufgabe2(X_train, X_test, y_train, y_test):
    print("Aufgabe 2")

    # normalisieren
    x_train_norm = normalise(X_train)
    x_test_norm = normalise(X_test)
    y_train_norm = normalise(y_train)
    y_test_norm = normalise(y_test)

    # 1 einfügen
    x_train_n_one = np.insert(x_train_norm, 0, 1, axis=1)
    x_test_n_one = np.insert(x_test_norm, 0, 1, axis=1)

    
    print()



def p_a(x,beta):
    return np.exp(np.dot(x.T,beta)) / (1 + np.exp(np.dot(x.T,beta)))

def p_b(x, beta):
    return 1 - p_a(x,beta)












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




def fisherTest():
    c1 = np.array([(1,2),(2,3),(3,3),(4,5),(5,5)])
    c2 = np.array([(1,0),(2,1),(3,1),(3,2),(5,3),(6,5)])
    print("class 1")
    print(c1)
    print("Class 2")
    print(c2)

    # compute means, should be u1 = (3, 3.6), u2 = (3.3, 2)

    u1 = np.mean(c1, axis=0)
    u2 = np.mean(c2, axis=0)

    print("Mean c1" + str(u1))
    print("Mean c2" + str(u2))
    #

    # compute scatter matrices:
    # s1 = 4*cov(c1) = ([[(10,8)],[(8.0,7.2)]])
    # s2 = 5*cov(c2) = ([[(17.3,16)],[(16,16)]])

    # s1 = 4*np.cov(c1.T)  # warum 4
    # s2 = 5*np.cov(c2.T) # warum 5

    s1 = np.cov(c1.T)  # warum 4
    s2 = np.cov(c2.T)  # warum 5

    print("Kovariance c1" + str(s1))
    print("Kovariance c2" + str(s2))

    #SW = s1 + s2 = ([[(27.3, 24)],[(24, 23.2)]])

    SW = s1 + s2

    print("SW: " + str(SW))

    #SW^-1 = ([[(0.39,-0.41)],[(-0.41,0.47)]])

    SWinv = np.linalg.inv(SW)

    print("SW: " + str(SWinv))


    # optimal line direction: v = SW^-1 *(u1-u2) = ([(-0.79,0.89)])

    v = np.dot(SWinv,np.subtract(u1,u2))
    print("v: " + str(v))

    # Y1 line vektor = v.T * c1.T  = [0.81 ... 0.4]
    # Y2 line vektor = v.T * c2.T  = [-0.65 ... -0.25]

    y1 = np.dot(v.T,c1.T)
    y2 = np.dot(v.T,c2.T)
    print("y1: " + str(y1))
    print("y2: " + str(y2))

    print("länge v: " + str(np.linalg.norm(v)))


    z= (np.dot(c1, v) > 0)
    a = True
    for i in z:
        a = a and i

    r = (np.dot(c1, v) > 0)
    b = True
    for i in z:
        b = b and i
    print("a: " + str(a))
    print("b: " + str(a))
    # fig, axs = plt.subplots(1, 1, figsize=(5, 5) )
    #
    # x, y = c1[:,0],  c1[:,1]
    # x1, y1 = c2[:,0],  c2[:,1]
    # axs.scatter(x, y, c ="blue", s=1)
    # axs.scatter(x1, y1, c="red", s=1)
    # axs.quiver(0,0, v[0], v[1], color="green")
    # axs.quiver(0, 0, y1[0], y1[1], color="blue")
    # axs.quiver(0, 0, y2[0], y2[1], color="red")
    #
    # u = calcFisherDisk(c1,c2)
    # axs.quiver(1, 1, u[0], u[1], color="yellow")
    # plt.show()


def main():
    X_train, X_test, y_train, y_test = load_spam_data()
    #fisherTest()

    #aufgabe1(X_train, X_test, y_train, y_test)
    aufgabe2(X_train, X_test, y_train, y_test)




#
# def add_intercept(X):
#     intercept = np.ones((X.shape[0], 1))
#     return np.concatenate((intercept, X), axis=1)
#
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
#
# def loss(h, y):
#     return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
#
#
# def beispiel_log(X, y):
#     lr=0.01
#     num_iter=100000
#     fit_intercept=True
#     verbose=False
#
#
#
#     if fit_intercept:
#         X = add_intercept(X)
#
#     # weights initialization
#     theta = np.zeros(X.shape[1])
#
#     for i in range(num_iter):
#         z = np.dot(X, theta)
#         h = sigmoid(z)
#         gradient = np.dot(X.T, (h - y)) / y.size
#         theta -= lr * gradient
#
#         if (verbose == True and i % 10000 == 0):
#             z = np.dot(X, theta)
#             h = sigmoid(z)
#             print(f'loss: {loss(h, y)} \t')
#
#
# def predict_prob(X, theta, fit_intercept):
#     if fit_intercept:
#         X = add_intercept(X)
#
#     return sigmoid(np.dot(X, theta)) ,fit_intercept
#
#
# def predict(X, threshold, theta, fit_intercept):
#     a, b = predict_prob(X, theta, fit_intercept)
#     return  a >= threshold , b
#




if __name__ == "__main__":
    main()
