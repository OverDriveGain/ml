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

    plt.show()
    print("finished")





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
'''
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
'''
def reduce_dimensions(data, dim, u, m):
    #print("data")
    v = data
    vreduced = np.subtract(v,u)
    #mreg = regularize_covariance_matrix(m, 0.01)
    v = np.linalg.eigh(m)
    eigenwerte = v[0]
    eigenvektoren = v[1]
    idx = eigenwerte.argsort()[::-1]
    #eigenValuesSort = eigenwerte[idx]
    eigenVectorsSort = eigenvektoren[:, idx]
    tossed = eigenVectorsSort[:,:dim]
    newdata = np.dot(tossed.T, vreduced.T).T
    return newdata

def getEigenface(face,dim, u, m):
    v = face
    vreduced = np.subtract(v, u)
    v = np.linalg.eigh(m)
    eigenwerte = v[0]  # np.sort(v[0])[::-1]
    eigenvektoren = v[1]
    idx = eigenwerte.argsort()[::-1]
   # eigenValuesSort = eigenwerte[idx]
    eigenVectorsSort = eigenvektoren[:, idx]
    tossed = eigenVectorsSort[:, :dim]
    newdata = np.dot(tossed.T, vreduced.T).T
    return newdata

def Aufgabe1():
    X_train, y_train = load_from_file("zip.train/zip.train")
    X_test, y_test = load_from_file("zip.test/zip.test")

    u = calc_u(X_train)
    m = covariance_matrix_loop(X_train, u)

    X_redu = reduce_dimensions(X_train,2,u,m)
    X_test_redu = reduce_dimensions(X_test,2,u,m)

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
    return (sum(np.multiply(b,np.insert(datapoint,0,1,axis = 1))[0]))


def linregTest():
    v1 = np.array([(1,10),(2,10),(1,9),(2,9)])
    v2 = np.array([(6, 2), (6, 1), (7, 2), (7, 1)])

    b = calc_linear_regression_2D(v1,v2)
    p1 = np.array([(1.5, 9.5)])
    p2 = np.array([(6.5, 1.5)])

    print(klassify_newPoint_regression(p1,b))# -> sollte nahe 1 sein

    print(klassify_newPoint_regression(p2, b))# -> sollte nahe -1 sein


def faces_Aufgabe():
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
    u = calc_u(X)
    print("u calculated")
    m = covariance_matrix_loop(X,u)
    print("covariance calculated")
    #subtract u from each face

    X_redu = X#reduce_dimensions(X[:100], 400)



    num_samples = 90
    indices = np.random.choice(range(len(X_redu)), num_samples)
    sample_faces = X_redu[indices]

    fig = plt.figure(figsize=(20, 6))

    for i in range(num_samples):
        ax = plt.subplot(6, 15, i + 1)
        if i % 2 == 1:
            img = getEigenface(sample_faces[i-1].reshape((64, 64)),64,u,m)
        else:
            img = sample_faces[i].reshape((64, 64))
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.show()
    print("finished")

def recenter(image, min_rows, min_cols):
    r, c = image.shape
    top, bot, left, right = 0, r, 0, c
    if r > min_rows:
        top = r - min_rows
    if c > min_cols:
        right = min_cols
    return image[top:bot, left:right]

def main():
    #linregTest()
    Aufgabe1()
    #faces_Aufgabe()
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
