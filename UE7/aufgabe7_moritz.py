import pandas as pd
import numpy as np


# id 3 algo https://data-science-blog.com/blog/2017/12/08/id3-algorithmus-ein-rechenbeispiel/


def load_from_file(path):
    #importfunktion für daten. Übernommen aus Tutoriumsvorlage
    df = pd.read_csv(path, header=None, sep=",")
    X = df.iloc[:, 0:7].values # there is an empty string at position 257, because every line ends with a space (== separator)
    #y = df.iloc[:, 0].values
    return X


def main():
    X_train = load_from_file("car.data")
    print("hello")
  #  X_test, y_test = load_from_file("zip.test/zip.test")

if __name__ == "__main__":
    main()