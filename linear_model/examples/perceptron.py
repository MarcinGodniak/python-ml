from linear_model.perceptron import PerceptronWithHistory, PerceptronWithPocket, PerceptronWithPocketAndHistory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import matplotlib
matplotlib.rcParams["backend"] = "TkAgg"


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict_all(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


def download_setosa():
    df = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv')
    df_virginica = df[df['Name'] == 'Iris-virginica']
    df_setosa = df[df['Name'] == 'Iris-setosa']
    df_versicolor = df[df['Name'] == 'Iris-versicolor']

    return df_virginica, df_setosa, df_versicolor

def plots():
    virginica, setosa, versicolor = download_setosa()

    versicolor_2d = versicolor.iloc[:, [0, 2]].values
    virginica_2d = virginica.iloc[:, [0, 2]].values
    setosa_2d = setosa.iloc[:, [0, 2]].values

    plt.scatter(virginica_2d[:, 0], virginica_2d[:, 1], label='virginica')
    plt.scatter(setosa_2d[:, 0], setosa_2d[:, 1], label='setosa')
    plt.legend(loc='upper left')
    plt.draw()


    X = np.concatenate((virginica_2d, setosa_2d))
    y = np.concatenate((np.repeat(1., len(virginica_2d)), np.repeat(-1., len(virginica_2d))))

    p = PerceptronWithHistory()
    errors = p.fit(X, y)

    plt.figure()
    plt.plot(errors)
    plt.xlabel('iteration')
    plt.ylabel('error rate')
    plt.draw()

    print(errors)

    plt.figure()
    plot_decision_regions(X, y, classifier=p)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')


    plt.show()

def generalization():
    virginica, setosa, versicolor = download_setosa()
    virginica_x = virginica.iloc[:-5, :4].values
    versicolor_x = versicolor.iloc[:-5, :4].values
    X = np.concatenate(( virginica_x, versicolor_x ))
    y = np.concatenate((np.repeat(1., len(virginica_x)), np.repeat(-1., len(versicolor_x))))

    p = PerceptronWithPocketAndHistory(0.1)
    errors = p.fit(X, y)

    plt.figure()
    plt.plot(errors)
    plt.xlabel('iteration')
    plt.ylabel('error rate')
    plt.draw()

    virginica_vefiry = virginica.iloc[-5:, :4].values
    versicolor_verify = versicolor.iloc[-5:, :4].values
    X_verify = np.concatenate(( virginica_vefiry, versicolor_verify ))
    y_verify = np.concatenate((np.repeat(1., len(virginica_vefiry)), np.repeat(-1., len(versicolor_verify))))

    predicted = p.predict_all(X_verify)
    error = p.get_error(X_verify, y_verify)

    print(y_verify, predicted)
    print(error, p.best_error)

    plt.show()




generalization()