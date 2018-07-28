from sklearn.metrics import regression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def run(x, y, xTeste, yTeste, xAmostra, yAmostra, **config):
    nn = MLPRegressor(**config)

    nn.fit(xAmostra, yAmostra.ravel())
    yPredTeste = nn.predict(xTeste)
    yPred = nn.predict(x)

    print("Score amostra: {:02.50f}".format(regression.explained_variance_score(yTeste, yPredTeste)))
    print("Score todo: {:02.50f}".format(regression.explained_variance_score(y, yPred)))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x[:, 0], y, s=1, c='b', marker="s", label='Real')
    ax1.scatter(x[:, 0], yPred, s=1, c='r', marker="s", label='Preditado Todo')
    ax1.scatter(xTeste[:, 0], yPredTeste, s=1, c='g', marker="s", label='Preditado Amostra')

    plt.title("Preco de Compra X Taxa de Compra")

    plt.legend()
    plt.show()
