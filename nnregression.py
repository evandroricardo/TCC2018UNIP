from sklearn.metrics import regression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def run(x, y, xTeste, yTeste, xAmostra, yAmostra, before_plot=None, plot_title=None, **config):
    nn = MLPRegressor(**config)

    nn.fit(xAmostra, yAmostra.ravel())
    yPredTeste = nn.predict(xTeste)
    yPred = nn.predict(x)

    score_amostra = regression.explained_variance_score(yTeste, yPredTeste)
    score_todo = regression.explained_variance_score(y, yPred)

    print("Score amostra: {:02.50f}".format(score_amostra))
    print("Score todo: {:02.50f}".format(score_todo))

    if before_plot is not None:
        yPred, yPredTeste = before_plot(nn, yPred, yPredTeste, score_amostra, score_todo)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x[:, 0], y, s=1, c='b', marker="s", label='Real')
    ax1.scatter(x[:, 0], yPred, s=1, c='r', marker="s", label='Preditado Todo')
    ax1.scatter(xTeste[:, 0], yPredTeste, s=1, c='g', marker="s", label='Preditado Amostra')

    if plot_title is not None:
        plt.title(plot_title)
    plt.legend()
    plt.show()
