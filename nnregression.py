from sklearn.metrics import regression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def run(x, y, xTeste, yTeste, xAmostra, yAmostra, **config):
    nn = MLPRegressor(**config)

    nn.fit(xAmostra, yAmostra.ravel())
    yPredTeste = nn.predict(xTeste)
    yPred = nn.predict(x)

    preco_selic_2023 = nn.predict([[1674, 0.15]])[0] + 8990
    score_amostra = regression.explained_variance_score(yTeste, yPredTeste)
    score_todo = regression.explained_variance_score(y, yPred)

    print("Score amostra: {:02.50f}".format(score_amostra))
    print("Score todo: {:02.50f}".format(score_todo))
    print("Previsao preco SELIC 2023 no vencimento: {0}; Min: {1}; Max: {2}".format(
        preco_selic_2023,
        preco_selic_2023 * score_todo,
        preco_selic_2023 * (2 - score_todo)
    ))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x[:, 0], y, s=1, c='b', marker="s", label='Real')
    ax1.scatter(x[:, 0], yPred, s=1, c='r', marker="s", label='Preditado Todo')
    ax1.scatter(xTeste[:, 0], yPredTeste, s=1, c='g', marker="s", label='Preditado Amostra')
    ax1.scatter(x[:, 0], [p + 1772.37 for p in yPred], s=1, c='g', marker="s", label='Preditado Todo Deslocado')

    plt.title("Preco de Compra X data em dias")

    plt.legend()
    plt.show()
