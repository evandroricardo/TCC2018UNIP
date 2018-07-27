from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def run(csv, x, y, xTeste, yTeste, xAmostra, yAmostra, **config):
    nn = MLPRegressor(**config)

    nn.fit(xAmostra, yAmostra.ravel())
    pred_y = nn.predict(xTeste)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, s=1, c='b', marker="s", label='Real')
    ax1.scatter(xTeste, pred_y, s=1, c='g', marker="s", label='Preditado')
    # ax1.scatter(xTeste, yTeste, s=1, c='r', marker="s", label='Esperado')

    plt.title("Preco de Compra X Taxa de Compra")

    plt.legend()
    plt.show()
