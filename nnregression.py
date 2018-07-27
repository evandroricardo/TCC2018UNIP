from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def run(csv, x, y, xAmostra, yAmostra, **config):
    nn = MLPRegressor(**config)

    n = nn.fit((xAmostra, yAmostra), yAmostra)
    pred_y = n.predict((x, y))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, s=1, c='g', marker="s", label='Real')
    ax1.scatter(x, pred_y, s=10, c='b', marker="o", label='NN Prediction')

    plt.title("Preco de Compra X Taxa de Compra")

    plt.legend()
    plt.show()
