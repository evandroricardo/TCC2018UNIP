import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def run(csv, x, y, xAmostra, yAmostra, before_plot=None, **config):
    lnr = LinearRegression(**config)
    lnr.fit(xAmostra, yAmostra)

    pred = lnr.predict(x)

    if before_plot is not None:
        before_plot(lnr, pred)

    # Plotagem das saídas
    plt.scatter(x[:, 0], y, s=1, color='g', marker="s", label='Real')
    plt.plot(x[:, 0], pred, color='b', linewidth=3, label='Prediction')
    plt.title("Data X Taxa de Compra")

    plt.xticks(())
    plt.yticks(())

    plt.legend()
    plt.show()
