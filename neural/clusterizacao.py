import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def run(csv, x, *args, **config):
    plt.figure(figsize=(12, 12))

    if "random_state" not in config:
        config["random_state"] = 0

    # Incorrect number of clusters
    y_pred = KMeans(*args, **config).fit_predict(x)

    plt.subplot(221)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.title("Data X Preco de Compra")

    plt.show()
