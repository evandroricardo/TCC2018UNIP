import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Classe para clusterizacao recebendo como parametros a serie historica a matriz utilizada
# argumentos (quantidade de clusters) e parametros de configuracao
def run(csv, x, *args, **config):

    # Inicia a plotagem do grafico com resultados, configura a dimensao na tela
    plt.figure(figsize=(12, 12))

    # Seta o inicio de estado sempre para 0 caso nao definido
    if "random_state" not in config:
        config["random_state"] = 0

    # Chamada do metodo para predicao
    y_pred = KMeans(*args, **config).fit_predict(x)

    # Plotagem na tela
    plt.subplot(221)

    # Configuracao da exibicao do grafico
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)

    # Titulo do grafico para exibicao em tela
    plt.title("Data X Preco de Compra")

    # Carrega grafico na tela
    plt.show()
