import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Classe com metodos da regressao linear recebendo  as matrizes para tracar grafico
# amostras de treinamento considerando os dados da serie historica na matriz utilizada
# argumentos para não plotagem dos graficos e parametros de configuracao da rede
def run(csv, x, y, xAmostra, yAmostra, before_plot=None, dont_plot=False, **config):

    # Definicao das configuracoes da regressao linear  utilizando as calibragens definidas
    lnr = LinearRegression(**config)

    # Chamada do metodo de treinamento
    lnr.fit(xAmostra, yAmostra)

     # Predicao do teste
    pred = lnr.predict(x)

    # Antes da exibicao dos graficos exibe o score da amostra
    if not dont_plot:
        if before_plot is not None:
            before_plot(lnr, pred)
            
        # Plotagem das saídas

        # Plotagem da curva com valores reais 
        plt.scatter(x[:, 0], y, s=1, color='g', marker="s", label='Real')

        # Plotagem da curva com valores preditados do total da serie historica
        plt.plot(x[:, 0], pred, color='b', linewidth=3, label='Prediction')

        # Plotagem do titulo da imagem (grafico)
        plt.title("Data X Taxa de Compra")

        # Plotagem dos eixos da imagem (grafico)
        plt.xticks(())
        plt.yticks(())

        # Plotagem da legenda da imagem (grafico) 
        plt.legend()

        # Exibicao do grafico em tela
        plt.show()
    return pred
