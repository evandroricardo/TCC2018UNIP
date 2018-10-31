from sklearn.metrics import regression
from sklearn.neural_network import MLPRegressor

# Classe com metodos da rede neural recebendo  as matrizes para treinamento e predicao
# amostras de treinamento considerando os dados da serie historica na matriz utilizada
# argumentos para não plotagem dos graficos e parametros de configuracao da rede
def run(x, y, xTeste, yTeste, xAmostra, yAmostra, before_plot=None, plot_title=None, dont_plot=True, **config):
    
    # Definicao das configuracoes do multilayerperceptron de regrssão utilizando as calibragens da rede
    nn = MLPRegressor(**config)

    # Chamada do metodo de treinamento da rede neural 
    nn.fit(xAmostra, yAmostra.ravel())
    
    # Predicao do teste (amostra)
    yPredTeste = nn.predict(xTeste)
    
    # Predicao do valor da matriz informada (serie historica)
    yPred = nn.predict(x)

    # Extracao do score para afericao da acertividade 
    score_todo = regression.explained_variance_score(y, yPred)
    
    # Print o score na tela
    print("Score todo: {:02.50f}".format(score_todo))
    
    # Antes da exibicao dos graficos exibe o score da amostra
    if not dont_plot:
        if before_plot is not None:
            
            # Extracao do score para afericao da acertividade 
            score_amostra = regression.explained_variance_score(yTeste, yPredTeste)
            
            # Print o score na tela
            print("Score amostra: {:02.50f}".format(score_amostra))

            # Devolutiva dos parametros para variaveis (preservar os valores)
            yPred, yPredTeste = before_plot(nn, yPred, yPredTeste, score_amostra, score_todo)
        import matplotlib.pyplot as plt
        
        # Inicia abertura do metodo para a plotagem do grafico
        fig = plt.figure()

        # Adicao dos eixos e parametros a figura (grafico)
        ax1 = fig.add_subplot(111)

        # Plotagem da curva com valores reais 
        ax1.scatter(x[:, 0], y, s=1, c='b', marker="s", label='Real')

        # Plotagem da curva com valores preditados do total da serie historica
        ax1.scatter(x[:, 0], yPred, s=1, c='r', marker="s", label='Preditado Todo')

        # Plotagem da curva com valores preditados dentro da amostra gerada
        ax1.scatter(xTeste[:, 0], yPredTeste, s=1, c='g', marker="s", label='Preditado Amostra')

        # Plotagem do titulo da imagem (grafico)
        if plot_title is not None:
            plt.title(plot_title)

        # Plotagem da legenda da imagem (grafico)    
        plt.legend()

        # Exibicao do grafico em tela
        plt.show()
        
    return yPred, yPredTeste
