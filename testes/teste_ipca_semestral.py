from data_layer.data import CSV
from numpy import random
from sklearn.model_selection import train_test_split

# Classe para preditar o valor do titulo IPCASemestral
class TesteIPCASemestral:
    @property
    def name(self):
        return self.__class__.__name__

    # Leitura da serie historica presente no arquivo .csv
    def pega_csv(self):
        return CSV.ipca_jur_semestral

    # Funcao de clusterizacao recebendo como parametro a serie historica presente no arquivo .csv
    def clusterizacao(self, csv):
        from neural.clusterizacao import run
        # Passagem dos parametros de datga e preco unitario de compra para clusterizar
        x = csv[["data", "puCompra"]].values
        return run(csv, x, 2, random_state=0)

    # Funcao de regressao linear recebendo como parametro a serie historica presente no arquivo .csv
    def regressao_linear(self, csv):
        from neural.regressaolinear import run
        # Passagem dos parametros de data, taxa de compra e preco unitario de compra para traçar 
        # a regressao 
        X = csv[["data", "taxaCompra"]].values
        y = csv["puCompra"].values
        # Passagem dos parametros de amostra com data, taxa de compra e preco unitario de compra 
        # metodo utilizado para teste da regressao
        XAmostra = csv[["data", "taxaCompra"]].values[:200, :]
        yAmostra = csv["puCompra"].values[:200]
        return run(csv, X, y, XAmostra, yAmostra)

    # Funcao de chamada da rede neural de regressao recebendo como parametro a serie historica
    # considerando a matriz de treinamento (X_test) e inibicao da plotagem do grafico usando dont_plot
    def nn_regressao(self, csv, X_test=None, dont_plot=False):
        from neural.nnregression import run
        # Matrizes com data, taxa de compra e preco unitario de compra
        X = csv[["data", "taxaCompra"]].values
        y = csv["puCompra"].values

        # Matrizes e parametros de configurações, considerando percentual de teste (test_size)
        # bem como o shuffle setado como false para não embaralhar os valores
        X_train, _X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        if X_test is None:
            X_test = _X_test
        # Parametros de configuracao para a rede neural considerando calibragens     
        return run(
            X, y, X_test, y_test, X_train, y_train, 
            dont_plot=dont_plot,
            before_plot=self.before_plot_nn_regressao, plot_title="Dias corridos x Preco de compra",
            hidden_layer_sizes=(2000,),  activation='relu', solver='adam', learning_rate='constant', alpha=0.001, 
            batch_size='auto', learning_rate_init=0.01, power_t=0.5, max_iter=10000, shuffle=True,
            random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=False,
            early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08
        )

    # Funcao para testes de predicao do simulador para demonstrar resultados antes da plotagem de graficos
    def before_plot_nn_regressao(self, nn, yPred, yPredTeste, score_amostra, score_todo):
        preco_ipcasemestral_2015 = nn.predict([[2937, 0.10]])[0] + 3286.48
        print("Previsao preco IPCA+ Jur Semestral 2026 no vencimento: {0}; Min: {1}; Max: {2}".format(
            preco_ipcasemestral_2015,
            preco_ipcasemestral_2015 * score_todo,
            preco_ipcasemestral_2015 * (2 - score_todo)
        ))
        return yPred, yPredTeste
    
    # Funcao para predicao dos valores usando metologia determinada para o titulo
    def predict(self, data, taxa_compra, K=0):
        result = self.nn_regressao(self.pega_csv(), X_test=[[data, taxa_compra]], dont_plot=True)
        return result[-1] + K, "Rede Neural"
